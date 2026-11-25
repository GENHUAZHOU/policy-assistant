"""向量存储抽象：本地 JSON+npy（默认）与 pgvector（配置 DATABASE_URL 时）。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np

Chunk = dict[str, Any]


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]: ...

    def all(self) -> list[Chunk]: ...

    def clear(self) -> None: ...

    def count(self) -> int: ...


class LocalVectorStore:
    """零依赖向量库：chunks.json + vectors.npy，适合单机/演示。"""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_path = self.index_dir / "chunks.json"
        self._vectors_path = self.index_dir / "vectors.npy"

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._chunks_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        matrix = np.asarray(vectors, dtype=np.float32) if vectors else np.empty((0, 0), dtype=np.float32)
        np.save(self._vectors_path, matrix)

    def all(self) -> list[Chunk]:
        if not self._chunks_path.exists():
            return []
        return json.loads(self._chunks_path.read_text(encoding="utf-8"))

    def search(self, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        chunks = self.all()
        if not chunks or not self._vectors_path.exists():
            return []
        matrix = np.load(self._vectors_path)
        query = np.asarray(vector, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != query.shape[0] or len(chunks) != matrix.shape[0]:
            return []
        scores = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-9)
        order = np.argsort(-scores)[:top_k]
        return [(chunks[int(i)], float(scores[int(i)])) for i in order]

    def clear(self) -> None:
        self._chunks_path.unlink(missing_ok=True)
        self._vectors_path.unlink(missing_ok=True)

    def count(self) -> int:
        return len(self.all())


class PgVectorStore:
    """pgvector 实现：与 checkpoint 共用同一个 Postgres 实例。"""

    TABLE = "policy_chunks"

    def __init__(self, dsn: str, dim: int) -> None:
        self.dsn = dsn
        self.dim = dim
        self.setup()

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    def setup(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.dim}) NOT NULL
                )
                """
            )

    @staticmethod
    def _literal(vector: list[float]) -> str:
        if not all(math.isfinite(float(x)) for x in vector):
            raise ValueError("embedding contains non-finite value")
        return "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"

    @staticmethod
    def _chunk(row: tuple[Any, ...]) -> Chunk:
        metadata = row[2]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return {"id": row[0], "text": row[1], "metadata": metadata}

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            for chunk, vector in zip(chunks, vectors):
                cur.execute(
                    f"""
                    INSERT INTO {self.TABLE} (id, content, metadata, embedding)
                    VALUES (%s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        chunk["id"],
                        chunk["text"],
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                        self._literal(vector),
                    ),
                )

    def all(self) -> list[Chunk]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id, content, metadata FROM {self.TABLE}")
            return [self._chunk(row) for row in cur.fetchall()]

    def search(self, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        literal = self._literal(vector)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self.TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (literal, literal, top_k),
            )
            rows = cur.fetchall()
        return [(self._chunk(row), float(row[3])) for row in rows]

    def clear(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.TABLE}")

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.TABLE}")
            return int(cur.fetchone()[0])


def create_vector_store(settings) -> VectorStore:
    """优先 pgvector；连接失败时降级到本地索引，避免服务整体不可用。"""
    if settings.database_url:
        try:
            return PgVectorStore(settings.database_url, settings.embedding_dim)
        except Exception as exc:  # pragma: no cover - 取决于外部数据库
            print(f"[warn] pgvector 不可用，降级到本地索引：{exc}")
    return LocalVectorStore(settings.index_dir)
