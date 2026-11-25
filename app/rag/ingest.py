"""制度文档解析、按条款切分与入库。"""
from __future__ import annotations

import argparse
import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings
from app.rag.store import Chunk, VectorStore, create_vector_store

SUPPORTED = {".md", ".txt", ".pdf", ".docx"}
CLAUSE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(第[一二三四五六七八九十百千零〇\d]+条|\d+(?:\.\d+)*[\s、.]|[-*]\s)",
    re.MULTILINE,
)


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    end = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = index
            break
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip("\"'")
    if end is None:
        return {}, text
    return meta, "\n".join(lines[end + 1 :]).lstrip()


def _load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        from docx import Document

        return "\n".join(paragraph.text for paragraph in Document(str(path)).paragraphs)
    raise ValueError(f"不支持的文档类型：{suffix}")


def load_documents(docs_dir: Path) -> list[dict[str, Any]]:
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return []
    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        text = _load_text(path)
        if not text.strip():
            continue
        meta, body = _front_matter(text)
        documents.append(
            {
                "doc_id": path.stem,
                "text": body,
                "metadata": {
                    "doc_id": path.stem,
                    "source_file": str(path),
                    "version": meta.get("version", "v1.0"),
                    "effective_date": meta.get("effective_date", date.fromtimestamp(path.stat().st_mtime).isoformat()),
                },
            }
        )
    return documents


def split_clauses(document: dict[str, Any], chunk_size: int = 300, overlap: int = 50) -> list[Chunk]:
    text = re.sub(r"\n{3,}", "\n\n", document["text"]).strip()
    matches = list(CLAUSE_RE.finditer(text))
    boundaries = [m.start() for m in matches] + [len(text)]
    chunks: list[Chunk] = []

    for i, start in enumerate(boundaries[:-1]):
        end = boundaries[i + 1]
        segment = text[start:end].strip()
        if not segment:
            continue
        clause = matches[i].group(1).strip() if i < len(matches) else "前言"
        metadata = {**document["metadata"], "clause": clause}
        if i < len(matches):
            segment = re.sub(r"^#{1,6}\s*[^\n]*\n", "", segment, count=1).strip()
            if not segment:
                continue

        if len(segment) <= chunk_size:
            pieces = [segment]
        else:
            pieces = []
            cursor = 0
            while cursor < len(segment):
                pieces.append(segment[cursor : cursor + chunk_size])
                if cursor + chunk_size >= len(segment):
                    break
                cursor += chunk_size - overlap

        for piece_index, piece in enumerate(pieces):
            content = piece if piece.startswith(clause) else f"{clause} {piece}"
            chunk_id = hashlib.md5(
                f"{document['doc_id']}|{clause}|{piece_index}|{content}".encode("utf-8")
            ).hexdigest()
            chunks.append({"id": chunk_id, "text": content, "metadata": metadata})
    return chunks


def ingest(
    settings: Settings,
    embeddings: Embeddings,
    store: VectorStore | None = None,
    chunk_size: int = 300,
    overlap: int = 50,
) -> int:
    """重新构建索引：解析 -> 切分 -> 向量化 -> 写入。"""
    store = store or create_vector_store(settings)
    documents = load_documents(settings.docs_dir)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(split_clauses(document, chunk_size=chunk_size, overlap=overlap))
    if not chunks:
        return 0
    vectors = embeddings.embed_documents([chunk["text"] for chunk in chunks])
    store.clear()
    store.add(chunks, vectors)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="制度文档入库")
    parser.add_argument("--docs", help="文档目录，覆盖 DOCS_DIR")
    parser.add_argument("--index", help="索引目录，覆盖 DATA_DIR/index（仅本地存储）")
    args = parser.parse_args()

    settings = get_settings()
    if args.docs:
        settings = settings.with_overrides(docs_dir=Path(args.docs))
    if args.index:
        from app.rag.store import LocalVectorStore

        settings = settings.with_overrides(data_dir=Path(args.index).parent)
        store: VectorStore | None = LocalVectorStore(Path(args.index))
    else:
        store = None

    from app.models import build_embeddings

    count = ingest(settings, build_embeddings(settings), store)
    print(f"已入库 {count} 个条款片段 -> {settings.docs_dir}")


if __name__ == "__main__":
    main()
