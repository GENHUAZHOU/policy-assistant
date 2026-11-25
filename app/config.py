"""集中配置：环境变量优先，缺省值保证项目开箱可运行。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _resolve_path(value: str, default: Path) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else ROOT / path


@dataclass
class Settings:
    """运行时配置。测试可用 dataclasses.replace 覆盖局部字段。"""

    docs_dir: Path = field(default_factory=lambda: _resolve_path(_env("DOCS_DIR"), ROOT / "docs"))
    data_dir: Path = field(default_factory=lambda: _resolve_path(_env("DATA_DIR"), ROOT / "data"))

    database_url: str = field(default_factory=lambda: _env("DATABASE_URL"))

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model_name: str = field(default_factory=lambda: _env("MODEL_NAME", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small"))
    embedding_dim_override: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 0))

    approval_threshold: float = field(default_factory=lambda: _env_float("APPROVAL_THRESHOLD", 2000.0))
    token_budget: int = field(default_factory=lambda: _env_int("TOKEN_BUDGET", 8000))
    max_turns: int = field(default_factory=lambda: _env_int("MAX_TURNS", 6))
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 4))
    prompt_top_k: int = field(default_factory=lambda: _env_int("PROMPT_TOP_K", 2))

    langfuse_public_key: str = field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str = field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY"))
    langfuse_host: str = field(default_factory=lambda: _env("LANGFUSE_HOST", "http://localhost:3000"))
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL"))

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def audit_log(self) -> Path:
        return self.data_dir / "audit.jsonl"

    @property
    def approval_log(self) -> Path:
        return self.data_dir / "approvals.jsonl"

    @property
    def business_log(self) -> Path:
        return self.data_dir / "business.jsonl"

    @property
    def llm_mode(self) -> str:
        return "openai" if self.openai_api_key else "local"

    @property
    def embedding_dim(self) -> int:
        if self.embedding_dim_override:
            return self.embedding_dim_override
        return 1536 if self.openai_api_key else 256

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)


def get_settings() -> Settings:
    return Settings()
