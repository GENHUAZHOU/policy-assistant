from pathlib import Path

import pytest

from app.config import ROOT, Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        docs_dir=ROOT / "docs",
        data_dir=tmp_path / "data",
        openai_api_key="",
        database_url="",
        token_budget=8000,
        max_turns=6,
    )
