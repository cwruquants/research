import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRANSCRIPTS = FIXTURES / "transcripts"


@pytest.fixture
def transcripts() -> Path:
    """Folder of synthetic transcript XMLs (see tests/fixtures/make_transcripts.py)."""
    return TRANSCRIPTS


@pytest.fixture
def test_database_url() -> str:
    """A disposable Postgres for tests marked db. Never the real research database."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    if url.rstrip("/").rsplit("/", 1)[-1].split("?")[0] == "research":
        pytest.fail("TEST_DATABASE_URL points at the research database; use a throwaway database")
    return url
