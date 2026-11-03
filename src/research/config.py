"""Paths and settings shared by every pipeline step.

Paths are relative to the repo checkout, so run commands from anywhere inside it.
Settings come from environment variables (a `.env` file is not read automatically):

    DATABASE_URL       Postgres connection string for steps that write to the DB
    RESEARCH_DATA_DIR  where licensed data lives (default: data/private)
"""
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WORD_SETS_DIR = DATA_DIR / "word_sets"
RESULTS_DIR = PROJECT_ROOT / "results"
KEYWORD_SETS_FILE = PROJECT_ROOT / "config" / "keyword_sets.toml"


def private_data_dir() -> Path:
    """Root of the licensed data (LSEG, CRSP, WRDS caches, transcripts); never committed."""
    return Path(os.environ.get("RESEARCH_DATA_DIR", DATA_DIR / "private"))


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("Set DATABASE_URL to the Postgres connection string (see docs/setup.md).")
    return url


@dataclass(frozen=True)
class KeywordSet:
    name: str
    file: Path
    normalize_phrases: bool = False
    exclusions: Path | None = None
    exclude_safe_harbor: bool = False

    def matching_options(self) -> dict:
        """Keyword arguments for MatchingAgent."""
        return {
            "keywords_path": str(self.file),
            "normalize_phrases": self.normalize_phrases,
            "exclude_phrases_path": str(self.exclusions) if self.exclusions else None,
            "exclude_safe_harbor": self.exclude_safe_harbor,
        }


def keyword_sets(path: Path = KEYWORD_SETS_FILE) -> dict[str, KeywordSet]:
    """The keyword sets in config/keyword_sets.toml, by DB name."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    sets = {}
    for name, entry in raw.items():
        exclusions = entry.get("exclusions")
        sets[name] = KeywordSet(
            name=name,
            file=PROJECT_ROOT / entry["file"],
            normalize_phrases=entry.get("normalize_phrases", False),
            exclusions=PROJECT_ROOT / exclusions if exclusions else None,
            exclude_safe_harbor=entry.get("exclude_safe_harbor", False),
        )
    return sets
