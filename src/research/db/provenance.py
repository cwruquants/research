"""Record what made the rows a pipeline run writes, in public.pipeline_runs.

`ingest` and `backfill` call start_run() once before writing anything, then stamp every
documents row or exposure run they write with the id it returns. A run records the code
(git commit, and a hash of the source that also works where the checkout is a plain copy),
the Python and package versions, and the run's settings: source, sentiment model and its
revision, and every word list with its SHA-256.
"""

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from research import config

PACKAGES = (
    "research",
    "nltk",
    "textstat",
    "transformers",
    "torch",
    "pysentiment2",
    "psycopg",
    "pandas",
    "numpy",
)


def file_sha256(path) -> Optional[str]:
    """SHA-256 of a file's bytes, or None if there is no such file."""
    path = Path(path)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_sha256(root: Path = config.PROJECT_ROOT / "src" / "research") -> str:
    """One hash over the package's .py files (paths and contents, in a fixed order)."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(config.PROJECT_ROOT), *args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> Dict[str, Any]:
    """HEAD commit and whether src/ has uncommitted changes; both None outside a git checkout."""
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(_git("status", "--porcelain", "--", "src"))}


def package_versions(packages: Iterable[str] = PACKAGES) -> Dict[str, Any]:
    versions: Dict[str, Any] = {"python": platform.python_version(), "platform": platform.system()}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def model_revision(repo_id: str) -> Optional[str]:
    """The Hugging Face commit of a cached model, read from the local cache (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        path = try_to_load_from_cache(repo_id, "config.json")
    except Exception:
        return None
    if not isinstance(path, str):
        return None
    parts = Path(path).parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def word_list(path) -> Optional[Dict[str, Any]]:
    """A word list's path relative to the repo and its SHA-256."""
    if path is None:
        return None
    path = Path(path)
    try:
        shown = path.resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        shown = str(path)
    return {"file": shown, "sha256": file_sha256(path)}


def start_run(conn, step: str, settings: Dict[str, Any], command: Optional[str] = None) -> int:
    """Insert a pipeline_runs row for a run that is about to write, and return its id."""
    row = {
        "step": step,
        # the research CLI sets argv[0] to "research <step>"
        "command": command if command is not None else " ".join(sys.argv),
        **git_state(),
        "code_sha256": code_sha256(),
        "versions": json.dumps(package_versions()),
        "settings": json.dumps(settings, sort_keys=True, default=str),
    }
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.pipeline_runs (step, command, git_commit, git_dirty, code_sha256, versions, settings) "
            "VALUES (%(step)s, %(command)s, %(git_commit)s, %(git_dirty)s, %(code_sha256)s, "
            "%(versions)s::jsonb, %(settings)s::jsonb) RETURNING id",
            row,
        )
        return cur.fetchone()[0]
