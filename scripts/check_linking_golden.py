"""Check that route B and route A still give the same links on a frozen set of inputs.

Runs `research link route-b`, then `route-a` on its output, on the inputs that made the
October 2026 links, and compares both CSVs byte for byte with saved baselines. Run it
before merging any change to src/research/linking/ (AGENTS.md: refactors must not
change stored numbers). Inputs and baselines are licensed data and stay in
data/private/ (git-ignored); this script only names them.

    uv run python scripts/check_linking_golden.py            # compare with the baselines
    uv run python scripts/check_linking_golden.py --freeze   # record the inputs' hashes

--freeze writes data/private/golden/linking_inputs.json with the SHA-256 of every input
and baseline. A normal run checks those hashes first, so a replaced stocknames or LSEG
file is reported as changed input, not as a code change. A full run reads the 3.8 GB
QuoteFile and takes several minutes. On a difference it prints which columns and
statuses moved and some document ids, and exits 1.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
GOLDEN = PRIVATE / "golden"
MANIFEST = GOLDEN / "linking_inputs.json"

INPUTS = {
    "calls": PRIVATE / "derived" / "acquirer_linking" / "calls.csv",
    "docs": PRIVATE / "derived" / "crsp_linking" / "documents_export.csv.gz",
    "stocknames": PRIVATE / "crsp" / "stocknames.csv",
    "quote": PRIVATE / "lseg" / "QuoteFile.txt",
    "cusip": PRIVATE / "lseg" / "CUSIPFile.txt",
}
BASELINES = {
    "route_b": GOLDEN / "out" / "route_b.final.csv",
    "route_a": GOLDEN / "out" / "route_a.final.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes() -> dict:
    files = {**INPUTS, **BASELINES}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        sys.exit("missing: " + ", ".join(missing))
    return {name: sha256(path) for name, path in files.items()}


def link(step: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", f"research.linking.{step}", *args], check=True)


def describe(old_path: Path, new_path: Path, limit: int = 10) -> None:
    """Which rows, columns and statuses differ between two link CSVs."""
    old = pd.read_csv(old_path, dtype=str, keep_default_na=False).set_index("document_id")
    new = pd.read_csv(new_path, dtype=str, keep_default_na=False).set_index("document_id")
    if list(old.columns) != list(new.columns):
        print(f"  columns differ:\n    old {list(old.columns)}\n    new {list(new.columns)}")
        return
    print(f"  rows: {len(old):,} old, {len(new):,} new")
    print(
        f"  only in old: {len(old.index.difference(new.index)):,}; only in new: {len(new.index.difference(old.index)):,}"
    )
    both = old.index.intersection(new.index)
    old, new = old.loc[both], new.loc[both]
    changed = old.ne(new)
    rows = changed.any(axis=1)
    print(f"  rows with a changed value: {rows.sum():,}")
    per_column = changed.sum()
    print("  changed values per column:")
    print("    " + per_column[per_column > 0].to_string().replace("\n", "\n    "))
    if "status" in old.columns:
        moved = pd.crosstab(
            old.loc[rows, "status"], new.loc[rows, "status"], rownames=["old"], colnames=["new"]
        )
        print("  status, old (rows) vs new (columns), changed rows only:\n" + moved.to_string())
    print(f"  first document ids: {', '.join(rows[rows].index[:limit])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--freeze", action="store_true", help="Record the inputs' and baselines' hashes")
    ap.add_argument("--keep", help="Write the new CSVs to this folder instead of a temporary one")
    args = ap.parse_args()

    if args.freeze:
        MANIFEST.write_text(json.dumps(hashes(), indent=2) + "\n")
        print(f"Recorded {MANIFEST}")
        return
    if not MANIFEST.is_file():
        sys.exit(f"no {MANIFEST}; run with --freeze once on the inputs that made the baselines")
    expected, actual = json.loads(MANIFEST.read_text()), hashes()
    stale = [name for name in expected if expected[name] != actual.get(name)]
    if stale:
        sys.exit(f"inputs or baselines changed since --freeze: {', '.join(stale)}")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(args.keep or tmp)
        out.mkdir(parents=True, exist_ok=True)
        new = {"route_b": out / "route_b.csv", "route_a": out / "route_a.csv"}
        link(
            "route_b",
            "--calls",
            str(INPUTS["calls"]),
            "--stocknames",
            str(INPUTS["stocknames"]),
            "--out",
            str(new["route_b"]),
        )
        link(
            "route_a",
            "--docs",
            str(INPUTS["docs"]),
            "--stocknames",
            str(INPUTS["stocknames"]),
            "--quote",
            str(INPUTS["quote"]),
            "--cusip",
            str(INPUTS["cusip"]),
            "--route-b",
            str(new["route_b"]),
            "--out",
            str(new["route_a"]),
        )
        failed = False
        for name in ("route_b", "route_a"):
            if sha256(new[name]) == expected[name]:
                print(f"{name}: identical to the baseline")
            else:
                failed = True
                print(f"{name}: DIFFERS from {BASELINES[name]}")
                describe(BASELINES[name], new[name])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
