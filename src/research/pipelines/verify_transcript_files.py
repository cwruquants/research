"""
Check transcript_files against the zips it was loaded from.

Three checks, all of which must pass:

1. Every file in the sources that can be stored has a row with the same name and hash.
2. No row says it came from these sources unless the sources hold that file.
3. Every row's stored text still hashes to its stored SHA-256, so the text reproduces
   the original file byte for byte.

Exits with status 1 if any check fails.

Usage (DATABASE_URL set to a role that can read transcript_files):

    uv run research transcripts verify --source <ZIPS>/2002.zip --source <ZIPS>/2003-2015.zip
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Set, Tuple

import psycopg
from tqdm import tqdm

from research import config
from research.db.transcript_files import changed_rows, rows_loaded_from, stored_revisions
from research.transcripts.files import Skipped, read_source


def revisions_in(sources: List[str]) -> Tuple[Set[Tuple[str, bytes]], Counter]:
    """(source_file, sha256) of every storable file in the sources, and counts of what they hold."""
    revisions: Set[Tuple[str, bytes]] = set()
    counts: Counter = Counter()
    for source in sources:
        for item in tqdm(read_source(source), desc=Path(source).name, unit="file"):
            if isinstance(item, Skipped):
                counts[item.reason] += 1
            else:
                counts["storable files"] += 1
                revisions.add((item.source_file, item.sha256))
    return revisions, counts


def verify(conn, sources: List[str]) -> dict:
    """Run the three checks; `ok` is True only if all pass."""
    in_sources, counts = revisions_in(sources)
    archives = [Path(s).name for s in sources]
    missing = in_sources - stored_revisions(conn)
    unexpected = rows_loaded_from(conn, archives) - in_sources
    changed = changed_rows(conn, archives)
    return {
        "counts": counts,
        "distinct_revisions": len(in_sources),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "changed": changed,
        "ok": not missing and not unexpected and changed == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source", action="append", required=True, help="The zips or folders the table was loaded from"
    )
    args = parser.parse_args()

    with psycopg.connect(config.database_url()) as conn:
        result = verify(conn, args.source)

    print("In the sources: " + ", ".join(f"{k}={v}" for k, v in sorted(result["counts"].items())))
    print(f"Distinct revisions (file name and content) in the sources: {result['distinct_revisions']}")
    print(f"1. Revisions in the sources with no row: {result['missing']}")
    print(f"2. Rows from these sources that the sources do not hold: {result['unexpected']}")
    print(f"3. Rows whose stored text no longer matches its hash: {result['changed']}")
    print("All checks passed" if result["ok"] else "FAILED")
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
