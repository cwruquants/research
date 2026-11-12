"""
Load raw transcript XMLs from the vendor zips into transcript_files.

Reads each zip (or folder) in place and stores every file that is valid text, one row
per distinct revision: the same file name with different content is a later revision
and gets its own row, while a byte-identical second copy is dropped. macOS resource
files, empty files and files that are not valid UTF-8 are not stored; they are counted
and listed in a report.

Re-running is safe and resumes where it stopped: files already loaded from an archive
are not read again, and nothing is ever updated or deleted.

Usage (DATABASE_URL set to a role that can insert into transcript_files):

    uv run research transcripts load --source <ZIPS>/2002.zip --source <ZIPS>/2003-2015.zip

Then check the table against the same sources with `research transcripts verify`.
"""

import argparse
import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import psycopg
from tqdm import tqdm

from research import config
from research.db.transcript_files import insert_files, loaded_members
from research.transcripts.files import MACOS_JUNK, Skipped, TranscriptFile, read_source

BATCH_BYTES = 16 * 1024 * 1024
BATCH_FILES = 500


def load_source(conn, source: str, dry_run: bool = False, limit: int | None = None) -> dict:
    """Load one zip or folder. Returns counts and the files that were not stored."""
    archive = Path(source).name
    done = set() if dry_run else loaded_members(conn, archive)
    counts: Counter = Counter(already_loaded=len(done))
    skipped: List[Skipped] = []
    batch: List[TranscriptFile] = []
    batch_bytes = 0

    def flush() -> None:
        nonlocal batch, batch_bytes
        if batch and not dry_run:
            new = insert_files(conn, batch)
            counts["inserted"] += new
            counts["identical_copy"] += len(batch) - new
        batch, batch_bytes = [], 0

    with tqdm(desc=archive, unit="file") as progress:
        for item in read_source(source, already_loaded=done):
            progress.update(1)
            if isinstance(item, Skipped):
                counts[item.reason] += 1
                if item.reason != MACOS_JUNK:  # tens of thousands of these; the count is enough
                    skipped.append(item)
                continue
            counts["read"] += 1
            counts["bytes"] += item.byte_length
            batch.append(item)
            batch_bytes += item.byte_length
            if batch_bytes >= BATCH_BYTES or len(batch) >= BATCH_FILES:
                flush()
            if limit and counts["read"] >= limit:
                break
        flush()
    return {"archive": archive, "counts": counts, "skipped": skipped}


def load_sources(
    sources: List[str], database_url: str | None, limit: int | None = None
) -> Tuple[Counter, List[Skipped]]:
    """
    Load every source over one connection; with no database_url, only read and count.
    The connection is in autocommit mode so that each batch's transaction is the outermost
    one and commits when it ends: otherwise the first SELECT opens a transaction, the
    batches become savepoints inside it, and closing the connection rolls everything back.
    """
    totals: Counter = Counter()
    skipped: List[Skipped] = []
    conn = psycopg.connect(database_url, autocommit=True) if database_url else None
    try:
        for source in sources:
            result = load_source(conn, source, dry_run=conn is None, limit=limit)
            totals.update(result["counts"])
            skipped.extend(result["skipped"])
            print(
                f"{result['archive']}: " + ", ".join(f"{k}={v}" for k, v in sorted(result["counts"].items()))
            )
    finally:
        if conn is not None:
            conn.close()
    return totals, skipped


def _write_skipped(skipped: List[Skipped], report_dir: str, suffix: str) -> None:
    if not skipped:
        return
    report = Path(report_dir) / f"transcript_files_skipped_{suffix}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["archive", "member", "reason"])
        writer.writerows((s.archive, s.member, s.reason) for s in skipped)
    print(f"{len(skipped)} files not stored; see {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Zip of transcript XMLs, or a folder of them; repeat for several",
    )
    parser.add_argument("--limit", type=int, help="Read at most this many files per source (for a trial run)")
    parser.add_argument("--dry-run", action="store_true", help="Read and count the files; write nothing")
    parser.add_argument("--report-dir", default=str(config.RESULTS_DIR))
    args = parser.parse_args()

    started = time.time()
    database_url = None if args.dry_run else config.database_url()
    totals, skipped = load_sources(args.source, database_url, limit=args.limit)
    seconds = time.time() - started
    megabytes = totals["bytes"] / 1024**2
    print(
        f"Read {totals['read']} files ({megabytes:,.0f} MB) in {seconds:,.0f} s; "
        f"{totals['inserted']} new rows, {totals['identical_copy']} identical copies dropped, "
        f"{totals['already_loaded']} already loaded"
    )
    _write_skipped(skipped, args.report_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))


if __name__ == "__main__":
    main()
