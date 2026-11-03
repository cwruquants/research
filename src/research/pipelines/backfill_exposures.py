"""
Backfill missing direct-match exposure runs into Postgres.

Finds every (document, keyword set) pair in the database that has no exposure_runs row,
locates each document's transcript by file name in the given sources (folders of XMLs
and/or zip archives, read in place), runs direct matching for all of that document's
missing keyword sets off a single load and tokenization, and writes the runs and their
keyword matches. Work is split across processes, each with its own DB connection; each
batch of documents is one transaction (runs via INSERT ... RETURNING, matches via COPY).

Re-running is safe: runs that already exist are skipped (ON CONFLICT DO NOTHING) and
their matches are not written again.

Usage (from the repo root, DATABASE_URL set to a role that can insert into
exposure_runs and keyword_matches):

    uv run research backfill --source <EARNINGS_CALLS>/2015.zip \
        --source <EARNINGS_CALLS>/2017.zip --years 2015 2017 --workers 10

The keyword sets and their matching options come from config/keyword_sets.toml.

Start with --dry-run to see how many documents were found without writing anything.
"""

import argparse
import csv
import multiprocessing
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg
from tqdm import tqdm

from research import config
from research.db.exposures import keyword_set_sizes, missing_runs, write_runs
from research.matching.direct import MatchingAgent
from research.transcripts.parse import load_document
from research.transcripts.sources import index_sources, open_transcript, resolve_location
from research.transcripts.tokenize import document_to_sentence, document_to_word

# ---- worker process state (set up once per process by _init_worker) ----
_conn: Optional[psycopg.Connection] = None
_agents: Dict[str, MatchingAgent] = {}


def _init_worker(database_url: str) -> None:
    global _conn, _agents
    _conn = psycopg.connect(database_url)
    _agents = {name: MatchingAgent(**ks.matching_options()) for name, ks in config.keyword_sets().items()}


def process_batch(
    batch: List[dict], run_timestamp: str, agents: Optional[Dict[str, MatchingAgent]] = None, conn=None
) -> dict:
    """Match one batch of documents and write it in a single transaction."""
    agents = _agents if agents is None else agents
    conn = _conn if conn is None else conn
    run_rows = []
    match_rows: Dict[Tuple[int, int], List[tuple]] = {}
    failures = []

    for doc in batch:
        with open_transcript(doc["location"]) as fh:
            document = load_document(fh)
        if not document.text:
            failures.append(
                (doc["document_id"], doc["source_file"], "transcript has no presentation or Q&A text")
            )
            continue

        # Load and tokenize once for every keyword set this document needs
        start = time.time()
        words = document_to_word(document)
        sentences = document_to_sentence(document)
        shared_seconds = time.time() - start

        for keyword_set_id, name in doc["keyword_sets"]:
            results = agents[name].match_document(
                document, doc["source_file"], document_words=words, sentences=sentences
            )

            run_rows.append(
                (
                    doc["document_id"],
                    keyword_set_id,
                    "direct",
                    None,
                    results.runtime_seconds + shared_seconds,
                    results.total_keywords_searched,
                    results.total_keywords_with_matches,
                    results.total_direct_matches,
                    results.total_cosine_matches,
                    results.total_sentences_in_document,
                    results.total_sentences_with_match,
                    results.percent_sentences_with_match,
                    run_timestamp,
                )
            )
            match_rows[(doc["document_id"], keyword_set_id)] = [
                (keyword, "direct", m.matched_text, m.context, m.position)
                for keyword, km in results.keyword_matches.items()
                for m in km.direct_matches
            ]

    runs_written, matches_written = write_runs(conn, run_rows, match_rows)
    return {
        "runs_written": runs_written,
        "runs_skipped": len(run_rows) - runs_written,
        "matches_written": matches_written,
        "failures": failures,
    }


def _load_missing(conn: psycopg.Connection, years: Optional[List[int]]) -> List[dict]:
    """Documents with at least one missing keyword-set run, with the sets each one needs."""
    sets = config.keyword_sets()
    db_sizes = keyword_set_sizes(conn, list(sets))
    for name, ks in sets.items():
        if name not in db_sizes:
            raise SystemExit(f"keyword_sets has no row named {name!r}")
        local_count = len(MatchingAgent(**ks.matching_options()).keywords_list)
        if db_sizes[name] != local_count:
            raise SystemExit(
                f"{ks.file.name} has {local_count} keywords but keyword_sets.{name} "
                f"says {db_sizes[name]}; refusing to mix word lists"
            )
    return missing_runs(conn, list(sets), years)


def _write_failures(failures: List[tuple], report_dir: str, suffix: str) -> None:
    if not failures:
        return
    report = Path(report_dir) / f"exposure_backfill_failures_{suffix}.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document_id", "source_file", "reason"])
        writer.writerows(failures)
    print(f"{len(failures)} documents not backfilled; see {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Folder of transcript XMLs or a zip of them; repeat for several",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", help="Only documents with these documents.year values"
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--batch-size", type=int, default=100, help="Documents per transaction")
    parser.add_argument("--limit", type=int, help="Process at most this many documents (for a trial run)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would run; write nothing")
    parser.add_argument("--report-dir", default=str(config.RESULTS_DIR))
    args = parser.parse_args()

    database_url = config.database_url()
    with psycopg.connect(database_url) as conn:
        missing = _load_missing(conn, args.years)

    print(f"Indexing transcripts in {len(args.source)} source(s)...")
    index = index_sources(args.source)

    todo, not_found = [], []
    for doc in missing:
        candidates = index.get(doc["source_file"])
        if not candidates:
            not_found.append((doc["document_id"], doc["source_file"], "transcript not found in sources"))
            continue
        location, problem = resolve_location(doc, candidates)
        if location is None:
            not_found.append((doc["document_id"], doc["source_file"], problem))
        else:
            todo.append({**doc, "location": location})
    # Keep each archive's members together and in order so reads stay sequential
    todo.sort(key=lambda d: d["location"])
    if args.limit:
        todo = todo[: args.limit]

    pairs = sum(len(d["keyword_sets"]) for d in todo)
    print(
        f"{len(missing)} documents missing runs, {len(todo)} to process ({pairs} runs), "
        f"{len(not_found)} with no usable transcript in the sources"
    )
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dry_run:
        _write_failures(not_found, args.report_dir, f"dryrun_{run_timestamp}")
        return

    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    totals = defaultdict(int)
    failures = list(not_found)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(database_url,),
    ) as pool:
        futures = {pool.submit(process_batch, batch, run_timestamp): batch for batch in batches}
        with tqdm(total=len(todo), unit="doc") as progress:
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                except Exception as e:  # the batch's transaction was rolled back
                    failures.extend(
                        (d["document_id"], d["source_file"], f"batch failed: {e!r}") for d in batch
                    )
                else:
                    for key in ("runs_written", "runs_skipped", "matches_written"):
                        totals[key] += result[key]
                    failures.extend(result["failures"])
                progress.update(len(batch))

    print(
        f"Wrote {totals['runs_written']} runs and {totals['matches_written']} matches; "
        f"{totals['runs_skipped']} runs already existed"
    )
    _write_failures(failures, args.report_dir, run_timestamp)


if __name__ == "__main__":
    main()
