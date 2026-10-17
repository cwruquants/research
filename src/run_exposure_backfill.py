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

    uv run python -m src.run_exposure_backfill --source <EARNINGS_CALLS>/2015.zip \
        --source <EARNINGS_CALLS>/2017.zip --years 2015 2017 --workers 10

Start with --dry-run to see how many documents were found without writing anything.
"""
import argparse
import csv
import multiprocessing
import os
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg
from tqdm import tqdm

from src.analysis.match_extraction.matching_agent import MatchingAgent
from src.document.decompose_text import document_to_sentence, document_to_word
from src.document.decompose_transcript import get_company_metadata, load_document

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# keyword_sets.name -> word list the existing runs were made with
KEYWORD_SET_FILES = {
    "risk": PROJECT_ROOT / "data" / "word_sets" / "Risk.csv",
    "activeness": PROJECT_ROOT / "data" / "word_sets" / "activeness.csv",
}

MISSING_RUNS_SQL = """
    SELECT d.id, d.source_file, d.year, d.event_title, ks.id, ks.name
    FROM public.documents d
    CROSS JOIN public.keyword_sets ks
    WHERE ks.name = ANY(%(names)s)
        AND (%(years)s::int[] IS NULL OR d.year = ANY(%(years)s::int[]))
        AND d.source_file IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM public.exposure_runs er
            WHERE er.document_id = d.id AND er.keyword_set_id = ks.id
        )
    ORDER BY d.id, ks.id
"""

INSERT_RUNS_SQL = """
    INSERT INTO public.exposure_runs (
        document_id, keyword_set_id, matching_method, cosine_threshold, runtime_seconds,
        total_keywords_searched, total_keywords_with_matches, total_direct_matches,
        total_cosine_matches, total_sentences_in_document, total_sentences_with_match,
        pct_sentences_with_match, run_timestamp
    )
    SELECT * FROM unnest(
        %s::bigint[], %s::bigint[], %s::text[], %s::float8[], %s::float8[],
        %s::int[], %s::int[], %s::int[],
        %s::int[], %s::int[], %s::int[],
        %s::float8[], %s::text[]
    )
    ON CONFLICT (document_id, keyword_set_id) DO NOTHING
    RETURNING id, document_id, keyword_set_id
"""

COPY_MATCHES_SQL = """
    COPY public.keyword_matches (exposure_run_id, keyword, match_type, matched_text, context, position)
    FROM STDIN
"""

# A transcript location: ("dir", file path, None) or ("zip", archive path, member name)
Location = Tuple[str, str, Optional[str]]


def index_sources(sources: List[str]) -> Dict[str, List[Location]]:
    """Map transcript file name -> every location holding a file with that name."""
    index: Dict[str, List[Location]] = defaultdict(list)
    for source in sources:
        path = Path(source)
        if path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    name = member.rsplit("/", 1)[-1]
                    if name.lower().endswith(".xml"):
                        index[name].append(("zip", str(path), member))
        elif path.is_dir():
            for xml_path in path.rglob("*.xml"):
                index[xml_path.name].append(("dir", str(xml_path), None))
        else:
            raise SystemExit(f"Source is neither a folder nor a zip file: {source}")
    return index


def _folder_year(location: Location) -> Optional[int]:
    """The innermost year-named folder in a transcript's path (e.g. .../2021/123_T.xml -> 2021)."""
    kind, path, member = location
    parts = (member if kind == "zip" else path).replace("\\", "/").split("/")[:-1]
    years = [int(p) for p in parts if len(p) == 4 and p.isdigit()]
    return years[-1] if years else None


def _normalize_title(title: Optional[str]) -> str:
    return " ".join((title or "").split()).casefold()


def _resolve_location(doc: dict, candidates: List[Location]) -> Tuple[Optional[Location], str]:
    """
    Pick the transcript for a document when its file name is in more than one folder
    (the same name in two year folders is usually a different transcript): the one whose
    eventTitle matches documents.event_title, then the one in the document's year folder.
    """
    if len(candidates) == 1:
        return candidates[0], ""
    want = _normalize_title(doc["event_title"])
    if want:
        titled = []
        for location in candidates:
            with _open_transcript(location) as fh:
                if _normalize_title(get_company_metadata(fh)["eventTitle"]) == want:
                    titled.append(location)
        if len(titled) == 1:
            return titled[0], ""
        candidates = titled or candidates
    same_year = [loc for loc in candidates if _folder_year(loc) == doc["year"]]
    if len(same_year) == 1:
        return same_year[0], ""
    return None, f"{len(candidates)} transcripts named {doc['source_file']}; none uniquely matches its title or year"


# ---- worker process state (set up once per process by _init_worker) ----
_conn: Optional[psycopg.Connection] = None
_agents: Dict[str, MatchingAgent] = {}
_archives: Dict[str, zipfile.ZipFile] = {}


def _init_worker(database_url: str) -> None:
    global _conn, _agents
    _conn = psycopg.connect(database_url)
    _agents = {name: MatchingAgent(keywords_path=str(path)) for name, path in KEYWORD_SET_FILES.items()}


def _open_transcript(location: Location):
    kind, path, member = location
    if kind == "zip":
        if path not in _archives:
            _archives[path] = zipfile.ZipFile(path)
        return _archives[path].open(member)
    return open(path, "rb")


def _process_batch(batch: List[dict], run_timestamp: str) -> dict:
    """Match one batch of documents and write it in a single transaction."""
    run_rows = []
    match_rows: Dict[Tuple[int, int], List[tuple]] = {}
    failures = []

    for doc in batch:
        with _open_transcript(doc["location"]) as fh:
            document = load_document(fh)
        if not document.text:
            failures.append((doc["document_id"], doc["source_file"], "transcript has no presentation or Q&A text"))
            continue

        # Load and tokenize once for every keyword set this document needs
        start = time.time()
        words = document_to_word(document)
        sentences = document_to_sentence(document)
        shared_seconds = time.time() - start

        for keyword_set_id, name in doc["keyword_sets"]:
            agent = _agents[name]
            agent.document = document
            agent.document_name = doc["source_file"]
            results = agent._direct_match(document_words=words, sentences=sentences)

            run_rows.append((
                doc["document_id"], keyword_set_id, "direct", None,
                results.runtime_seconds + shared_seconds,
                results.total_keywords_searched, results.total_keywords_with_matches,
                results.total_direct_matches, results.total_cosine_matches,
                results.total_sentences_in_document, results.total_sentences_with_match,
                results.percent_sentences_with_match, run_timestamp,
            ))
            match_rows[(doc["document_id"], keyword_set_id)] = [
                (keyword, "direct", m.matched_text, m.context, m.position)
                for keyword, km in results.keyword_matches.items()
                for m in km.direct_matches
            ]

    runs_written = matches_written = 0
    if run_rows:
        columns = [list(col) for col in zip(*run_rows)]
        with _conn.transaction():
            with _conn.cursor() as cur:
                cur.execute(INSERT_RUNS_SQL, columns)
                new_runs = cur.fetchall()
                with cur.copy(COPY_MATCHES_SQL) as copy:
                    for run_id, document_id, keyword_set_id in new_runs:
                        for row in match_rows[(document_id, keyword_set_id)]:
                            copy.write_row((run_id, *row))
                            matches_written += 1
        runs_written = len(new_runs)

    return {
        "runs_written": runs_written,
        "runs_skipped": len(run_rows) - runs_written,
        "matches_written": matches_written,
        "failures": failures,
    }


def _load_missing(conn: psycopg.Connection, years: Optional[List[int]]) -> List[dict]:
    """Documents with at least one missing keyword-set run, with the sets each one needs."""
    with conn.cursor() as cur:
        cur.execute("SELECT name, total_keywords FROM public.keyword_sets WHERE name = ANY(%s)",
                    [list(KEYWORD_SET_FILES)])
        db_sets = dict(cur.fetchall())
        for name, path in KEYWORD_SET_FILES.items():
            if name not in db_sets:
                raise SystemExit(f"keyword_sets has no row named {name!r}")
            local_count = len(MatchingAgent(keywords_path=str(path)).keywords_list)
            if db_sets[name] != local_count:
                raise SystemExit(f"{path.name} has {local_count} keywords but keyword_sets.{name} "
                                 f"says {db_sets[name]}; refusing to mix word lists")

        cur.execute(MISSING_RUNS_SQL, {"names": list(KEYWORD_SET_FILES), "years": years})
        docs: Dict[int, dict] = {}
        for document_id, source_file, year, event_title, keyword_set_id, name in cur.fetchall():
            doc = docs.setdefault(document_id, {"document_id": document_id, "source_file": source_file,
                                                "year": year, "event_title": event_title,
                                                "keyword_sets": []})
            doc["keyword_sets"].append((keyword_set_id, name))
    return list(docs.values())


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
    parser.add_argument("--source", action="append", required=True,
                        help="Folder of transcript XMLs or a zip of them; repeat for several")
    parser.add_argument("--years", type=int, nargs="+", help="Only documents with these documents.year values")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--batch-size", type=int, default=100, help="Documents per transaction")
    parser.add_argument("--limit", type=int, help="Process at most this many documents (for a trial run)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would run; write nothing")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    database_url = os.environ["DATABASE_URL"]
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
        location, problem = _resolve_location(doc, candidates)
        if location is None:
            not_found.append((doc["document_id"], doc["source_file"], problem))
        else:
            todo.append({**doc, "location": location})
    # Keep each archive's members together and in order so reads stay sequential
    todo.sort(key=lambda d: d["location"])
    if args.limit:
        todo = todo[:args.limit]

    pairs = sum(len(d["keyword_sets"]) for d in todo)
    print(f"{len(missing)} documents missing runs, {len(todo)} to process ({pairs} runs), "
          f"{len(not_found)} with no usable transcript in the sources")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dry_run:
        _write_failures(not_found, args.report_dir, f"dryrun_{run_timestamp}")
        return

    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    totals = defaultdict(int)
    failures = list(not_found)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(database_url,)) as pool:
        futures = {pool.submit(_process_batch, batch, run_timestamp): batch for batch in batches}
        with tqdm(total=len(todo), unit="doc") as progress:
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    result = future.result()
                except Exception as e:  # the batch's transaction was rolled back
                    failures.extend((d["document_id"], d["source_file"], f"batch failed: {e!r}") for d in batch)
                else:
                    for key in ("runs_written", "runs_skipped", "matches_written"):
                        totals[key] += result[key]
                    failures.extend(result["failures"])
                progress.update(len(batch))

    print(f"Wrote {totals['runs_written']} runs and {totals['matches_written']} matches; "
          f"{totals['runs_skipped']} runs already existed")
    _write_failures(failures, args.report_dir, run_timestamp)


if __name__ == "__main__":
    main()
