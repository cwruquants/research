"""
Parse transcript XMLs and write their companies and documents rows to Postgres.

For each XML: company metadata, then sentence and word counts, readability scores, the
LM and Garcia ML dictionary tone counts (research.features.lexicon) and, with --sentiment,
the transformer sentiment score. Rows are upserted on documents.source_file, so re-running
a folder updates it in place. Keyword exposures are a separate step: `research backfill`.

With --from-db the transcripts come from the transcript_files table instead of a folder,
always the latest published revision of each call (transcript_files_latest). With
--measures-only that is every call that has a documents row; without it, every call that
does not have one yet. Use --from-db for the rerun: a folder holds whichever revision was
saved there, often an older one.

A transcript with no spoken text gets NULL readability and tone, not zeros.

To recompute the measures of calls already in the database, pass --measures-only. A plain
re-run also rewrites company, year, title, date and city from the file, which would undo
hand fixes such as the October 2026 company cleanup. --measures-only updates only the
measure columns and the event type the file states, never creates a row or a company, and
skips files with no row. Without --sentiment it leaves the stored sentiment scores as they
are.

Every text measure uses the spoken text (every section except the participant lists, with
speaker lines removed), the same text keyword matching reads. Rows stored before 2026-11-11 used the whole XML
body and have not been recomputed (see docs/known_issues.md).

A --from-db run that was cut short can be continued: the same command with --resume picks
up the latest run made with the same code, package versions and options, writes under that
run's id and, with --measures-only, skips the calls it already wrote.

Usage (DATABASE_URL set to a role that can write companies and documents):

    uv run research ingest --input-dir <EARNINGS_CALLS>/2016 --year 2016 --threads 4
    uv run research ingest --input-dir <EARNINGS_CALLS>/2016 --year 2016 --sentiment   # needs --extra ml
    uv run research ingest --input-dir <EARNINGS_CALLS>/2016 --year 2016 --sentiment --measures-only
    uv run research ingest --from-db --measures-only --sentiment --years 2016 2017
    uv run research ingest --from-db --measures-only --sentiment --years 2016 2017 --resume
    uv run research ingest --from-db --years 2022 2023 2024   # calls with no documents row yet
    uv run research ingest --from-db --measures-only --sentiment --device 0 --workers 8

The last form is the fast one on a GPU: eight worker processes read, parse and tokenize the
transcripts, and one more process runs the transformer on GPU 0 in large batches
(research.features.sentiment_server). This process only writes the rows.
"""

import argparse
import concurrent.futures
import contextlib
import io
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import psycopg
from tqdm.auto import tqdm

from research import config
from research.db.documents import PostgresUploader
from research.db.provenance import find_matching_run, model_revision, word_list
from research.db.transcript_files import latest_transcript_names, latest_xml
from research.features.lexicon import LexiconScorer
from research.features.readability import readability_scores
from research.features.sentiment import (
    CPU_BATCH_SIZE,
    GPU_MAX_TOKENS,
    GPU_PAD_MULTIPLE,
    HF_MODEL,
    MAX_LENGTH,
    SentimentSetup,
    mean_sentiment,
    sentence_texts,
)
from research.transcripts._nltk import ensure_nltk_data
from research.transcripts.models import DocumentAttr
from research.transcripts.parse import get_company_metadata, spoken_text

_lexicon: Optional[LexiconScorer] = None


def _lexicon_scorer() -> LexiconScorer:
    """One scorer per process: it only reads its word lists, so threads can share it."""
    global _lexicon
    if _lexicon is None:
        _lexicon = LexiconScorer()
    return _lexicon


def analyze_document(path: Union[str, Path], sentiment=None, xml: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Metadata and text measures for one transcript, shaped for PostgresUploader:
    {"document": metadata, "document_attr": measures}. Pass a SentimentSetup (or anything
    with its sentence_scores, such as a RemoteSentiment) as `sentiment` to add the
    transformer sentiment score: the mean over the call's sentences. Pass the
    file's bytes as `xml` to analyze a transcript that is not on disk (`path` then only
    names it, e.g. its source_file).

    A transcript with no spoken text has num_words 0 and NULL for every score, so a missing
    measure never looks like a real zero.
    """
    path = Path(path)

    def source():
        return io.BytesIO(xml) if xml is not None else path

    company = get_company_metadata(source())
    text = spoken_text(source())
    doc = DocumentAttr(text, store_paragraphs=True, store_sentences=True, store_words=False)
    measures: Dict[str, Any] = {"num_sentences": doc.num_sentences, "num_words": doc.num_words}
    if doc.num_words:
        measures.update(readability_scores(text))
    else:
        measures.update({name: None for name in readability_scores("")})
    measures.update(_lexicon_scorer().scores(text))

    if sentiment is not None:
        measures["sentiment"] = (
            mean_sentiment(sentiment.sentence_scores(sentence_texts(doc))) if doc.num_sentences else None
        )

    metadata = {
        "file_name": path.name,
        "file_path": str(path),
        "company_name": company.get("companyName"),
        "company_ticker": company.get("companyTicker"),
        "headline_ticker": company.get("headlineTicker"),
        "event_type_id": company.get("eventTypeId"),
        "event_type": company.get("eventTypeName"),
        "start_date": company.get("startDate"),
        "quarter": company.get("quarter"),
        "year": company.get("year"),
        "city": company.get("city"),
        "event_title": company.get("eventTitle"),
    }
    return {"document": metadata, "document_attr": measures}


def _gather(
    input_dir: Path, pattern: str, recursive: bool, specific_files: Optional[Sequence[Union[str, Path]]]
) -> List[Path]:
    if specific_files is not None:
        files = []
        for path_like in specific_files:
            candidate = Path(path_like)
            if not candidate.is_file():
                raise FileNotFoundError(f"Specified file not found: {candidate}")
            files.append(candidate)
    else:
        files = [
            p for p in (input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)) if p.is_file()
        ]
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {input_dir}")
    return sorted(files)


def run_settings(
    sentiment_device: Optional[int], measures_only: bool, from_db: bool, years, input_dir, pattern: str
) -> Dict[str, Any]:
    """What a pipeline_runs row records about an ingest run.

    Sentiment is scored when sentiment_device is given (-1 for the CPU, else a GPU index).
    A GPU run batches differently from a CPU one, which moves scores by up to about 1e-6,
    so the two record different settings and never resume each other.
    """
    settings: Dict[str, Any] = {
        "source": "transcript_files_latest" if from_db else {"input_dir": str(input_dir), "pattern": pattern},
        "measures_only": measures_only,
        "years": list(years) if years else None,
        "text": "spoken_text: every section except the participant lists, speaker lines removed",
        "lexicon": {
            name: {side: word_list(path) for side, path in files.items()}
            for name, files in _lexicon_scorer().files().items()
        },
        "sentiment": None,
    }
    if sentiment_device is not None:
        settings["sentiment"] = {
            "hf_model": HF_MODEL,
            "model_revision": model_revision(HF_MODEL),
            "max_length": MAX_LENGTH,
            "document_score": "mean over sentences",
        }
        if sentiment_device >= 0:
            settings["sentiment"]["gpu_batches"] = {
                "max_tokens": GPU_MAX_TOKENS,
                "pad_multiple": GPU_PAD_MULTIPLE,
            }
        else:
            settings["sentiment"]["batch_size"] = CPU_BATCH_SIZE
    return settings


@dataclass(frozen=True)
class StoredTranscript:
    """A transcript in transcript_files, read from its latest published revision."""

    source_file: str
    folder_year: Optional[int]

    def __str__(self) -> str:
        return self.source_file


@dataclass(frozen=True)
class Analysis:
    """One transcript's analyze_document() output, and the year to store if it has no parseable date."""

    doc: Dict[str, Any]
    fallback_year: Optional[int]


Transcript = Union[Path, StoredTranscript]
# What analyzing one transcript gave: an Analysis, None when the database has no
# published revision of it, or the exception that stopped it.
Outcome = Union[Analysis, None, Exception]


def _analyze(
    path: Transcript, sentiment, year: Optional[int], fetch_xml: Callable[[str], Optional[str]]
) -> Optional[Analysis]:
    """Read and analyze one transcript, writing nothing. fetch_xml(source_file) reads a stored transcript."""
    if isinstance(path, StoredTranscript):
        stored = fetch_xml(path.source_file)
        if stored is None:
            return None
        doc = analyze_document(path.source_file, sentiment, xml=stored.encode("utf-8"))
        return Analysis(doc, path.folder_year if path.folder_year is not None else year)
    return Analysis(analyze_document(path, sentiment), year)


def _outcome(path: Transcript, sentiment, year: Optional[int], fetch_xml) -> Tuple[Transcript, Outcome]:
    try:
        return path, _analyze(path, sentiment, year, fetch_xml)
    except Exception as e:
        return path, e


@contextlib.contextmanager
def _analyzed_here(files, threads: int, analyze: Callable[[Transcript], Tuple[Transcript, Outcome]]):
    """Analyze files in this process, on `threads` threads; yields an iterator of (path, outcome)."""
    if threads == 1:
        yield map(analyze, files)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(analyze, p) for p in files]
        try:
            yield (f.result() for f in concurrent.futures.as_completed(futures))
        except BaseException:
            # --stop-on-error: drop the queued files; ones already running still finish
            executor.shutdown(cancel_futures=True)
            raise


# Per worker process, set by _init_worker
_worker_sentiment = None
_worker_year: Optional[int] = None
_worker_db_url: Optional[str] = None
_worker_conn: Optional[psycopg.Connection] = None


def _init_worker(
    lock, sentiment_device: Optional[int], server_args, torch_threads: int, db_url: str, year: Optional[int]
) -> None:
    global _worker_sentiment, _worker_year, _worker_db_url
    _worker_year, _worker_db_url = year, db_url
    # One worker at a time: four Windows workers that loaded NLTK data and the model at
    # the same moment deadlocked
    with lock:
        ensure_nltk_data()
        readability_scores("Load the syllable dictionary.")
        _lexicon_scorer()
        if sentiment_device is None:
            return
        import torch

        torch.set_num_threads(torch_threads)
        if server_args is not None:
            from research.features.sentiment_server import connect

            # CPU copy for the tokenizer; the transformer runs on the server
            _worker_sentiment = connect(SentimentSetup(-1), server_args)
        else:
            _worker_sentiment = SentimentSetup(-1)
        _worker_sentiment.sentence_scores(["Load the models once.", "This is a second sentence."])


def _worker_fetch_xml(source_file: str) -> Optional[str]:
    """latest_xml over this worker's own connection, reopened if it broke."""
    global _worker_conn
    if _worker_conn is None or _worker_conn.closed or _worker_conn.broken:
        _worker_conn = psycopg.connect(_worker_db_url, autocommit=True)
    return latest_xml(_worker_conn, [source_file]).get(source_file)


def _analyze_in_worker(path: Transcript) -> Tuple[Transcript, Outcome]:
    """_outcome in a worker process; an error comes back as a RuntimeError naming its type."""
    path, outcome = _outcome(path, _worker_sentiment, _worker_year, _worker_fetch_xml)
    if isinstance(outcome, Exception):
        outcome = RuntimeError(f"{type(outcome).__name__}: {outcome}")
    return path, outcome


@contextlib.contextmanager
def _analyzed_in_workers(
    files, workers: int, sentiment_device: Optional[int], db_url: str, year: Optional[int]
):
    """Analyze files in `workers` spawned processes; yields an iterator of (path, outcome).

    With a GPU sentiment_device the transformer runs in one ScoringServer process on that
    GPU, which the workers send token ids to; with -1 every worker holds a CPU copy.
    """
    ctx = multiprocessing.get_context("spawn")  # fork is unsafe once torch has started threads
    with contextlib.ExitStack() as stack:
        server_args = None
        if sentiment_device is not None and sentiment_device >= 0:
            from research.features.sentiment_server import ScoringServer  # needs the [ml] extra

            server_args = stack.enter_context(ScoringServer(ctx, workers, sentiment_device)).client_args()
        cpu_threads = 1 if server_args else max(1, (os.cpu_count() or 2) // workers)
        pool = stack.enter_context(
            ctx.Pool(
                workers,
                initializer=_init_worker,
                initargs=(ctx.Lock(), sentiment_device, server_args, cpu_threads, db_url, year),
            )
        )
        yield pool.imap_unordered(_analyze_in_worker, files, chunksize=1)


def ingest(
    input_dir: Optional[Union[str, Path]],
    year: Optional[int],
    database_url: Optional[str] = None,
    pattern: str = "*.xml",
    recursive: bool = False,
    skip_on_error: bool = True,
    specific_files: Optional[Sequence[Union[str, Path]]] = None,
    num_threads: int = 1,
    measures_only: bool = False,
    from_db: bool = False,
    years: Optional[Sequence[int]] = None,
    resume: bool = False,
    num_workers: int = 1,
    sentiment_device: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyze every matching XML in input_dir and upsert its companies and documents rows.
    `year` is stored only for transcripts with no parseable start date; otherwise the
    year comes from the call's start date.

    With from_db, read transcript_files instead of input_dir (see the module docstring);
    `years` then filters on documents.year (measures_only) or on the revision's year
    folder, which is also the fallback year for a call with no parseable start date.

    With resume (from_db only), continue the latest run made by the same code, package
    versions and settings: rows are written under that run's id, and with measures_only the
    calls it already wrote are skipped. With no such run, a new one starts as usual.

    sentiment_device adds the transformer sentiment score: -1 on the CPU, or a GPU index.
    Transcripts are analyzed on num_threads threads, or with num_workers > 1 in that many
    processes (num_threads is then unused); with a GPU the transformer then runs in one
    more process, in large batches. Rows are always written from this process, in the
    order the analyses finish.
    """
    url = database_url or config.database_url()
    if resume and not from_db:
        raise ValueError("resume needs from_db: only a run that reads transcript_files can be continued")
    workers = max(1, num_workers or 1)
    scores_sentiment = sentiment_device is not None
    # In worker processes each worker builds its own (see _init_worker)
    sentiment = SentimentSetup(sentiment_device) if scores_sentiment and workers == 1 else None
    settings = run_settings(sentiment_device, measures_only, from_db, years, input_dir, pattern)
    resumed_run: Optional[int] = None
    if from_db:
        with psycopg.connect(url) as conn:
            if resume:
                resumed_run = find_matching_run(conn, "ingest", settings)
            names = latest_transcript_names(
                conn,
                with_document=measures_only,
                years=years,
                unwritten_by=resumed_run,
                with_sentiment=scores_sentiment,
            )
        files: List[Transcript] = [StoredTranscript(*row) for row in names]
        if not files and resumed_run is not None:
            print(f"pipeline_runs id {resumed_run} already wrote every call; nothing left to do")
            return _summary(year, resumed_run, True, [], [], [])
        if not files:
            raise FileNotFoundError("No transcripts in transcript_files match these options")
    else:
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")
        files = _gather(input_dir, pattern, recursive, specific_files)
    threads = 1 if workers > 1 else max(1, num_threads or 1)
    # Fetch and load the NLTK data in this thread: the workers would race to download it,
    # and NLTK's lazy cmudict loader fails when several threads touch it first
    ensure_nltk_data()
    readability_scores("Load the syllable dictionary.")

    # Threads read stored transcripts through the pool too
    uploader = PostgresUploader(url, max_connections=threads)
    upload_results, errors = [], []
    try:
        if not measures_only:
            print(f"Loaded {uploader.initialize_company_cache()} existing companies")
        if resumed_run is not None:
            run_id = resumed_run
            print(f"Resuming pipeline_runs id {run_id}: {len(files)} calls left to write")
        else:
            if resume:
                print("No earlier run with this code and these options; starting a new one")
            run_id = uploader.start_run("ingest", settings)
            print(f"Recording this run as pipeline_runs id {run_id}")

        def fetch_xml(source_file: str) -> Optional[str]:
            with uploader.pool.connection() as conn:
                return latest_xml(conn, [source_file]).get(source_file)

        def write(path: Transcript, analysis: Optional[Analysis]) -> Dict[str, Any]:
            if analysis is None:
                return {"status": "skipped", "reason": "No published transcript", "file_path": str(path)}
            measures = analysis.doc["document_attr"]
            measures["measures_run_id"] = run_id
            if scores_sentiment:
                measures["sentiment_run_id"] = run_id
            if measures_only:
                result = uploader.update_measures(analysis.doc)
            else:
                result = uploader.upload_document(analysis.doc, analysis.fallback_year)
            result["file_path"] = str(path)
            return result

        if workers > 1:
            analyses = _analyzed_in_workers(files, workers, sentiment_device, url, year)
        else:
            analyses = _analyzed_here(files, threads, lambda p: _outcome(p, sentiment, year, fetch_xml))
        progress = tqdm(
            total=len(files), desc="Processing & Uploading", unit="file", dynamic_ncols=True, file=sys.stderr
        )
        with analyses as outcomes, progress:
            for path, outcome in outcomes:
                try:
                    if isinstance(outcome, Exception):
                        raise outcome
                    result = write(path, outcome)
                except Exception as e:
                    if not skip_on_error:
                        raise
                    errors.append({"file": str(path), "reason": str(e)})
                else:
                    upload_results.append(result)
                    if result["status"] != "success":
                        errors.append({"file": str(path), "reason": result.get("reason", "Unknown")})
                progress.update(1)
    finally:
        uploader.finalize()

    summary = _summary(year, run_id, resumed_run is not None, files, upload_results, errors)
    if summary["num_year_fallbacks"]:
        where = "their year folder's year" if from_db else f"year {year}"
        print(f"{summary['num_year_fallbacks']} documents had no parseable start date; stored with {where}")
    return summary


def _summary(year, run_id: int, resumed: bool, files, upload_results, errors) -> Dict[str, Any]:
    """What ingest() returns: the run, what it processed and what failed."""
    return {
        "year": year,
        "run_id": run_id,
        "resumed": resumed,
        "num_files_processed": len(files),
        "num_uploaded": sum(1 for r in upload_results if r["status"] == "success"),
        "num_year_fallbacks": sum(1 for r in upload_results if r.get("year_fallback")),
        "num_errors": len(errors),
        "upload_results": upload_results,
        "errors": errors,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", help="Folder of transcript XMLs")
    source.add_argument(
        "--from-db",
        action="store_true",
        help="Read the latest published revision of each call from transcript_files",
    )
    parser.add_argument(
        "--year", type=int, help="Year stored for transcripts with no parseable start date (with --input-dir)"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        help="With --from-db, only these years: documents.year with --measures-only, else the year folder",
    )
    parser.add_argument("--pattern", default="*.xml")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--sentiment", action="store_true", help="Also score sentiment (needs --extra ml)")
    parser.add_argument(
        "--device", type=int, default=-1, help="GPU index for the sentiment transformer; -1 (default) is CPU"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Read and analyze in this many processes (replaces --threads); with --sentiment "
        "--device 0, one more process runs the transformer on the GPU. 8 suits a 12-core CPU.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--measures-only",
        action="store_true",
        help="Only overwrite the text measures and event type of existing documents rows; never "
        "change company, year, title, date or city, and skip files that have no row",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="With --from-db, continue the latest run made with the same code, package versions "
        "and options instead of starting a new one; with --measures-only the calls it already "
        "wrote are skipped",
    )
    args = parser.parse_args(argv)
    if args.input_dir and args.year is None:
        parser.error("--input-dir needs --year")
    if args.resume and not args.from_db:
        parser.error("--resume only applies with --from-db")
    if args.years and not args.from_db:
        parser.error("--years only applies with --from-db")
    if args.device != -1 and not args.sentiment:
        parser.error("--device only applies with --sentiment")

    summary = ingest(
        args.input_dir,
        args.year,
        sentiment_device=args.device if args.sentiment else None,
        pattern=args.pattern,
        recursive=args.recursive,
        skip_on_error=not args.stop_on_error,
        num_threads=args.threads,
        measures_only=args.measures_only,
        from_db=args.from_db,
        years=args.years,
        resume=args.resume,
        num_workers=args.workers,
    )
    print(
        f"{summary['num_uploaded']} of {summary['num_files_processed']} documents uploaded, "
        f"{summary['num_errors']} errors"
    )
    for err in summary["errors"][:20]:
        print(f"  {err['file']}: {err['reason']}")


if __name__ == "__main__":
    main()
