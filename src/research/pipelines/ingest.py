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
measure columns, never creates a row or a company, and skips files with no row. Without
--sentiment it leaves the stored sentiment scores as they are.

Every text measure uses the spoken text (Presentation and Q&A with speaker lines removed),
the same text keyword matching reads. Rows stored before 2026-11-11 used the whole XML
body and have not been recomputed (see docs/known_issues.md).

Usage (DATABASE_URL set to a role that can write companies and documents):

    uv run research ingest --input-dir <EARNINGS_CALLS>/2016 --year 2016 --threads 4
    uv run research ingest --input-dir <EARNINGS_CALLS>/2016 --year 2016 --sentiment   # needs --extra ml
    uv run research ingest --input-dir <EARNINGS_CALLS>/2016 --year 2016 --sentiment --measures-only
    uv run research ingest --from-db --measures-only --sentiment --years 2016 2017
    uv run research ingest --from-db --years 2022 2023 2024   # calls with no documents row yet
"""

import argparse
import concurrent.futures
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import psycopg
from tqdm.auto import tqdm

from research import config
from research.db.documents import PostgresUploader
from research.db.provenance import model_revision, word_list
from research.db.transcript_files import latest_transcript_names, latest_xml
from research.features.lexicon import LexiconScorer
from research.features.readability import readability_scores
from research.transcripts._nltk import ensure_nltk_data
from research.transcripts.models import DocumentAttr
from research.transcripts.parse import get_company_metadata, spoken_text

DEFAULT_SENTIMENT = {
    "sheet_name_positive": "ML_positive_unigram",
    "sheet_name_negative": "ML_negative_unigram",
    "file_path": str(config.WORD_SETS_DIR / "original_xlsx" / "Garcia_MLWords.xlsx"),
    "hf_model": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "device": -1,
    "batch_size": "auto",
    "max_length": 512,
}


def default_sentiment_setup(overrides: Optional[Dict[str, Any]] = None):
    """A SentimentSetup with the settings the stored scores were made with."""
    from research.features.sentiment import SentimentSetup  # needs the [ml] extra

    cfg = {**DEFAULT_SENTIMENT, **(overrides or {})}
    return SentimentSetup(
        sheet_name_positive=cfg["sheet_name_positive"],
        sheet_name_negative=cfg["sheet_name_negative"],
        ml_wordlist_path=cfg["file_path"],
        hf_model=cfg["hf_model"],
        device=cfg["device"],
        batch_size=cfg["batch_size"],
        max_length=cfg["max_length"],
    )


_lexicon: Optional[LexiconScorer] = None


def _lexicon_scorer() -> LexiconScorer:
    """One scorer per process: it only reads its word lists, so threads can share it."""
    global _lexicon
    if _lexicon is None:
        _lexicon = LexiconScorer()
    return _lexicon


def _mean_sentence_sentiment(doc: DocumentAttr) -> Optional[float]:
    """
    The mean transformer score over every sentence of the call. Each sentence counts once,
    so a long Q&A weighs more than a short Presentation. (fit_all's own document score is
    the mean of its paragraph means, and real transcripts have no blank lines between
    turns, so that was the plain average of the Presentation's and the Q&A's scores.)
    """
    scores = [s.sentiment for p in doc.paragraphs for s in p.sentences if s.sentiment is not None]
    return float(sum(scores) / len(scores)) if scores else None


def analyze_document(path: Union[str, Path], sentiment=None, xml: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Metadata and text measures for one transcript, shaped for PostgresUploader:
    {"document": metadata, "document_attr": measures, "doc_fit": fitted DocumentAttr or None}.
    Pass a SentimentSetup as `sentiment` to add the transformer sentiment score. Pass the
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

    doc_fit = None
    if sentiment is not None:
        if doc.num_sentences:
            doc_fit = sentiment.fit_all(doc)
            measures["sentiment"] = _mean_sentence_sentiment(doc_fit)
        else:
            measures["sentiment"] = None

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
    return {"document": metadata, "document_attr": measures, "doc_fit": doc_fit}


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
    sentiment, measures_only: bool, from_db: bool, years, input_dir, pattern: str
) -> Dict[str, Any]:
    """What a pipeline_runs row records about an ingest run."""
    settings: Dict[str, Any] = {
        "source": "transcript_files_latest" if from_db else {"input_dir": str(input_dir), "pattern": pattern},
        "measures_only": measures_only,
        "years": list(years) if years else None,
        "text": "spoken_text: Presentation and Q&A, speaker lines removed",
        "lexicon": {
            name: {side: word_list(path) for side, path in files.items()}
            for name, files in _lexicon_scorer().files().items()
        },
        "sentiment": None,
    }
    if sentiment is not None:
        model = DEFAULT_SENTIMENT["hf_model"]
        settings["sentiment"] = {
            "hf_model": model,
            "model_revision": model_revision(model),
            "max_length": getattr(sentiment, "max_length", DEFAULT_SENTIMENT["max_length"]),
            "batch_size": getattr(sentiment, "batch_size", DEFAULT_SENTIMENT["batch_size"]),
            "document_score": "mean over sentences",
        }
    return settings


@dataclass(frozen=True)
class StoredTranscript:
    """A transcript in transcript_files, read from its latest published revision."""

    source_file: str
    folder_year: Optional[int]

    def __str__(self) -> str:
        return self.source_file


def ingest(
    input_dir: Optional[Union[str, Path]],
    year: Optional[int],
    database_url: Optional[str] = None,
    sentiment=None,
    pattern: str = "*.xml",
    recursive: bool = False,
    skip_on_error: bool = True,
    specific_files: Optional[Sequence[Union[str, Path]]] = None,
    num_threads: int = 1,
    measures_only: bool = False,
    from_db: bool = False,
    years: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Analyze every matching XML in input_dir and upsert its companies and documents rows.
    `year` is stored only for transcripts with no parseable start date; otherwise the
    year comes from the call's start date.

    With from_db, read transcript_files instead of input_dir (see the module docstring);
    `years` then filters on documents.year (measures_only) or on the revision's year
    folder, which is also the fallback year for a call with no parseable start date.
    """
    url = database_url or config.database_url()
    if from_db:
        with psycopg.connect(url) as conn:
            names = latest_transcript_names(conn, with_document=measures_only, years=years)
        files: List[Union[Path, StoredTranscript]] = [StoredTranscript(*row) for row in names]
        if not files:
            raise FileNotFoundError("No transcripts in transcript_files match these options")
    else:
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")
        files = _gather(input_dir, pattern, recursive, specific_files)
    threads = max(1, num_threads or 1)
    # Fetch and load the NLTK data in this thread: the workers would race to download it,
    # and NLTK's lazy cmudict loader fails when several threads touch it first
    ensure_nltk_data()
    readability_scores("Load the syllable dictionary.")

    uploader = PostgresUploader(url, max_connections=threads)
    upload_results, errors = [], []
    try:
        if not measures_only:
            print(f"Loaded {uploader.initialize_company_cache()} existing companies")
        run_id = uploader.start_run(
            "ingest", run_settings(sentiment, measures_only, from_db, years, input_dir, pattern)
        )
        print(f"Recording this run as pipeline_runs id {run_id}")

        def process(path: Union[Path, StoredTranscript]) -> Dict[str, Any]:
            fallback_year = year
            if isinstance(path, StoredTranscript):
                with uploader.pool.connection() as conn:
                    stored = latest_xml(conn, [path.source_file]).get(path.source_file)
                if stored is None:
                    return {"status": "skipped", "reason": "No published transcript", "file_path": str(path)}
                data = {
                    "doc_attrs": analyze_document(path.source_file, sentiment, xml=stored.encode("utf-8"))
                }
                fallback_year = path.folder_year if path.folder_year is not None else year
            else:
                data = {"doc_attrs": analyze_document(path, sentiment)}
            attrs = data["doc_attrs"]["document_attr"]
            attrs["measures_run_id"] = run_id
            if sentiment is not None:
                attrs["sentiment_run_id"] = run_id
            if measures_only:
                result = uploader.update_measures(data)
            else:
                result = uploader.upload_document(data, fallback_year)
            result["file_path"] = str(path)
            return result

        def record(path: Path, result: Optional[Dict[str, Any]], error: Optional[Exception]) -> None:
            if error is not None:
                if not skip_on_error:
                    raise error
                errors.append({"file": str(path), "reason": str(error)})
                return
            upload_results.append(result)
            if result["status"] != "success":
                errors.append({"file": str(path), "reason": result.get("reason", "Unknown")})

        progress = dict(
            total=len(files), desc="Processing & Uploading", unit="file", dynamic_ncols=True, file=sys.stderr
        )
        if threads > 1:
            with (
                concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor,
                tqdm(**progress) as bar,
            ):
                futures = {executor.submit(process, p): p for p in files}
                try:
                    for future in concurrent.futures.as_completed(futures):
                        error = future.exception()
                        record(futures[future], None if error else future.result(), error)
                        bar.update(1)
                except BaseException:
                    # --stop-on-error: drop the queued files; ones already running still finish
                    executor.shutdown(cancel_futures=True)
                    raise
        else:
            for path in tqdm(files, **progress):
                try:
                    result, error = process(path), None
                except Exception as e:
                    result, error = None, e
                record(path, result, error)
    finally:
        uploader.finalize()

    successful = sum(1 for r in upload_results if r["status"] == "success")
    year_fallbacks = sum(1 for r in upload_results if r.get("year_fallback"))
    if year_fallbacks:
        where = "their year folder's year" if from_db else f"year {year}"
        print(f"{year_fallbacks} documents had no parseable start date and were stored with {where}")
    return {
        "year": year,
        "num_files_processed": len(files),
        "num_uploaded": successful,
        "num_year_fallbacks": year_fallbacks,
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
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--measures-only",
        action="store_true",
        help="Only overwrite the text measures of existing documents rows; never change company, "
        "year, title, date or city, and skip files that have no row",
    )
    args = parser.parse_args(argv)
    if args.input_dir and args.year is None:
        parser.error("--input-dir needs --year")
    if args.years and not args.from_db:
        parser.error("--years only applies with --from-db")

    summary = ingest(
        args.input_dir,
        args.year,
        sentiment=default_sentiment_setup() if args.sentiment else None,
        pattern=args.pattern,
        recursive=args.recursive,
        skip_on_error=not args.stop_on_error,
        num_threads=args.threads,
        measures_only=args.measures_only,
        from_db=args.from_db,
        years=args.years,
    )
    print(
        f"{summary['num_uploaded']} of {summary['num_files_processed']} documents uploaded, "
        f"{summary['num_errors']} errors"
    )
    for err in summary["errors"][:20]:
        print(f"  {err['file']}: {err['reason']}")


if __name__ == "__main__":
    main()
