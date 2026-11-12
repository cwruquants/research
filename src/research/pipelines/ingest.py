"""
Parse transcript XMLs and write their companies and documents rows to Postgres.

For each XML: company metadata, then sentence and word counts, readability scores and,
with --sentiment, the transformer, LM, HIV4 and Garcia ML sentiment scores. Rows are
upserted on documents.source_file, so re-running a folder updates it in place. Keyword
exposures are a separate step: `research backfill`.

Every text measure uses the spoken text (Presentation and Q&A with speaker lines removed),
the same text keyword matching reads. Rows stored before 2026-11-11 used the whole XML
body and have not been recomputed (see docs/known_issues.md).

Usage (DATABASE_URL set to a role that can write companies and documents):

    uv run research ingest --input-dir <EARNINGS_CALLS>/2016 --year 2016 --threads 4
    uv run research ingest --input-dir <EARNINGS_CALLS>/2016 --year 2016 --sentiment   # needs --extra ml
"""

import argparse
import concurrent.futures
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from tqdm.auto import tqdm

from research import config
from research.db.documents import PostgresUploader
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


def analyze_document(path: Union[str, Path], sentiment=None) -> Dict[str, Any]:
    """
    Metadata and text measures for one transcript, shaped for PostgresUploader:
    {"document": metadata, "document_attr": measures, "doc_fit": fitted DocumentAttr or None}.
    Pass a SentimentSetup as `sentiment` to add the sentiment scores.
    """
    path = Path(path)
    company = get_company_metadata(path)

    text = spoken_text(path)
    doc = DocumentAttr(text, store_paragraphs=True, store_sentences=True, store_words=False)
    measures = {"num_sentences": doc.num_sentences, "num_words": doc.num_words, **readability_scores(text)}

    doc_fit = None
    if sentiment is not None:
        doc_fit = sentiment.fit_all(doc)
        for key, value in (
            ("sentiment", doc_fit.sentiment),
            ("ML", doc_fit.ML),
            ("LM", doc_fit.LM),
            ("HIV4", doc_fit.HIV4),
        ):
            measures[key] = float(value) if value is not None else 0.0

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


def ingest(
    input_dir: Union[str, Path],
    year: int,
    database_url: Optional[str] = None,
    sentiment=None,
    pattern: str = "*.xml",
    recursive: bool = False,
    skip_on_error: bool = True,
    specific_files: Optional[Sequence[Union[str, Path]]] = None,
    num_threads: int = 1,
) -> Dict[str, Any]:
    """
    Analyze every matching XML in input_dir and upsert its companies and documents rows.
    `year` is stored only for transcripts with no parseable start date; otherwise the
    year comes from the call's start date.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    files = _gather(input_dir, pattern, recursive, specific_files)
    threads = max(1, num_threads or 1)
    # Fetch and load the NLTK data in this thread: the workers would race to download it,
    # and NLTK's lazy cmudict loader fails when several threads touch it first
    ensure_nltk_data()
    readability_scores("Load the syllable dictionary.")

    uploader = PostgresUploader(database_url or config.database_url(), max_connections=threads)
    upload_results, errors = [], []
    try:
        print(f"Loaded {uploader.initialize_company_cache()} existing companies")

        def process(path: Path) -> Dict[str, Any]:
            result = uploader.upload_document({"doc_attrs": analyze_document(path, sentiment)}, year)
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
        print(f"{year_fallbacks} documents had no parseable start date and were stored with year {year}")
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
    parser.add_argument("--input-dir", required=True, help="Folder of transcript XMLs")
    parser.add_argument(
        "--year", type=int, required=True, help="Year stored for transcripts with no parseable start date"
    )
    parser.add_argument("--pattern", default="*.xml")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--sentiment", action="store_true", help="Also score sentiment (needs --extra ml)")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)

    summary = ingest(
        args.input_dir,
        args.year,
        sentiment=default_sentiment_setup() if args.sentiment else None,
        pattern=args.pattern,
        recursive=args.recursive,
        skip_on_error=not args.stop_on_error,
        num_threads=args.threads,
    )
    print(
        f"{summary['num_uploaded']} of {summary['num_files_processed']} documents uploaded, "
        f"{summary['num_errors']} errors"
    )
    for err in summary["errors"][:20]:
        print(f"  {err['file']}: {err['reason']}")


if __name__ == "__main__":
    main()
