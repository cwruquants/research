from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Union
import xml.etree.ElementTree as ET
import csv
import re
import os
import toml
from datetime import datetime
from dateutil import parser as date_parser
import pytz
import textstat
from tqdm.notebook import tqdm

from src.document.decompose_transcript import extract_presentation_section, extract_qa_section, clean_spoken_content
from src.document.abstract_classes.attribute import DocumentAttr
from src.document.abstract_classes.setup_module import Setup

class Analyst:
    def __init__(self, setup: Optional["Setup"] = None, keyword_path = None):
        # I want an analyst to specialize in a single Setup, and to specialize in a single keyword doc
        # If we want to do a different SetUp or keyword doc, we should create another analyst
        self.setup = setup
        self.keyword_path = keyword_path
        # Lazy import to avoid loading heavy dependencies unless needed
        self.matching_agent = None
        if keyword_path:
            from src.analysis.match_extraction.matching_agent import MatchingAgent
            self.matching_agent = MatchingAgent(keywords_path=keyword_path)

    @staticmethod
    def _build_setup_from_dict(setup_dict: Dict[str, Any]) -> "Setup":
        defaults = {
            "sheet_name_positive": "ML_positive_unigram",
            "sheet_name_negative": "ML_negative_unigram",
            "file_path": "data/word_sets/Garcia_MLWords.xlsx",
            "hf_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "device": -1,
        }
        cfg = {**defaults, **(setup_dict or {})}
        return Setup(
            sheet_name_positive=cfg["sheet_name_positive"],
            sheet_name_negative=cfg["sheet_name_negative"],
            file_path=cfg["file_path"],
            hf_model=cfg["hf_model"],
            device=cfg["device"],
        )

    @staticmethod
    def _make_doc_attr(text: str) -> "DocumentAttr":
        return DocumentAttr(
            text,
            store_paragraphs=True,
            store_sentences=True,
            store_words=True,
        )

    def _fit_sentiment(
        self,
        earnings_call_path: Union[str, Path],
        setup_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Pipeline:
          1) Extract entire Body section from XML.
          2) Clean the text.
          3) Convert into DocumentAttr.
          4) Ensure we have a Setup (use self.setup or build from setup_dict).
          5) Fit/score the document.

        Returns:
            DocAttr properly fitted
        """
        earnings_call_path = Path(earnings_call_path)
        if not earnings_call_path.exists():
            raise FileNotFoundError(f"Earnings call not found: {earnings_call_path}")

        # Extract entire body text from XML
        tree = ET.parse(earnings_call_path)
        root = tree.getroot()
        body_elem = root.find(".//Body")
        body_text = body_elem.text if body_elem is not None else ""

        # Clean the text
        text_clean = clean_spoken_content(body_text)

        # Create document attribute
        doc = self._make_doc_attr(text_clean)

        num_sentences = doc.num_sentences
        num_words = doc.num_words

        # make sure setup exists
        setup_obj = self.setup or self._build_setup_from_dict(setup_dict or {})
        # fit
        doc_fit = setup_obj.fit_all(doc)

        return doc_fit, num_sentences, num_words, text_clean
    
    def _fit_matching(
        self,
        earnings_call_path: Union[str, Path],
        similarity: str = "cosine"
    ):
        earnings_call_path = Path(earnings_call_path)
        if not earnings_call_path.exists():
            raise FileNotFoundError(f"Earnings call not found: {earnings_call_path}")
        
        if similarity == "cosine" or similarity == "direct":
            return self.matching_agent.single_processing(
                document_path=earnings_call_path,
                matching_function=similarity
            )
        else: 
            raise ValueError(f"Matching method invalid: {similarity}")
        
    def _get_company_attr(self, earnings_call_path):
        tree = ET.parse(earnings_call_path)
        root = tree.getroot()

        event_title = root.findtext("eventTitle", "")

        match = re.search(r"(Q[1-4])\s+(\d{4})", event_title)
        quarter, year = (match.group(1), match.group(2)) if match else (None, None)

        # Parse and standardize startDate to GMT
        start_date_raw = root.findtext("startDate")
        start_date_gmt = None
        if start_date_raw:
            try:
                # Parse the date string with timezone info
                dt = date_parser.parse(start_date_raw)
                # Convert to GMT/UTC
                if dt.tzinfo is None:
                    # If no timezone, assume GMT
                    dt = pytz.UTC.localize(dt)
                else:
                    dt = dt.astimezone(pytz.UTC)
                # Format as standard GMT string
                start_date_gmt = dt.strftime("%d-%b-%y %I:%M%p GMT")
            except Exception:
                # If parsing fails, keep original
                start_date_gmt = start_date_raw

        data = {
            "eventTitle": event_title,
            "city": root.findtext("city"),
            "companyName": root.findtext("companyName"),
            "companyTicker": root.findtext("companyTicker"),
            "startDate": start_date_gmt,
            "quarter": quarter,
            "year": year,
        }

        return data

    def get_document_attributes(
        self,
        earnings_call_path: Union[str, Path],
        setup_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract document metadata and perform sentiment analysis.

        Args:
            earnings_call_path: Path to the earnings call XML file
            setup_dict: Optional setup configuration dictionary

        Returns:
            dict with 'document', 'document_attr', and 'doc_fit' keys
        """
        earnings_call_path = Path(earnings_call_path)

        # Get company metadata
        company_data = self._get_company_attr(earnings_call_path)

        # Perform sentiment analysis
        doc_fit, num_sentences, num_words, text_clean = self._fit_sentiment(
            earnings_call_path=earnings_call_path,
            setup_dict=setup_dict
        )

        # Build document metadata
        document_metadata = {
            "file_name": earnings_call_path.name,
            "file_path": str(earnings_call_path),
            "company_name": company_data.get("companyName"),
            "company_ticker": company_data.get("companyTicker"),
            "start_date": company_data.get("startDate"),
            "quarter": company_data.get("quarter"),
            "year": company_data.get("year"),
            "city": company_data.get("city"),
            "event_title": company_data.get("eventTitle"),
        }

        # Build document attribute data (sentiment scores)
        document_attr_data = {
            "sentiment": float(doc_fit.sentiment) if doc_fit.sentiment is not None else 0.0,
            "ML": float(doc_fit.ML) if doc_fit.ML is not None else 0.0,
            "LM": float(doc_fit.LM) if doc_fit.LM is not None else 0.0,
            "HIV4": float(doc_fit.HIV4) if doc_fit.HIV4 is not None else 0.0,
            "num_sentences": num_sentences,
            "num_words": num_words,
            "flesch_reading_ease": textstat.flesch_reading_ease(text_clean),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text_clean),
            "smog_index": textstat.smog_index(text_clean),
            "coleman_liau_index": textstat.coleman_liau_index(text_clean),
            "automated_readability_index": textstat.automated_readability_index(text_clean),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(text_clean),
            "difficult_words": textstat.difficult_words(text_clean),
            "linsear_write_formula": textstat.linsear_write_formula(text_clean),
            "gunning_fog": textstat.gunning_fog(text_clean),
            "text_standard": textstat.text_standard(text_clean),
        }

        return {
            "document": document_metadata,
            "document_attr": document_attr_data,
            "doc_fit": doc_fit,
        }
    
    def fit_single_document(
        self,
        earnings_call_path,
        setup_dict = None,
        similarity = None,
        output_dir = "results",
        run_matching = True,
    ):
        """
        Analyze a single earnings call document with optional keyword matching.

        Args:
            earnings_call_path: Path to the earnings call XML file
            setup_dict: Optional setup configuration dictionary
            similarity: Matching method ("cosine" or "direct"), required if run_matching=True
            output_dir: Base directory for output (default: "results")
            run_matching: Whether to run keyword matching (default: True)

        Returns:
            dict with analysis results and file paths
        """
        # Get document attributes (always runs sentiment analysis)
        doc_attrs = self.get_document_attributes(
            earnings_call_path=earnings_call_path,
            setup_dict=setup_dict
        )

        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_name = doc_attrs["document"].get("company_name", "unknown").replace(" ", "_").replace("/", "_")
        dir_name = f"{company_name}_{timestamp}"
        output_path = Path(output_dir) / dir_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Create TOML file with document metadata
        toml_data = {
            "document": doc_attrs["document"],
            "document_attr": doc_attrs["document_attr"]
        }

        # Optionally run matching analysis
        exposure_results = None
        if run_matching:
            if not self.matching_agent:
                raise ValueError("Cannot run matching: no keyword_path provided to Analyst")
            if not similarity:
                raise ValueError("similarity parameter required when run_matching=True")

            exposure_results = self._fit_matching(
                earnings_call_path=earnings_call_path,
                similarity=similarity
            )

            # Add matching metadata to TOML
            toml_data["metadata_matching"] = {
                "exposure_results_path": str(output_path / "exposure_results.json"),
                "keyword_path": str(self.keyword_path) if self.keyword_path else None
            }

            # Save matching agent results
            exposure_results.export_to_json(
                output_dir=str(output_path),
                export_format="word",
                filename="exposure_results.json"
            )

        # Save TOML file
        toml_path = output_path / "analysis_metadata.toml"
        with open(toml_path, 'w', encoding='utf-8') as f:
            toml.dump(toml_data, f)

        result = {
            "output_directory": str(output_path),
            "toml_path": str(toml_path),
            "doc_fit": doc_attrs["doc_fit"],
        }

        if run_matching:
            result["exposure_results_path"] = str(output_path / "exposure_results.json")
            result["exposure_results"] = exposure_results

        return result
    
    @staticmethod
    def _flatten_analysis_toml(toml_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten the analysis_metadata.toml structure into a single dict suitable for CSV.
        Keeps stable column names across files.
        """
        doc = toml_data.get("document", {})
        meta = toml_data.get("metadata_matching", {})
        da = toml_data.get("document_attr", {})

        start_date_str = doc.get("start_date")
        date_val, time_val = None, None
        if start_date_str:
            try:
                # The format is set in _get_company_attr
                dt = datetime.strptime(start_date_str, "%d-%b-%y %I:%M%p GMT")
                date_val = dt.strftime("%Y-%m-%d")
                time_val = dt.strftime("%H:%M:%S")
            except (ValueError, TypeError):
                # Fallback for unexpected format
                parts = start_date_str.split(" ", 1)
                date_val = parts[0]
                time_val = parts[1] if len(parts) > 1 else ""

        flat = {
            # document (top-level identifiers)
            "file_name": doc.get("file_name"),
            "company_name": doc.get("company_name"),
            "company_ticker": doc.get("company_ticker"),
            "date": date_val,
            "time": time_val,
            "quarter": doc.get("quarter"),
            "city": doc.get("city"),
            "event_title": doc.get("event_title"),

            # document attributes
            "sentiment": da.get("sentiment"),
            "ML": da.get("ML"),
            "LM": da.get("LM"),
            "HIV4": da.get("HIV4"),
            "num_sentences": da.get("num_sentences"),
            "num_words": da.get("num_words"),
            "flesch_reading_ease": da.get("flesch_reading_ease"),
            "flesch_kincaid_grade": da.get("flesch_kincaid_grade"),
            "smog_index": da.get("smog_index"),
            "coleman_liau_index": da.get("coleman_liau_index"),
            "automated_readability_index": da.get("automated_readability_index"),
            "dale_chall_readability_score": da.get("dale_chall_readability_score"),
            "difficult_words": da.get("difficult_words"),
            "linsear_write_formula": da.get("linsear_write_formula"),
            "gunning_fog": da.get("gunning_fog"),
            "text_standard": da.get("text_standard"),
        }
        return flat

    def fit_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path] = "results",
        setup_dict: Optional[Dict[str, Any]] = None,
        similarity: Optional[str] = None,
        pattern: str = "*.xml",
        recursive: bool = False,
        csv_name: str = "batch_summary.csv",
        skip_on_error: bool = True,
        run_matching: bool = False,
    ) -> Dict[str, Any]:
        """
        Process all earnings-call XMLs in a folder (sentiment analysis only by default).

        Args:
            input_dir: folder containing earnings-call XML files
            output_dir: base folder for batch outputs
            setup_dict: optional setup config dict (passed through)
            similarity: "cosine" or "direct" (required if run_matching=True)
            pattern: glob pattern for files (default: *.xml)
            recursive: whether to recurse into subdirs
            csv_name: name of the overall CSV in the batch folder
            skip_on_error: continue on per-file errors if True
            run_matching: whether to run keyword matching (default: False)

        Returns:
            dict with paths and per-file results
        """
        input_dir = Path(input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found or not a directory: {input_dir}")

        if run_matching and not similarity:
            raise ValueError("similarity parameter required when run_matching=True")

        # Create a single batch folder to hold all per-call outputs
        batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = Path(output_dir) / f"batch_{batch_stamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Discover files
        files = (
            list(input_dir.rglob(pattern)) if recursive
            else list(input_dir.glob(pattern))
        )
        files = [p for p in files if p.is_file()]
        if not files:
            raise FileNotFoundError(f"No files matched pattern '{pattern}' in {input_dir} (recursive={recursive}).")

        rows = []
        results = []

        files_sorted = sorted(files)
        for fpath in tqdm(files_sorted, total=len(files_sorted), desc="Processing files", unit="file", dynamic_ncols=True):
            try:
                # Each call will create its own subdirectory inside `batch_dir`
                res = self.fit_single_document(
                    earnings_call_path=fpath,
                    setup_dict=setup_dict,
                    similarity=similarity,
                    output_dir=str(batch_dir),
                    run_matching=run_matching,
                )
                results.append(res)

                # Load the TOML we just wrote so the CSV is guaranteed to match file output
                toml_path = Path(res["toml_path"])
                with open(toml_path, "r", encoding="utf-8") as tf:
                    tdata = toml.load(tf)

                row = self._flatten_analysis_toml(tdata)
                rows.append(row)

            except Exception as e:
                if skip_on_error:
                    # Record an error row with minimal info so you can see what failed
                    rows.append({
                        "file_name": fpath.name,
                        "file_path": str(fpath),
                        "error": str(e),
                    })
                    continue
                else:
                    raise

        # Write overall CSV
        csv_path = batch_dir / csv_name
        # union of keys across all rows to avoid missing columns
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in fieldnames})

        return {
            "batch_directory": str(batch_dir),
            "csv_path": str(csv_path),
            "num_files_processed": len(files),
            "results": results,  # list of per-file return dicts from fit_single_document
        }

    def match_existing_batch(
        self,
        batch_dir: Union[str, Path],
        keyword_path: str,
        similarity: str = "cosine",
        skip_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Run keyword matching on an existing batch directory created by fit_directory.

        Args:
            batch_dir: Path to existing batch directory
            keyword_path: Path to keywords CSV file
            similarity: Matching method ("cosine" or "direct")
            skip_on_error: Continue on per-file errors if True

        Returns:
            dict with paths to exposure results and summary CSV
        """
        batch_dir = Path(batch_dir)
        if not batch_dir.exists() or not batch_dir.is_dir():
            raise NotADirectoryError(f"Batch directory not found: {batch_dir}")

        # Create a matching agent for this keyword set
        from src.analysis.match_extraction.matching_agent import MatchingAgent
        matching_agent = MatchingAgent(keywords_path=keyword_path)

        # Generate unique identifier for this matching run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        keyword_name = Path(keyword_path).stem
        match_id = f"{keyword_name}_{timestamp}"

        # Find all subdirectories (each represents an earnings call)
        subdirs = [d for d in batch_dir.iterdir() if d.is_dir()]

        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in {batch_dir}")

        results = []
        exposure_rows = []

        for subdir in tqdm(sorted(subdirs), total=len(subdirs), desc="Matching keywords", unit="file", dynamic_ncols=True):
            toml_path = subdir / "analysis_metadata.toml"

            if not toml_path.exists():
                if skip_on_error:
                    continue
                else:
                    raise FileNotFoundError(f"analysis_metadata.toml not found in {subdir}")

            try:
                # Load existing TOML to get earnings call path
                with open(toml_path, "r", encoding="utf-8") as f:
                    toml_data = toml.load(f)

                earnings_call_path = toml_data.get("document", {}).get("file_path")
                if not earnings_call_path or not Path(earnings_call_path).exists():
                    if skip_on_error:
                        continue
                    else:
                        raise FileNotFoundError(f"Earnings call not found: {earnings_call_path}")

                # Run matching
                exposure_results = matching_agent.single_processing(
                    document_path=earnings_call_path,
                    matching_function=similarity
                )

                # Save exposure results with unique name
                exposure_filename = f"exposure_results_{match_id}.json"
                exposure_path = subdir / exposure_filename
                exposure_results.export_to_json(
                    output_dir=str(subdir),
                    export_format="word",
                    filename=exposure_filename
                )

                # Update TOML with this matching run metadata
                if "metadata_matching_runs" not in toml_data:
                    toml_data["metadata_matching_runs"] = []

                toml_data["metadata_matching_runs"].append({
                    "match_id": match_id,
                    "keyword_path": keyword_path,
                    "keyword_name": keyword_name,
                    "similarity": similarity,
                    "timestamp": timestamp,
                    "exposure_results_path": str(exposure_path),
                })

                # Save updated TOML
                with open(toml_path, 'w', encoding='utf-8') as f:
                    toml.dump(toml_data, f)

                results.append({
                    "subdir": str(subdir),
                    "exposure_results_path": str(exposure_path),
                    "exposure_results": exposure_results,
                })

                # Build row for exposure summary CSV
                doc_metadata = toml_data.get("document", {})

                start_date_str = doc_metadata.get("start_date")
                date_val, time_val = None, None
                if start_date_str:
                    try:
                        # The format is set in _get_company_attr
                        dt = datetime.strptime(start_date_str, "%d-%b-%y %I:%M%p GMT")
                        date_val = dt.strftime("%Y-%m-%d")
                        time_val = dt.strftime("%H:%M:%S")
                    except (ValueError, TypeError):
                        # Fallback for unexpected format
                        parts = start_date_str.split(" ", 1)
                        date_val = parts[0]
                        time_val = parts[1] if len(parts) > 1 else ""

                exposure_row = {
                    "file_name": doc_metadata.get("file_name"),
                    "company_name": doc_metadata.get("company_name"),
                    "company_ticker": doc_metadata.get("company_ticker"),
                    "date": date_val,
                    "time": time_val,
                    "quarter": doc_metadata.get("quarter"),
                    "match_id": match_id,
                    "keyword_name": keyword_name,
                    "similarity": similarity,
                    "total_keywords_searched": exposure_results.total_keywords_searched,
                    "total_keywords_with_matches": exposure_results.total_keywords_with_matches,
                    "total_direct_matches": exposure_results.total_direct_matches,
                    "total_cosine_matches": exposure_results.total_cosine_matches,
                    "total_matches": exposure_results.total_direct_matches + exposure_results.total_cosine_matches,
                }
                exposure_rows.append(exposure_row)

            except Exception as e:
                if skip_on_error:
                    exposure_rows.append({
                        "file_name": subdir.name,
                        "error": str(e),
                    })
                    continue
                else:
                    raise

        # Write exposure summary CSV
        csv_filename = f"exposure_summary_{match_id}.csv"
        csv_path = batch_dir / csv_filename

        all_keys = set()
        for r in exposure_rows:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in exposure_rows:
                writer.writerow({k: r.get(k) for k in fieldnames})

        return {
            "batch_directory": str(batch_dir),
            "match_id": match_id,
            "exposure_summary_csv": str(csv_path),
            "num_files_processed": len(results),
            "results": results,
        }






