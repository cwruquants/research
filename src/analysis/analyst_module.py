from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Union, Sequence
import xml.etree.ElementTree as ET
import csv
import re
import os
import toml
from datetime import datetime
from dateutil import parser as date_parser
import pytz
import textstat

# Try to use notebook tqdm if in Jupyter, fallback to regular tqdm
try:
    from IPython import get_ipython
    if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
        try:
            from tqdm.notebook import tqdm
        except (ImportError, Exception):
            from tqdm import tqdm
    else:
        from tqdm import tqdm
except (ImportError, Exception):
    from tqdm import tqdm

from src.document.decompose_transcript import extract_presentation_section, extract_qa_section, clean_spoken_content
from src.document.abstract_classes.attribute import DocumentAttr
from src.document.abstract_classes.setup_module import SentimentSetup

class Analyst:
    def __init__(self, setups: Union["SentimentSetup", Sequence["SentimentSetup"], None] = None, keyword_path: Optional[str] = None):
        if setups is None:
            self.setups = []
        elif isinstance(setups, SentimentSetup):
            self.setups = [setups]
        else:
            self.setups = list(setups)
        
        self.setup = self.setups[0] if self.setups else None
        self.keyword_path = keyword_path
        # Lazy import to avoid loading heavy dependencies unless needed
        self.matching_agent = None
        if keyword_path:
            from src.analysis.match_extraction.matching_agent import MatchingAgent
            self.matching_agent = MatchingAgent(keywords_path=keyword_path)

    @staticmethod
    def _build_setup_from_dict(setup_dict: Dict[str, Any]) -> "SentimentSetup":
        defaults = {
            "sheet_name_positive": "ML_positive_unigram",
            "sheet_name_negative": "ML_negative_unigram",
            "file_path": "data/word_sets/Garcia_MLWords.xlsx",
            "hf_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "device": -1,
            "batch_size": 32,
            "max_length": 512,
        }
        cfg = {**defaults, **(setup_dict or {})}
        return SentimentSetup(
            sheet_name_positive=cfg["sheet_name_positive"],
            sheet_name_negative=cfg["sheet_name_negative"],
            file_path=cfg["file_path"],
            hf_model=cfg["hf_model"],
            device=cfg["device"],
            batch_size=cfg["batch_size"],
            max_length=cfg["max_length"],
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
        doc: "DocumentAttr",
        setup_dict: Optional[Dict[str, Any]] = None,
    ) -> DocumentAttr:
        """
        Fit/score the document for sentiment.

        Returns:
            DocAttr properly fitted
        """
        setup_obj = self.setup
        if setup_dict is not None:
            setup_obj = self._build_setup_from_dict(setup_dict)

        if not setup_obj:
            raise ValueError("Sentiment analysis requires a SentimentSetup to be loaded or a setup_dict to be provided.")

        # fit
        doc_fit = setup_obj.fit_all(doc)

        return doc_fit
    
    def _fit_matching(
        self,
        earnings_call_path: Union[str, Path],
        matching_method: str
    ):
        earnings_call_path = Path(earnings_call_path)
        if not earnings_call_path.exists():
            raise FileNotFoundError(f"Earnings call not found: {earnings_call_path}")
        
        if matching_method in ("cosine", "direct"):
            return self.matching_agent.single_processing(
                document_path=earnings_call_path,
                matching_function=matching_method
            )
        else: 
            raise ValueError(f"Matching method invalid: {matching_method}")
        
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
        run_sentiment: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract document metadata and optionally perform sentiment analysis.

        Args:
            earnings_call_path: Path to the earnings call XML file
            setup_dict: Optional setup configuration dictionary
            run_sentiment: If True, run sentiment analysis. Defaults to False.

        Returns:
            dict with 'document', 'document_attr', and 'doc_fit' keys
        """
        earnings_call_path = Path(earnings_call_path)

        # Get company metadata
        company_data = self._get_company_attr(earnings_call_path)

        # Extract and clean text from XML
        tree = ET.parse(earnings_call_path)
        root = tree.getroot()
        body_elem = root.find(".//Body")
        body_text = body_elem.text if body_elem is not None else ""
        text_clean = clean_spoken_content(body_text)
        doc = self._make_doc_attr(text_clean)

        # Initialize with readability scores
        document_attr_data = {
            "num_sentences": doc.num_sentences,
            "num_words": doc.num_words,
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
        doc_fit = None

        if run_sentiment:
            # Perform sentiment analysis
            doc_fit = self._fit_sentiment(
                doc=doc,
                setup_dict=setup_dict
            )
            sentiment_data = {
                "sentiment": float(doc_fit.sentiment) if doc_fit.sentiment is not None else 0.0,
                "ML": float(doc_fit.ML) if doc_fit.ML is not None else 0.0,
                "LM": float(doc_fit.LM) if doc_fit.LM is not None else 0.0,
                "HIV4": float(doc_fit.HIV4) if doc_fit.HIV4 is not None else 0.0,
            }
            document_attr_data.update(sentiment_data)

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

        return {
            "document": document_metadata,
            "document_attr": document_attr_data,
            "doc_fit": doc_fit,
        }
    
    def process_single_document(
        self,
        earnings_call_path,
        setup_dict = None,
        run_sentiment: bool = False,
        matching_method: Optional[str] = None,
        output_dir = "results",
    ):
        """
        Analyze a single earnings call document with optional keyword matching.

        Args:
            earnings_call_path: Path to the earnings call XML file
            setup_dict: Optional setup configuration dictionary
            run_sentiment: If True, run sentiment analysis. Defaults to False.
            matching_method: None to skip matching (default), or "direct"/"cosine" to run matching
            output_dir: Base directory for output (default: "results")

        Returns:
            dict with analysis results and file paths
        """
        # Get document attributes
        doc_attrs = self.get_document_attributes(
            earnings_call_path=earnings_call_path,
            setup_dict=setup_dict,
            run_sentiment=run_sentiment
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

        # Optionally run matching analysis based on matching_method
        exposure_results = None
        should_run_matching = matching_method in ("direct", "cosine")
        if should_run_matching:
            if not self.matching_agent:
                raise ValueError("Cannot run matching: no keyword_path provided to Analyst")

            exposure_results = self._fit_matching(
                earnings_call_path=earnings_call_path,
                matching_method=matching_method
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

        if should_run_matching:
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

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path] = "results",
        setup_dict: Optional[Dict[str, Any]] = None,
        run_sentiment: bool = False,
        matching_method: Optional[str] = None,
        pattern: str = "*.xml",
        recursive: bool = False,
        csv_name: str = "batch_summary.csv",
        skip_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all earnings-call XMLs in a folder.

        Args:
            input_dir: folder containing earnings-call XML files
            output_dir: base folder for batch outputs
            setup_dict: optional setup config dict (passed through)
            run_sentiment: If True, run sentiment analysis. Defaults to False.
            matching_method: None to skip matching (default), or "direct"/"cosine" to run matching
            pattern: glob pattern for files (default: *.xml)
            recursive: whether to recurse into subdirs
            csv_name: name of the overall CSV in the batch folder
            skip_on_error: continue on per-file errors if True

        Returns:
            dict with paths and per-file results
        """
        input_dir = Path(input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found or not a directory: {input_dir}")

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
                res = self.process_single_document(
                    earnings_call_path=fpath,
                    setup_dict=setup_dict,
                    run_sentiment=run_sentiment,
                    matching_method=matching_method,
                    output_dir=str(batch_dir),
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
            "results": results,  # list of per-file return dicts from process_single_document
        }

    def process_existing_directory(
        self,
        batch_dir: Union[str, Path],
        keyword_path: str,
        matching_method: str,
        skip_on_error: bool = True,
        transcript_roots: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
        search_recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Run keyword matching on an existing directory created by process_directory.

        Args:
            batch_dir: Path to existing batch directory
            keyword_path: Path to keywords CSV file
            matching_method: Matching method ("cosine" or "direct"). Required parameter.
            skip_on_error: Continue on per-file errors if True

        Returns:
            dict with paths to exposure results and summary CSV

        Notes:
            If the original earnings-call XML paths stored in the batch metadata are not
            available (for example, the batch was created on a different machine), you
            can supply one or more `transcript_roots` directories. The function will look
            for XMLs with matching filenames inside those directories (recursively by
            default) before giving up on the file.
        """
        batch_dir = Path(batch_dir)
        if not batch_dir.exists() or not batch_dir.is_dir():
            raise NotADirectoryError(f"Batch directory not found: {batch_dir}")

        # Ensure a valid matching method is provided for this operation
        if matching_method not in ("direct", "cosine"):
            raise ValueError("matching_method must be 'direct' or 'cosine' for process_existing_directory")

        # Normalise transcript_roots input
        resolved_roots: list[Path] = []
        # Try to infer project root by traversing up until we find a data directory
        current = batch_dir
        project_root = None
        for _ in range(6):
            if (current / "data").exists():
                project_root = current
                break
            current = current.parent

        if transcript_roots is not None:
            if isinstance(transcript_roots, (str, Path)):
                transcript_roots = [transcript_roots]
            for root in transcript_roots:
                try:
                    resolved_roots.append(Path(root))
                except TypeError:
                    continue
        # Always append inferred default root (broader search) if available,
        # even when the caller provided a narrower subfolder like ".../2016".
        if project_root is not None:
            inferred_root = project_root / "data" / "earnings_calls"
            resolved_roots.append(inferred_root)

        # Deduplicate roots while preserving order
        seen_roots = set()
        filtered_roots: list[Path] = []
        for root in resolved_roots:
            if root in seen_roots:
                continue
            seen_roots.add(root)
            filtered_roots.append(root)
        resolved_roots = filtered_roots

        # Build a lazy index of transcripts by filename, only when needed
        transcript_lookup: Dict[str, Path] = {}
        indexed_roots: set[Path] = set()

        def find_transcript_by_name(file_name: str) -> Optional[Path]:
            nonlocal transcript_lookup
            if not file_name:
                return None

            for root in resolved_roots:
                if not root.exists():
                    continue

                if root not in indexed_roots:
                    if search_recursive:
                        try:
                            for candidate in root.rglob("*.xml"):
                                transcript_lookup.setdefault(candidate.name, candidate)
                        except (OSError, ValueError):
                            # If we cannot access the directory, skip it gracefully
                            pass
                    else:
                        for candidate in root.glob("*.xml"):
                            transcript_lookup.setdefault(candidate.name, candidate)
                    indexed_roots.add(root)

                candidate_path = transcript_lookup.get(file_name)
                if candidate_path and candidate_path.exists():
                    return candidate_path

            return None

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

                document_metadata = toml_data.get("document", {})
                earnings_call_path = document_metadata.get("file_path")
                file_name = document_metadata.get("file_name")

                resolved_path: Optional[Path] = None
                if earnings_call_path:
                    path_obj = Path(earnings_call_path)
                    if path_obj.exists():
                        resolved_path = path_obj

                if resolved_path is None and file_name:
                    resolved_path = find_transcript_by_name(file_name)

                if resolved_path is None:
                    message = "Earnings call not found"
                    details = earnings_call_path or file_name or "<unknown>"
                    error_msg = f"{message}: {details}"
                    if skip_on_error:
                        exposure_rows.append({
                            "file_name": file_name or subdir.name,
                            "error": error_msg,
                        })
                        continue
                    else:
                        raise FileNotFoundError(error_msg)

                # Use resolved path and update TOML for future runs
                earnings_call_path = str(resolved_path)
                document_metadata["file_path"] = earnings_call_path
                toml_data["document"] = document_metadata

                # Run matching
                exposure_results = matching_agent.single_processing(
                    document_path=earnings_call_path,
                    matching_function=matching_method
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
                    "matching_method": matching_method,
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
                    "matching_method": matching_method,
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






