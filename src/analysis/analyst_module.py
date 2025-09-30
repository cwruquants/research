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

from src.document.decompose_transcript import extract_presentation_section, extract_qa_section, clean_spoken_content
from src.document.abstract_classes.attribute import DocumentAttr
from src.document.abstract_classes.setup_module import Setup
from src.analysis.match_extraction.matching_agent import MatchingAgent

class Analyst:
    def __init__(self, setup: Optional["Setup"] = None, keyword_path = None):
        # I want an analyst to specialize in a single Setup, and to specialize in a single keyword doc
        # If we want to do a different SetUp or keyword doc, we should create another analyst
        self.setup = setup
        self.keyword_path = keyword_path
        self.matching_agent = MatchingAgent(
            keywords_path=keyword_path,
        )

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

        return doc_fit, num_sentences, num_words
    
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
    
    def fit_single_document(
        self,
        earnings_call_path,
        setup_dict = None,
        similarity = "",
        output_dir = "results"
    ):
        """
            Calls fit_sentiment and fit_matching, then saves results to a new directory.
            
            Args:
                earnings_call_path: Path to the earnings call XML file
                setup_dict: Optional setup configuration dictionary
                similarity: Matching method ("cosine" or "direct")
                output_dir: Base directory for output (default: "results")
        """
        # Get company attributes from the earnings call
        company_attr = self._get_company_attr(earnings_call_path)

        # Run sentiment analysis
        doc_fit, num_sentences, num_words = self._fit_sentiment(
            earnings_call_path=earnings_call_path,
            setup_dict=setup_dict
        )

        # Run matching analysis
        exposure_results = self._fit_matching(
            earnings_call_path=earnings_call_path,
            similarity=similarity
        )
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_name = company_attr.get("companyName", "unknown").replace(" ", "_").replace("/", "_")
        dir_name = f"{company_name}_{timestamp}"
        output_path = Path(output_dir) / dir_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create TOML file with document metadata and analysis results
        toml_data = {
            "document": {
                "file_name": Path(earnings_call_path).name,
                "file_path": str(earnings_call_path),
                "company_name": company_attr.get("companyName"),
                "company_ticker": company_attr.get("companyTicker"),
                "start_date": company_attr.get("startDate"),
                "quarter": company_attr.get("quarter"),
                "year": company_attr.get("year"),
                "city": company_attr.get("city"),
                "event_title": company_attr.get("eventTitle")
            },
            "metadata_matching": {
                "exposure_results_path": str(output_path / "exposure_results.json"),
                "keyword_path": str(self.keyword_path) if self.keyword_path else None
            },
            "document_attr": {
                "sentiment": float(doc_fit.sentiment) if doc_fit.sentiment is not None else 0.0,
                "ML": float(doc_fit.ML),
                "LM": float(doc_fit.LM),
                "HIV4": float(doc_fit.HIV4),
                "num_sentences": num_sentences,
                "num_words": num_words
            }
        }
        
        # Save TOML file
        toml_path = output_path / "analysis_metadata.toml"
        with open(toml_path, 'w', encoding='utf-8') as f:
            toml.dump(toml_data, f)
        
        # Save matching agent results
        exposure_results.export_to_json(
            output_dir=str(output_path),
            export_format="word",
            filename="exposure_results.json"
        )
        
        # print(f"Analysis complete. Results saved to: {output_path}")
        # print(f"TOML metadata: {toml_path}")
        # print(f"Exposure results: {output_path / 'exposure_results.json'}")
        
        return {
            "output_directory": str(output_path),
            "toml_path": str(toml_path),
            "exposure_results_path": str(output_path / "exposure_results.json"),
            "doc_fit": doc_fit,
            "exposure_results": exposure_results
        }
    
    @staticmethod
    def _flatten_analysis_toml(toml_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten the analysis_metadata.toml structure into a single dict suitable for CSV.
        Keeps stable column names across files.
        """
        doc = toml_data.get("document", {})
        meta = toml_data.get("metadata_matching", {})
        da = toml_data.get("document_attr", {})

        flat = {
            # document (top-level identifiers)
            "file_name": doc.get("file_name"),
            "file_path": doc.get("file_path"),
            "company_name": doc.get("company_name"),
            "company_ticker": doc.get("company_ticker"),
            "start_date": doc.get("start_date"),
            "quarter": doc.get("quarter"),
            "year": doc.get("year"),
            "city": doc.get("city"),
            "event_title": doc.get("event_title"),

            # matching metadata
            "exposure_results_path": meta.get("exposure_results_path"),
            "keyword_path": meta.get("keyword_path"),

            # document attributes
            "sentiment": da.get("sentiment"),
            "ML": da.get("ML"),
            "LM": da.get("LM"),
            "HIV4": da.get("HIV4"),
            "num_sentences": da.get("num_sentences"),
            "num_words": da.get("num_words"),
        }
        return flat

    def fit_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path] = "results",
        setup_dict: Optional[Dict[str, Any]] = None,
        similarity: str = "cosine",
        pattern: str = "*.xml",
        recursive: bool = False,
        csv_name: str = "batch_summary.csv",
        skip_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all earnings-call XMLs in a folder:
          - Creates a batch output folder under `output_dir`
          - For each XML, runs `fit_single_document(...)` which creates its own subfolder
          - Builds a CSV with all information stored in analysis_metadata.toml per file

        Args:
            input_dir: folder containing earnings-call XML files
            output_dir: base folder for batch outputs
            setup_dict: optional setup config dict (passed through)
            similarity: "cosine" or "direct" (passed through)
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

        for fpath in sorted(files):
            try:
                # Each call will create its own subdirectory inside `batch_dir`
                res = self.fit_single_document(
                    earnings_call_path=fpath,
                    setup_dict=setup_dict,
                    similarity=similarity,
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
            "results": results,  # list of per-file return dicts from fit_single_document
        }






