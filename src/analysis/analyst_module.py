from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Union, Sequence
import xml.etree.ElementTree as ET
import json
import re
import os
import toml
from datetime import datetime
from dateutil import parser as date_parser
import pytz
import textstat

from src.document.decompose_transcript import (
    extract_presentation_section,
    extract_qa_section,
    clean_spoken_content,
    get_company_metadata,
    extract_full_body_text
)
from src.document.abstract_classes.attribute import DocumentAttr
from src.document.abstract_classes.setup_module import SentimentSetup

class Analyst:
    """
    Core analysis engine for earnings call transcripts.
    
    Focuses on processing a single document at a time. For batch processing,
    use the `BatchRunner` class.

    Available Methods:
    - analyze_document_memory  : CPU-bound analysis (parsing, sentiment, matching).
    - save_results             : I/O-bound saving of analysis results.
    - process_single_document  : Convenience wrapper for analyze + save.
    - get_document_attributes  : Extract metadata and attributes from a single file.
    """
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
            "hf_model": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            "device": -1,
            "batch_size": "auto",
            "max_length": 512,
        }
        cfg = {**defaults, **(setup_dict or {})}
        return SentimentSetup(
            sheet_name_positive=cfg["sheet_name_positive"],
            sheet_name_negative=cfg["sheet_name_negative"],
            ml_wordlist_path=cfg["file_path"],
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
            store_words=False,
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
        matching_method: Optional[str]
    ):
        earnings_call_path = Path(earnings_call_path)
        if not earnings_call_path.exists():
            raise FileNotFoundError(f"Earnings call not found: {earnings_call_path}")
        
        if matching_method is None:
            return None

        if matching_method in ("cosine", "direct"):
            return self.matching_agent.single_processing(
                document_path=earnings_call_path,
                matching_function=matching_method
            )
        else: 
            raise ValueError(f"Matching method invalid: {matching_method}")
        
    def get_document_attributes(
        self,
        earnings_call_path: Union[str, Path],
        setup_dict: Optional[Dict[str, Any]] = None,
        run_sentiment: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract document metadata and optionally perform sentiment analysis.

        Args:
            earnings_call_path (Union[str, Path]): Path to the earnings call XML file.
            setup_dict (Optional[Dict[str, Any]]): Optional setup configuration dictionary for sentiment analysis.
            run_sentiment (bool): If True, run sentiment analysis. Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - document: Metadata about the document (company name, date, etc.).
                - document_attr: Extracted attributes like readability scores and sentiment (if run).
                - doc_fit: The fitted DocumentAttr object (if sentiment was run), else None.
        """
        earnings_call_path = Path(earnings_call_path)

        # Get company metadata
        company_data = get_company_metadata(earnings_call_path)

        # Extract and clean text from XML
        body_text = extract_full_body_text(earnings_call_path)
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
    
    def analyze_document_memory(
        self,
        earnings_call_path: Union[str, Path],
        setup_dict: Optional[Dict[str, Any]] = None,
        run_sentiment: bool = False,
        matching_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform analysis (CPU-bound) without writing to disk.
        Returns a dictionary with all data needed for saving.
        """
        earnings_call_path = Path(earnings_call_path)
        
        # Get document attributes
        doc_attrs = self.get_document_attributes(
            earnings_call_path=earnings_call_path,
            setup_dict=setup_dict,
            run_sentiment=run_sentiment
        )

        # Optionally run matching analysis
        exposure_results = None
        should_run_matching = matching_method in ("direct", "cosine")
        if should_run_matching:
            if not self.matching_agent:
                raise ValueError("Cannot run matching: no keyword_path provided to Analyst")

            exposure_results = self._fit_matching(
                earnings_call_path=earnings_call_path,
                matching_method=matching_method
            )
            
        return {
            "earnings_call_path": earnings_call_path,
            "doc_attrs": doc_attrs,
            "exposure_results": exposure_results,
            "should_run_matching": should_run_matching,
        }

    def save_results(
        self,
        data: Dict[str, Any],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Save analysis results to disk (I/O-bound).
        """
        output_dir = Path(output_dir)
        doc_attrs = data["doc_attrs"]
        exposure_results = data["exposure_results"]
        should_run_matching = data["should_run_matching"]
        earnings_call_path = data["earnings_call_path"]

        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        company_name = doc_attrs["document"].get("company_name", "unknown").replace(" ", "_").replace("/", "_")
        dir_name = f"{company_name}_{timestamp}"
        output_path = output_dir / dir_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Create TOML file with document metadata
        toml_data = {
            "document": doc_attrs["document"],
            "document_attr": doc_attrs["document_attr"]
        }

        # Add matching metadata and save matching results
        if should_run_matching and exposure_results:
            toml_data["metadata_matching"] = {
                "exposure_results_path": str(output_path / "exposure_results.json"),
                "keyword_path": str(self.keyword_path) if self.keyword_path else None
            }

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

        if should_run_matching and exposure_results:
            result["exposure_results_path"] = str(output_path / "exposure_results.json")
            result["exposure_results"] = exposure_results

        return result

    def save_match_results(self, data_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper to save results from a re-match operation (I/O bound).
        """
        subdir = data_package["subdir"]
        exposure_results = data_package["exposure_results"]
        toml_data = data_package["toml_data"]
        match_id = data_package["match_id"]
        keyword_path = data_package["keyword_path"]
        matching_method = data_package["matching_method"]
        timestamp = data_package["timestamp"]
        keyword_name = data_package["keyword_name"]

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
        toml_path = subdir / "analysis_metadata.toml"
        with open(toml_path, 'w', encoding='utf-8') as f:
            toml.dump(toml_data, f)

        return {
            "subdir": str(subdir),
            "exposure_results_path": str(exposure_path),
            "exposure_results": exposure_results,
            "toml_data": toml_data, # Return updated toml for CSV generation
            "match_id": match_id,
            "keyword_name": keyword_name,
            "matching_method": matching_method,
        }

    def process_single_document(
        self,
        earnings_call_path: Union[str, Path],
        setup_dict: Optional[Dict[str, Any]] = None,
        run_sentiment: bool = False,
        matching_method: Optional[str] = None,
        output_dir: Union[str, Path] = "results",
    ) -> Dict[str, Any]:
        """
        Analyze a single earnings call document with optional keyword matching.

        Args:
            earnings_call_path (Union[str, Path]): Path to the earnings call XML file.
            setup_dict (Optional[Dict[str, Any]]): Optional setup configuration dictionary for sentiment analysis.
            run_sentiment (bool): If True, run sentiment analysis using the configured model. Defaults to False.
            matching_method (Optional[str]): The method to use for keyword matching ("direct" or "cosine").
                If None, matching is skipped. Defaults to None.
            output_dir (Union[str, Path]): Base directory where results will be saved. Defaults to "results".

        Returns:
            Dict[str, Any]: A dictionary containing analysis results and file paths:
                - output_directory: Path to the specific result folder for this call.
                - toml_path: Path to the generated analysis_metadata.toml file.
                - doc_fit: The fitted DocumentAttr object (if sentiment was run).
                - exposure_results_path: Path to the exposure JSON file (if matching was run).
                - exposure_results: The ExposureResults object (if matching was run).
        """
        data = self.analyze_document_memory(
            earnings_call_path=earnings_call_path,
            setup_dict=setup_dict,
            run_sentiment=run_sentiment,
            matching_method=matching_method
        )
        return self.save_results(data, output_dir)






