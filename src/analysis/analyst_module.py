from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Union
import xml.etree.ElementTree as ET
import re

from ..document.decompose_transcript import extract_presentation_section, extract_qa_section, clean_spoken_content
from ..document.abstract_classes.attribute import DocumentAttr
from ..document.abstract_classes.setup_module import Setup
from ..analysis.match_extraction.matching_agent import MatchingAgent

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
            "file_path": "../../data/word_sets/Garcia_MLWords.xlsx",
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
          1) Extract Q&A and Presentation sections from an earnings-call XML.
          2) Clean the text.
          3) Convert both sections into DocumentAttr.
          4) Ensure we have a Setup (use self.setup or build from setup_dict).
          5) Fit/score each section.

        Returns:
            DocAttr properly fitted
        """
        earnings_call_path = Path(earnings_call_path)
        if not earnings_call_path.exists():
            raise FileNotFoundError(f"Earnings call not found: {earnings_call_path}")

        qa_clean = clean_spoken_content(extract_qa_section(str(earnings_call_path)))
        pres_clean = clean_spoken_content(extract_presentation_section(str(earnings_call_path)))

        qa_doc, pres_doc = self._make_doc_attr(qa_clean), self._make_doc_attr(pres_clean)

        num_sentences = qa_doc.num_sentences+pres_doc.num_sentences
        num_words = qa_doc.num_words+pres_doc.num_words

        # make sure setup exists
        setup_obj = self.setup or self._build_setup_from_dict(setup_dict or {})
        # fit
        qa_fit, pres_fit = setup_obj.fit_all(qa_doc), setup_obj.fit_all(pres_doc)

        # self.qa = qa_fit.get_sentiment()
        # self.pres = pres_fit.get_sentiment()

        return qa_fit, pres_fit, num_sentences, num_words
    
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
        
        data = {
            "eventTitle": event_title,
            "city": root.findtext("city"),
            "companyName": root.findtext("companyName"),
            "companyTicker": root.findtext("companyTicker"),
            "startDate": root.findtext("startDate"),
            "quarter": quarter,
            "year": year,
        }
        
        return data
    
    def fit_single_document(
        self,
        earnings_call_path,
        setup_dict = None,
        similarity = ""             
    ):
        """
            Calls fit_sentiment and fit_matching.
        """
        qa_fit, pres_fit, num_sentences, num_words = self._fit_sentiment(
            earnings_call_path=earnings_call_path,
            setup_dict=setup_dict
        )
        exposure_results = self._fit_matching(
            earnings_call_path=earnings_call_path,
            similarity=similarity
        )#.export_to_dict()
        # need to format to toml
        print(qa_fit)
        print(pres_fit)
        print(exposure_results.export_to_dict)






