from typing import List, Dict, Any, Optional 
from transformers import pipeline
import numpy as np
import pandas as pd
import re
import json
import pysentiment2 as ps
from src.document.abstract_classes.attribute import Attr, ParagraphAttr, SentenceAttr, DocumentAttr


class Setup:
    def __init__(
        self,
        sheet_name_positive: str,
        sheet_name_negative: str,
        file_path: str,
        hf_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: int = -1,
    ):
        self.transformer = pipeline(
            "sentiment-analysis",
            model=hf_model,
            device=device
        )
        self.lm = ps.LM()
        self.hiv4 = ps.HIV4()
        
        self.ml_words_positive = self.ExceltoList(file_path, sheet_name_positive)
        self.ml_words_negative = self.ExceltoList(file_path, sheet_name_negative)

    def fit(self, attr_obj: Attr):
        text = getattr(attr_obj, "text", None)
        if text is None:
            raise ValueError("Attr object does not have a text")

        # Only apply HuggingFace model to short text (SentenceAttr)
        if isinstance(attr_obj, SentenceAttr):
            hf_res = self.transformer(text[:512])[0]
            if hf_res["label"] == "neutral":
                score = 0.0
            else:
                score = hf_res["score"] if hf_res["label"].lower() == "positive" else -hf_res["score"]
            attr_obj.sentiment = float(score)

        # LM sentiment
        lm_tokens = self.lm.tokenize(text)
        lm_scores = self.lm.get_score(lm_tokens)
        attr_obj.LM = lm_scores["Positive"] - lm_scores["Negative"]

        # HIV4 sentiment
        hiv4_tokens = self.hiv4.tokenize(text)
        hiv4_scores = self.hiv4.get_score(hiv4_tokens)
        attr_obj.HIV4 = hiv4_scores["Positive"] - hiv4_scores["Negative"]

        # ML sentiment
        ml_tokens = re.findall(r'\b\w+\b', text.lower())
        attr_obj.ML = 0
        for i in range(len(ml_tokens) - 1):
            if ml_tokens[i] in self.ml_words_negative:
                attr_obj.ML -= 1
            if ml_tokens[i] in self.ml_words_positive:
                attr_obj.ML += 1

        return attr_obj

    def fit_all(self, attr_obj: Attr):
        if hasattr(attr_obj, "paragraphs") and attr_obj.paragraphs:
            for paragraph in attr_obj.paragraphs:
                self.fit_all(paragraph)

            # Average sentiment
            sentiments = [p.sentiment for p in attr_obj.paragraphs if p.sentiment is not None]
            attr_obj.sentiment = float(np.mean(sentiments)) if sentiments else 0.0

            # Sum custom scores
            for score in ["HIV4", "ML", "LM"]:
                total = sum(getattr(p, score, 0.0) for p in attr_obj.paragraphs)
                setattr(attr_obj, score, float(total))

        elif hasattr(attr_obj, "sentences") and attr_obj.sentences:
            for sentence in attr_obj.sentences:
                self.fit_all(sentence)

            sentiments = [s.sentiment for s in attr_obj.sentences if s.sentiment is not None]
            attr_obj.sentiment = float(np.mean(sentiments)) if sentiments else 0.0

            for score in ["HIV4", "ML", "LM"]:
                total = sum(getattr(s, score, 0.0) for s in attr_obj.sentences)
                setattr(attr_obj, score, float(total))

        else:
            # Leaf node (sentence)
            self.fit(attr_obj)

        return attr_obj






    def export_to_json(self, attr_obj, filepath):
        def serialize(attr_obj: Attr) -> dict:
            result = {
                "text": attr_obj.text,
                "sentiment": float(attr_obj.sentiment) if attr_obj.sentiment is not None else 0.0,
                "HIV4": float(getattr(attr_obj, "HIV4", 0.0)),
                "ML": float(getattr(attr_obj, "ML", 0.0)),
                "LM": float(getattr(attr_obj, "LM", 0.0)),
            }

            if hasattr(attr_obj, "paragraphs"):
                result["paragraphs"] = [serialize(p) for p in attr_obj.paragraphs]
            elif hasattr(attr_obj, "sentences"):
                result["sentences"] = [serialize(s) for s in attr_obj.sentences]

            return result
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serialize(attr_obj), f, indent=2, ensure_ascii=False)







    def ExceltoList(self, file_path, sheet_name) -> list:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        values = df[0].tolist()
        normalized_list = []
        for val in values:
            if pd.isna(val) or str(val).strip() == '':
                continue
            normalized_list.append(str(val).replace('_', ' ').lower())
        return normalized_list

    
    #TODO: Diliya
    def findWeight(corpus: DocumentAttr = None):
        pass


