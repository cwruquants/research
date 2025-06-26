from typing import List, Dict, Any, Optional
from transformers import pipeline, Pipeline
from attribute import Attr
import numpy as np
import pandas as pd
import re

from transformers import pipeline
import pysentiment2 as ps

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
        """
        Inspect attr_obj for .word or .text, then compute and set:
        - attr_obj.sentiment   (float)
        - attr_obj.LM          (int)
        - attr_obj.HIV4        (int)
        - attr_obj.ML          (int)
        Returns the mutated attr_obj.
        """
        text = getattr(attr_obj, "text", None) # grab text
        if text is None:
            raise ValueError("Attr object does not have a text")

        hf_res = self.transformer(text)[0] # get sentiment

        if hf_res["label"] == "neutral": # 0 if neutral
            score = 0.0
        else: 
            score = hf_res["score"] if hf_res["label"].lower() == "positive" else -hf_res["score"]


        attr_obj.sentiment = np.float32(score)

        # lm sentiment
        lm_tokens = self.lm.tokenize(text)
        lm_scores = self.lm.get_score(lm_tokens)
        attr_obj.LM = lm_scores["Positive"] - lm_scores["Negative"]

        # hiv4 sentiment
        hiv4_tokens = self.hiv4.tokenize(text)
        hiv4_scores = self.hiv4.get_score(hiv4_tokens)
        attr_obj.HIV4 = hiv4_scores["Positive"] - hiv4_scores["Negative"]

        # ML sentiment
        ml_tokens = re.findall(r'\b\w+\b', text.lower())
        attr_obj.ML = 0
    
        for i in range(len(ml_tokens) - 1):
            if ml_tokens[i] in self.ml_words_negative and ml_tokens[i + 1] in self.ml_words_negative:
                attr_obj.ML -= 1
            if ml_tokens[i] in self.ml_words_positive and ml_tokens[i + 1] in self.ml_words_positive:
                attr_obj.ML += 1
        # Iterate through the words and count positive unigrams
        

        return attr_obj
    

    def ExceltoList(self, file_path, sheet_name) -> list:

        # Read the Excel file
        df = pd.read_excel(file_path,sheet_name=sheet_name ,header=None)
        values = df[0].tolist()

        # Normalize the strings in the list, skipping empty entries
        normalized_list = []
        for val in values:
            if pd.isna(val) or str(val).strip() == '':
                continue  # Skip empty or whitespace-only entries
            normalized_list.append(str(val).replace('_', ' ').lower())
        # Normalize by replacing underscores with spaces and converting to lowercase
        return normalized_list


