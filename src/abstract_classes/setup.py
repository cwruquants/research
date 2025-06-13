from typing import List, Dict, Any, Optional
from transformers import pipeline, Pipeline
from .attribute import Attr
import numpy as np

from transformers import pipeline
import pysentiment2 as ps

class Setup:
    def __init__(
        self,
        hf_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: int = -1
    ):
        self.transformer = pipeline(
            "sentiment-analysis",
            model=hf_model,
            device=device
        )
        self.lm = ps.LM()
        self.hiv4 = ps.HIV4()

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

        # ML sentiment..... DO THIS
        # pos_uni = ML_Unigrams_Positive(text)
        # pos_bi  = ML_Bigrams_Positive(text)
        # attr_obj.ML = pos_uni + pos_bi

        return attr_obj

