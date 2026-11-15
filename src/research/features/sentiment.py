import json
import re
from typing import List, Union

import numpy as np
import pandas as pd
import pysentiment2 as ps
from transformers import pipeline

from research.transcripts.models import Attr, DocumentAttr, SentenceAttr

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

# score_encoded's batches: at most this many padded tokens, padding rounded up to a multiple
# of GPU_PAD_MULTIPLE. Used on a GPU; pipeline_runs records them for such runs.
GPU_MAX_TOKENS = 4096
GPU_PAD_MULTIPLE = 32


class SentimentSetup:
    def __init__(
        self,
        sheet_name_positive: str = "ML_positive_unigram",
        sheet_name_negative: str = "ML_negative_unigram",
        ml_wordlist_path: str = "data/word_sets/original_xlsx/Garcia_MLWords.xlsx",
        hf_model: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        device: int = -1,
        batch_size: Union[int, str] = "auto",
        max_length: int = 512,
    ):
        """
        Args:
            sheet_name_positive (str): Excel sheet name for list of positive unigrams (for ML sentiment).
                Default is "ML_positive_unigram".
            sheet_name_negative (str): Excel sheet name for list of negative unigrams (for ML sentiment).
                Default is "ML_negative_unigram".
            ml_wordlist_path (str): Path to Excel wordlist file (MLWords).
                Default is "data/word_sets/original_xlsx/Garcia_MLWords.xlsx".
            hf_model (str): Huggingface model name for transformer-based sentiment analysis.
                Default is "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis".
            device (int): Device for transformer inference. -1 for CPU, 0 for first CUDA GPU, etc.
                Default is -1 (CPU).
            batch_size (Union[int, str]): Batch size for transformer model. Set "auto" to determine dynamically.
                Default is "auto".
            max_length (int): Maximum sequence length for transformer inputs. Default is 512.
        """
        self.transformer = pipeline(
            "sentiment-analysis",
            model=hf_model,
            device=device,
            truncation=True,
            max_length=max_length,
        )
        self.lm = ps.LM()
        self.hiv4 = ps.HIV4()
        self.batch_size = batch_size
        self.max_length = max_length

        self.ml_words_positive = self.excel_to_list(ml_wordlist_path, sheet_name_positive)
        self.ml_words_negative = self.excel_to_list(ml_wordlist_path, sheet_name_negative)

        # Resolve auto batch size after pipeline is initialized (so device info is available)
        if isinstance(self.batch_size, str) and self.batch_size.lower() == "auto":
            self.batch_size = self._auto_determine_batch_size()

        print(f"[SentimentSetup] Using HuggingFace model: {hf_model}")
        print(f"[SentimentSetup] Resolved batch_size: {self.batch_size}")

    @property
    def on_gpu(self) -> bool:
        """True when the transformer runs on a GPU (it then scores through score_encoded)."""
        return getattr(getattr(self.transformer, "device", None), "type", "cpu") != "cpu"

    def _hf_infer(self, texts: List[str]):
        """Score texts with the transformer, returning results in input order.

        Texts are run shortest first so each batch holds texts of similar length and
        needs little padding; the results are then put back in the original order.
        """
        if not texts:
            return []
        if self.on_gpu:
            # On a GPU, token-capped batches are faster than the pipeline's fixed batch size
            # (0.67 vs 1.9 transcripts a second on an RX 6950 XT); scores agree within 1e-6
            return self.score_encoded(*self.encode(texts))
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        scored = self._run_transformer([texts[i] for i in order])
        results = [None] * len(texts)
        for position, index in enumerate(order):
            results[index] = scored[position]
        return results

    def _run_transformer(self, texts: List[str]):
        try:
            from datasets import Dataset
            from transformers.pipelines.pt_utils import KeyDataset

            # Create a dataset to avoid "using pipelines sequentially on gpu" warning
            dataset = Dataset.from_dict({"text": texts})
            # Convert to list to ensure it's consumable/indexable (needed for single-item access in fit())
            return list(self.transformer(KeyDataset(dataset, "text"), batch_size=self.batch_size))
        except ImportError:
            # Fallback if datasets library is missing
            return self.transformer(texts, batch_size=self.batch_size)

    def _auto_determine_batch_size(self) -> int:
        """
        Determine a reasonable maximum batch size based on available GPU memory by probing
        the pipeline with increasing batch sizes and catching CUDA OOM. Falls back to 32
        on CPU or if torch is not available.
        """
        try:
            pipeline_device_type = getattr(getattr(self.transformer, "device", None), "type", None)
            if torch is None or pipeline_device_type != "cuda":
                return 32
        except Exception:
            return 32

        # Heuristic initial estimate based on free memory and sequence length
        try:
            free_bytes, _ = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
            free_gb = free_bytes / (1024**3)
        except Exception:
            free_gb = 4.0

        # Base guess roughly scales with memory and inverse with sequence length
        length_scale = max(1, 256 // max(1, int(self.max_length)))
        # Remove arbitrary 128 cap on initial guess, allow scaling with VRAM
        initial_guess = int(max(4, 16 * free_gb * length_scale))

        # Probe by doubling until OOM or cap, with a small number of attempts
        upper_cap = 4096
        best_working = 1
        candidate = max(1, min(initial_guess, upper_cap))
        attempts = 0

        # Use a dummy text roughly the size of max_length to ensure the batch size
        # is safe for worst-case inputs (assuming ~4 chars per token)
        test_text = "word " * max(1, int(self.max_length))

        while attempts < 5:
            batch_candidate = max(1, min(candidate, upper_cap))
            sample_inputs = [test_text] * batch_candidate
            try:
                if torch is not None:
                    with torch.inference_mode():
                        _ = self.transformer(sample_inputs, batch_size=len(sample_inputs))
                else:
                    _ = self.transformer(sample_inputs, batch_size=len(sample_inputs))
                best_working = batch_candidate
                candidate = batch_candidate * 2
            except RuntimeError as e:
                message = str(e).lower()
                if "out of memory" in message or "cuda" in message:
                    if torch is not None:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    # Reduce candidate
                    if batch_candidate == 1:
                        break
                    candidate = max(1, batch_candidate // 2)
                else:
                    # Unknown error; stop probing and use last best
                    break
            except Exception:
                # Non-CUDA error; stop probing
                break
            attempts += 1

        return int(max(1, min(best_working, upper_cap)))

    def fit(self, attr_obj: Attr):
        """Score one node from its own text. Only a sentence gets a transformer score."""
        text = getattr(attr_obj, "text", None)
        if text is None:
            raise ValueError("Attr object does not have a text")

        # Only apply HuggingFace model to short text (SentenceAttr)
        if isinstance(attr_obj, SentenceAttr):
            hf_results = self._hf_infer([text])
            attr_obj.sentiment = _signed_score(hf_results[0]) if hf_results else 0.0

        self._fit_lexicons(attr_obj)
        return attr_obj

    def _fit_lexicons(self, attr_obj: Attr):
        """Set the LM, HIV4 and Garcia ML scores of one node from its own text."""
        text = attr_obj.text

        # LM sentiment
        lm_scores = self.lm.get_score(self.lm.tokenize(text))
        attr_obj.LM = lm_scores["Positive"] - lm_scores["Negative"]

        # HIV4 sentiment
        hiv4_scores = self.hiv4.get_score(self.hiv4.tokenize(text))
        attr_obj.HIV4 = hiv4_scores["Positive"] - hiv4_scores["Negative"]

        # ML sentiment (custom bigram-based)
        ml_tokens = re.findall(r"\b\w+\b", text.lower())
        attr_obj.ML = 0
        for i in range(len(ml_tokens) - 1):
            if ml_tokens[i] in self.ml_words_negative and ml_tokens[i + 1] in self.ml_words_negative:
                attr_obj.ML -= 1
            if ml_tokens[i] in self.ml_words_positive and ml_tokens[i + 1] in self.ml_words_positive:
                attr_obj.ML += 1

    def fit_all(self, attr_obj: Attr, model_results=None):
        """Score attr_obj and everything under it.

        All sentences under attr_obj go through the transformer in one call, so a whole
        transcript is batched together instead of one paragraph at a time. Nothing is kept
        on self between calls, so threads can share one SentimentSetup.

        model_results, if given, are the transformer results for model_texts(attr_obj), in
        that order, computed elsewhere (for example by a ScoringServer); the transformer is
        then not called here.
        """
        sentences = list(_sentences_under(attr_obj))
        if model_results is None:
            model_results = self._hf_infer([s.text for s in sentences])
        elif len(model_results) != len(sentences):
            raise ValueError(f"got {len(model_results)} model results for {len(sentences)} sentences")
        for sentence, hf_res in zip(sentences, model_results):
            sentence.sentiment = _signed_score(hf_res)
        return self._fit_tree(attr_obj)

    @staticmethod
    def model_texts(attr_obj: Attr) -> List[str]:
        """The texts fit_all sends to the transformer for attr_obj, in order."""
        return [s.text for s in _sentences_under(attr_obj)]

    def encode(self, texts: List[str]):
        """Tokenize texts as the pipeline does (truncated to max_length), without padding.

        Returns (input_ids, attention_mask) as lists of lists, for score_encoded. Workers
        call this so the process holding the GPU only pads and runs the model.
        """
        if not texts:
            return [], []
        enc = self.transformer.tokenizer(list(texts), truncation=True, max_length=self.max_length)
        return enc["input_ids"], enc["attention_mask"]

    def score_encoded(
        self,
        input_ids,
        attention_mask,
        max_tokens: int = GPU_MAX_TOKENS,
        pad_multiple: int = GPU_PAD_MULTIPLE,
    ):
        """Transformer results for already-tokenized texts, in input order.

        Same output as the pipeline (softmax over the logits, top label and its
        probability), but texts are sorted by token length and cut into batches of at most
        max_tokens padded tokens, with padding rounded up to pad_multiple so the GPU sees
        few distinct shapes. On an RX 6950 XT this ran about 3.7x faster than the pipeline
        on transcript sentences; scores differ from it by under 1e-6.
        """
        if not input_ids:
            return []
        tok, model, device = self.transformer.tokenizer, self.transformer.model, self.transformer.device
        labels = model.config.id2label
        padded = [-(-len(ids) // pad_multiple) * pad_multiple for ids in input_ids]
        order = sorted(range(len(input_ids)), key=lambda i: len(input_ids[i]))

        batches, current, longest = [], [], 0
        for i in order:
            if current and max(longest, padded[i]) * (len(current) + 1) > max_tokens:
                batches.append(current)
                current, longest = [], 0
            current.append(i)
            longest = max(longest, padded[i])
        if current:
            batches.append(current)

        results = [None] * len(input_ids)
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                features = tok.pad(
                    {
                        "input_ids": [input_ids[i] for i in batch],
                        "attention_mask": [attention_mask[i] for i in batch],
                    },
                    pad_to_multiple_of=pad_multiple,
                    return_tensors="pt",
                ).to(device)
                probs = torch.softmax(model(**features).logits.float(), dim=-1).cpu().numpy()
                top = probs.argmax(axis=1)
                for row, i in enumerate(batch):
                    results[i] = {"label": labels[int(top[row])], "score": float(probs[row, top[row]])}
        return results

    def _fit_tree(self, attr_obj: Attr):
        """Lexicon scores for each sentence, then every parent from its children:
        sentiment is their mean, the lexicon scores their sum."""
        paragraphs = getattr(attr_obj, "paragraphs", None)
        sentences = getattr(attr_obj, "sentences", None)
        if paragraphs:
            children = paragraphs
            for paragraph in children:
                self._fit_tree(paragraph)
        elif sentences:
            children = sentences
            for sentence in children:
                self._fit_lexicons(sentence)
        else:
            return self.fit(attr_obj)  # a node with nothing under it

        sentiments = [c.sentiment for c in children if c.sentiment is not None]
        attr_obj.sentiment = float(np.mean(sentiments)) if sentiments else 0.0
        for score in ["HIV4", "ML", "LM"]:
            setattr(attr_obj, score, float(sum(getattr(c, score, 0.0) for c in children)))
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

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialize(attr_obj), f, indent=2, ensure_ascii=False)

    def excel_to_list(self, file_path, sheet_name) -> list:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        values = df[0].tolist()
        normalized_list = []
        for val in values:
            if pd.isna(val) or str(val).strip() == "":
                continue
            normalized_list.append(str(val).replace("_", " ").lower())
        return normalized_list

    # TODO: Diliya
    def find_weight(corpus: DocumentAttr = None):
        pass


def _signed_score(hf_res) -> float:
    """One transformer result as a signed score: +score positive, -score negative, 0 neutral."""
    if isinstance(hf_res, list):
        hf_res = hf_res[0]
    label = hf_res.get("label", "neutral").lower()
    if label == "neutral":
        return 0.0
    score = hf_res.get("score", 0.0)
    return float(score if label == "positive" else -score)


def _sentences_under(attr_obj):
    """Yield, in order, every sentence that sits in a `sentences` list under attr_obj."""
    if getattr(attr_obj, "paragraphs", None):
        for paragraph in attr_obj.paragraphs:
            yield from _sentences_under(paragraph)
    elif getattr(attr_obj, "sentences", None):
        yield from attr_obj.sentences
