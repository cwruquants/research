"""
Transformer sentiment for transcript sentences.

Each sentence gets a signed score from a financial-news sentiment model: the top label's
probability, positive or negative, and 0 for a neutral sentence. A call's stored
`documents.sentiment` is the mean over its sentences (see mean_sentiment).

Only SentimentSetup needs the [ml] extra (transformers, torch); the rest imports without it.
"""

from typing import List, Optional

from research.transcripts.models import DocumentAttr

HF_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
MAX_LENGTH = 512
# The pipeline's batch size on a CPU; pipeline_runs records it for such runs.
CPU_BATCH_SIZE = 32
# score_encoded's batches: at most this many padded tokens, padding rounded up to a multiple
# of GPU_PAD_MULTIPLE. Used on a GPU; pipeline_runs records them for such runs.
GPU_MAX_TOKENS = 4096
GPU_PAD_MULTIPLE = 32


class SentimentSetup:
    """The sentiment model on one device. Nothing is kept between calls, so threads can share one."""

    def __init__(self, device: int = -1, hf_model: str = HF_MODEL, max_length: int = MAX_LENGTH):
        """device is -1 for the CPU or a GPU index."""
        from transformers import pipeline

        self.transformer = pipeline(
            "sentiment-analysis",
            model=hf_model,
            device=device,
            truncation=True,
            max_length=max_length,
        )
        self.max_length = max_length

    @property
    def on_gpu(self) -> bool:
        """True when the transformer runs on a GPU (it then scores through score_encoded)."""
        return getattr(getattr(self.transformer, "device", None), "type", "cpu") != "cpu"

    def sentence_scores(self, texts: List[str]) -> List[float]:
        """The signed score of each text, in input order."""
        return [signed_score(result) for result in self._hf_infer(texts)]

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

            # A dataset avoids the "using pipelines sequentially on gpu" warning
            dataset = Dataset.from_dict({"text": texts})
            return list(self.transformer(KeyDataset(dataset, "text"), batch_size=CPU_BATCH_SIZE))
        except ImportError:
            return self.transformer(texts, batch_size=CPU_BATCH_SIZE)

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
        import torch

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


def signed_score(hf_res) -> float:
    """One transformer result as a signed score: +score positive, -score negative, 0 neutral."""
    if isinstance(hf_res, list):
        hf_res = hf_res[0]
    label = hf_res.get("label", "neutral").lower()
    if label == "neutral":
        return 0.0
    score = hf_res.get("score", 0.0)
    return float(score if label == "positive" else -score)


def sentence_texts(doc: DocumentAttr) -> List[str]:
    """The sentences of a call (a DocumentAttr with paragraphs and sentences), in order."""
    return [s.text for p in doc.paragraphs for s in p.sentences]


def mean_sentiment(scores: List[float]) -> Optional[float]:
    """
    A call's sentiment: the mean of its sentence scores, so a long Q&A weighs more than a
    short Presentation. None for a call with no sentences.
    """
    return float(sum(scores) / len(scores)) if scores else None
