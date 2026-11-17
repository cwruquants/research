"""sentence_scores scores a whole call in one length-sorted transformer call."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from research.features.sentiment import SentimentSetup, mean_sentiment, sentence_texts
from research.transcripts.models import DocumentAttr


class FakeTransformer:
    """Labels by text so results can be checked; records every batch it receives."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, batch_size=None):
        texts = [t["text"] if isinstance(t, dict) else t for t in texts]
        texts = list(texts)
        self.calls.append(texts)
        out = []
        for t in texts:
            if "great" in t:
                out.append({"label": "positive", "score": 0.9})
            elif "bad" in t:
                out.append({"label": "negative", "score": 0.8})
            else:
                out.append({"label": "neutral", "score": 0.5})
        return out


def make_setup():
    setup = SentimentSetup.__new__(SentimentSetup)
    setup.transformer = FakeTransformer()
    setup.max_length = 512
    return setup


def per_sentence_reference(setup, text):
    """Score each sentence on its own, the way the old code's results should come out."""
    doc = DocumentAttr(text, store_paragraphs=True, store_sentences=True, store_words=False)
    expected = []
    for p in doc.paragraphs:
        for s in p.sentences:
            r = setup.transformer([s.text])[0]
            expected.append(
                0.0 if r["label"] == "neutral" else (r["score"] if r["label"] == "positive" else -r["score"])
            )
    return expected


TEXT = (
    "Revenue was great this quarter and margins held up well across every segment. Costs were bad.\n\n"
    "We expect demand to stay level.\n\n"
    "A great year. The outlook for the second half is bad because of weather and freight delays."
)


def doc(text=TEXT):
    return DocumentAttr(text, store_paragraphs=True, store_sentences=True, store_words=False)


def test_one_call_sorted_by_length_and_results_in_order():
    setup = make_setup()
    got = setup.sentence_scores(sentence_texts(doc()))

    assert len(setup.transformer.calls) == 1
    batch = setup.transformer.calls[0]
    assert [len(t) for t in batch] == sorted(len(t) for t in batch)
    assert got == per_sentence_reference(make_setup(), TEXT)


def test_sentence_texts_are_every_sentence_in_order():
    d = doc()
    assert sentence_texts(d) == [s.text for p in d.paragraphs for s in p.sentences]
    assert len(sentence_texts(d)) == d.num_sentences


def test_mean_sentiment():
    assert mean_sentiment([0.9, -0.8, 0.0]) == pytest.approx(0.1 / 3)
    assert mean_sentiment([]) is None


def test_threads_can_share_one_setup():
    # `research ingest --sentiment --threads N` scores documents on N threads with one setup
    setup = make_setup()
    text = "\n\n".join([TEXT] * 60)
    texts = [sentence_texts(doc(text)) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(setup.sentence_scores, texts))  # re-raises a failure from any thread

    expected = per_sentence_reference(make_setup(), text)
    assert all(scores == expected for scores in results)


def test_hf_infer_keeps_input_order():
    setup = make_setup()
    texts = ["a much longer bad sentence here", "great", "neutral words"]
    labels = [r["label"] for r in setup._hf_infer(texts)]
    assert labels == ["negative", "positive", "neutral"]
    assert setup._hf_infer([]) == []
