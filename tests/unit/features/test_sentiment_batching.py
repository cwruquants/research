"""fit_all scores a whole document in one length-sorted transformer call."""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("transformers")
pytest.importorskip("pysentiment2")

import pysentiment2 as ps  # noqa: E402

from research.features.sentiment import SentimentSetup  # noqa: E402
from research.transcripts.models import DocumentAttr  # noqa: E402


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
    setup.lm, setup.hiv4 = ps.LM(), ps.HIV4()
    setup.batch_size, setup.max_length = 32, 512
    setup.ml_words_positive, setup.ml_words_negative = [], []
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


def test_one_call_sorted_by_length_and_results_in_order():
    setup = make_setup()
    doc = DocumentAttr(TEXT, store_paragraphs=True, store_sentences=True, store_words=False)
    setup.fit_all(doc)

    assert len(setup.transformer.calls) == 1
    batch = setup.transformer.calls[0]
    assert [len(t) for t in batch] == sorted(len(t) for t in batch)

    got = [s.sentiment for p in doc.paragraphs for s in p.sentences]
    assert got == per_sentence_reference(make_setup(), TEXT)


def test_threads_can_share_one_setup():
    # `research ingest --sentiment --threads N` scores documents on N threads with one setup
    setup = make_setup()
    text = "\n\n".join([TEXT] * 60)
    docs = [
        DocumentAttr(text, store_paragraphs=True, store_sentences=True, store_words=False) for _ in range(8)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(setup.fit_all, docs))  # re-raises a failure from any thread

    expected = per_sentence_reference(make_setup(), text)
    for doc in docs:
        assert [s.sentiment for p in doc.paragraphs for s in p.sentences] == expected


def test_hf_infer_keeps_input_order():
    setup = make_setup()
    texts = ["a much longer bad sentence here", "great", "neutral words"]
    labels = [r["label"] for r in setup._hf_infer(texts)]
    assert labels == ["negative", "positive", "neutral"]
    assert setup._hf_infer([]) == []
