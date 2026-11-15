"""Scoring sentiment in one process for several workers: fit_all with given results,
token-capped batching, and the request pooling of the scoring server."""

import queue
import threading

import pytest

pytest.importorskip("transformers")
pytest.importorskip("pysentiment2")
torch = pytest.importorskip("torch")

from test_sentiment_batching import TEXT, FakeTransformer, make_setup  # noqa: E402

from research.features.sentiment_server import RemoteSentiment, serve_requests  # noqa: E402
from research.transcripts.models import DocumentAttr  # noqa: E402


def doc(text=TEXT):
    return DocumentAttr(text, store_paragraphs=True, store_sentences=True, store_words=False)


def scores(d):
    return [s.sentiment for p in d.paragraphs for s in p.sentences], (d.sentiment, d.LM, d.HIV4, d.ML)


def test_fit_all_with_given_results_matches_fit_all():
    setup = make_setup()
    expected = scores(setup.fit_all(doc()))

    d = doc()
    results = FakeTransformer()(setup.model_texts(d))
    assert scores(setup.fit_all(d, model_results=results)) == expected


def test_fit_all_rejects_results_of_the_wrong_length():
    setup = make_setup()
    d = doc()
    with pytest.raises(ValueError):
        setup.fit_all(d, model_results=[{"label": "neutral", "score": 0.5}])


def test_model_texts_are_the_sentences_in_order():
    d = doc()
    assert make_setup().model_texts(d) == [s.text for p in d.paragraphs for s in p.sentences]


# --- score_encoded -------------------------------------------------------------------


class Features(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0

    def pad(self, features, pad_to_multiple_of=None, return_tensors=None):
        ids, masks = features["input_ids"], features["attention_mask"]
        width = max(len(i) for i in ids)
        if pad_to_multiple_of:
            width = -(-width // pad_to_multiple_of) * pad_to_multiple_of
        return Features(
            input_ids=torch.tensor([i + [0] * (width - len(i)) for i in ids]),
            attention_mask=torch.tensor([m + [0] * (width - len(m)) for m in masks]),
        )


class FakeModel:
    """Label from the first token: 1 positive, 2 negative, else neutral. Records batch shapes."""

    class config:
        id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(self):
        self.shapes = []

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        self.shapes.append(tuple(input_ids.shape))
        first = input_ids[:, 0]
        logits = torch.stack(
            [(first == 1) * 5.0, (first == 2) * 5.0, ((first != 1) & (first != 2)) * 5.0], dim=1
        )
        return type("Out", (), {"logits": logits})()


def scorer_setup():
    setup = make_setup()
    setup.transformer.tokenizer, setup.transformer.model = FakeTokenizer(), FakeModel()
    setup.transformer.device = "cpu"
    return setup


def test_score_encoded_keeps_order_and_caps_batches():
    setup = scorer_setup()
    lengths = [40, 3, 70, 3, 10, 33]
    ids = [[(k % 3) + 1] + [7] * (n - 1) for k, n in enumerate(lengths)]
    masks = [[1] * n for n in lengths]
    out = setup.score_encoded(ids, masks, max_tokens=128, pad_multiple=32)

    assert [r["label"] for r in out] == ["positive", "negative", "neutral", "positive", "negative", "neutral"]
    shapes = setup.transformer.model.shapes
    assert sum(rows for rows, _ in shapes) == len(ids)
    assert all(width % 32 == 0 and rows * width <= 128 for rows, width in shapes)
    assert [w for _, w in shapes] == sorted(w for _, w in shapes)  # shortest first
    assert setup.score_encoded([], []) == []


# --- the server's request loop --------------------------------------------------------


def label_by_text(ids, masks):
    return FakeTransformer()(ids)  # in these tests the "token ids" are the texts themselves


def run_server(score, n_clients, **kw):
    requests, replies = queue.Queue(), [queue.Queue() for _ in range(n_clients)]
    stats = {}
    thread = threading.Thread(target=lambda: stats.update(serve_requests(requests, replies, score, **kw)))
    thread.start()
    return requests, replies, thread, stats


def test_server_pools_requests_and_routes_replies():
    requests, replies, thread, stats = run_server(label_by_text, 2, min_sentences=5, max_wait=1.0)
    requests.put((0, 1, (["great one", "plain"], [None, None])))
    requests.put((1, 7, (["bad news", "great", "fine"], [None] * 3)))
    a, b = replies[0].get(timeout=5), replies[1].get(timeout=5)
    requests.put(None)
    thread.join(timeout=5)

    assert a[0] == 1 and [r["label"] for r in a[1]] == ["positive", "neutral"]
    assert b[0] == 7 and [r["label"] for r in b[1]] == ["negative", "positive", "neutral"]
    assert stats["batches"] == 1 and stats["sentences"] == 5  # both requests scored together


def test_server_passes_scoring_errors_to_the_client():
    def broken(ids, masks):
        raise MemoryError("out of GPU memory")

    requests, replies, thread, _ = run_server(broken, 1, min_sentences=1)
    setup = make_setup()
    setup.encode = lambda texts: (list(texts), [None] * len(texts))
    remote = RemoteSentiment(setup, requests, replies[0], 0)
    with pytest.raises(RuntimeError, match="server failed"):
        remote.fit_all(doc())
    requests.put(None)
    thread.join(timeout=5)


def test_remote_sentiment_scores_like_a_local_setup():
    requests, replies, thread, _ = run_server(label_by_text, 3, min_sentences=1000, max_wait=0.05)
    expected = scores(make_setup().fit_all(doc()))

    def client(i, out):
        setup = make_setup()
        setup.encode = lambda texts: (list(texts), [None] * len(texts))
        remote = RemoteSentiment(setup, requests, replies[i], i)
        out[i] = [scores(remote.fit_all(doc())) for _ in range(3)]

    out = {}
    clients = [threading.Thread(target=client, args=(i, out)) for i in range(3)]
    for c in clients:
        c.start()
    for c in clients:
        c.join(timeout=10)
    requests.put(None)
    thread.join(timeout=5)

    assert all(result == expected for i in range(3) for result in out[i])
