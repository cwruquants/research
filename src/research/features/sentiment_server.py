"""
Score transformer sentiment for several worker processes on one GPU process.

Several processes each holding the model on one GPU slow each other down (on an RX 6950 XT,
eight were six times slower than one), and the Python work around the model (parsing,
readability, the LM/HIV4/ML lexicons) holds the GIL, so threads do not help either. So:

- A ScoringServer process owns the model on the GPU. It pools the token ids of every
  waiting request into large length-sorted batches (SentimentSetup.score_encoded) and
  sends each worker its results.
- Each worker process keeps a CPU SentimentSetup for the tokenizer and the lexicons and
  scores documents through a RemoteSentiment, which has the same fit_all as SentimentSetup.

`research ingest --sentiment --device 0 --workers 8` uses this. On the 6950 XT with a
Ryzen 9 5900X it analyzed 7.7 transcripts a second, against 1.9 for one process.
"""

import queue
import time
from typing import Any, Callable, Dict, List, Optional

_STOP = None  # sentinel on the request queue


class RemoteSentiment:
    """A SentimentSetup stand-in for worker processes: the transformer runs on the server."""

    def __init__(self, local_setup, requests, replies, client_id: int):
        self.local = local_setup  # CPU SentimentSetup: tokenizer, lexicons and word lists
        self.requests, self.replies, self.client_id = requests, replies, client_id
        self._seq = 0

    def score_texts(self, texts: List[str]):
        if not texts:
            return []
        self._seq += 1
        self.requests.put((self.client_id, self._seq, self.local.encode(texts)))
        seq, results = self.replies.get()
        if isinstance(results, BaseException):
            raise RuntimeError("the sentiment scoring server failed") from results
        if seq != self._seq:
            raise RuntimeError(f"reply {seq} arrived for request {self._seq}")
        return results

    def fit_all(self, attr_obj):
        return self.local.fit_all(attr_obj, model_results=self.score_texts(self.local.model_texts(attr_obj)))


def serve_requests(
    requests, replies: List, score: Callable, min_sentences: int = 2048, max_wait: float = 0.02
) -> Dict[str, Any]:
    """Answer requests until the stop sentinel; returns counters.

    Each request is (client_id, seq, (input_ids, attention_mask)). Requests are pooled
    until they hold min_sentences texts or max_wait seconds pass, scored together with
    score(input_ids, attention_mask), and each client gets (seq, its results). If scoring
    fails, every waiting client gets the exception instead of hanging.
    """
    stats = {"batches": 0, "sentences": 0, "busy_seconds": 0.0}
    stopping = False
    while not stopping:
        item = requests.get()
        if item is _STOP:
            break
        pending, n = [item], len(item[2][0])
        deadline = time.monotonic() + max_wait
        while n < min_sentences:
            try:
                item = requests.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            if item is _STOP:
                stopping = True
                break
            pending.append(item)
            n += len(item[2][0])

        ids = [x for _, _, (i, _) in pending for x in i]
        masks = [x for _, _, (_, m) in pending for x in m]
        start = time.monotonic()
        try:
            results = score(ids, masks)
        except Exception as exc:  # pass it to the clients; they raise it
            for client_id, seq, _ in pending:
                replies[client_id].put((seq, exc))
            continue
        stats["busy_seconds"] += time.monotonic() - start
        stats["batches"] += 1
        stats["sentences"] += len(results)
        k = 0
        for client_id, seq, (client_ids, _) in pending:
            replies[client_id].put((seq, results[k : k + len(client_ids)]))
            k += len(client_ids)
    return stats


def _server_main(
    setup_options, max_tokens, pad_multiple, min_sentences, max_wait, requests, replies, ready, errors
):
    try:
        import torch

        torch.set_num_threads(1)
        from research.pipelines.ingest import default_sentiment_setup

        setup = default_sentiment_setup(setup_options)

        def score(ids, masks):
            return setup.score_encoded(ids, masks, max_tokens=max_tokens, pad_multiple=pad_multiple)

        ready.set()
    except BaseException as exc:
        errors.put(repr(exc))
        ready.set()
        raise
    serve_requests(requests, replies, score, min_sentences=min_sentences, max_wait=max_wait)


class ScoringServer:
    """The GPU scoring process, as a context manager. Start it before the workers."""

    def __init__(
        self,
        ctx,
        num_clients: int,
        device: int,
        max_tokens: Optional[int] = None,
        pad_multiple: Optional[int] = None,
        min_sentences: int = 2048,
        max_wait: float = 0.02,
        setup_options: Optional[Dict[str, Any]] = None,
    ):
        # batch_size only sizes pipeline calls, which the server does not make; a fixed
        # value skips the GPU memory probe
        from research.features.sentiment import GPU_MAX_TOKENS, GPU_PAD_MULTIPLE

        max_tokens = GPU_MAX_TOKENS if max_tokens is None else max_tokens
        pad_multiple = GPU_PAD_MULTIPLE if pad_multiple is None else pad_multiple
        options = {"batch_size": 32, **(setup_options or {}), "device": device}
        self.requests = ctx.Queue()
        self.replies = [ctx.Queue() for _ in range(num_clients)]
        self.client_ids = ctx.Queue()
        for i in range(num_clients):
            self.client_ids.put(i)
        self._ready, self._errors = ctx.Event(), ctx.Queue()
        self._process = ctx.Process(
            target=_server_main,
            args=(
                options,
                max_tokens,
                pad_multiple,
                min_sentences,
                max_wait,
                self.requests,
                self.replies,
                self._ready,
                self._errors,
            ),
            daemon=True,
        )

    def __enter__(self):
        self._process.start()
        self._ready.wait()
        if not self._errors.empty():
            raise RuntimeError(f"the sentiment scoring server did not start: {self._errors.get()}")
        return self

    def client_args(self):
        """What a worker needs to build its RemoteSentiment (see connect)."""
        return self.requests, self.replies, self.client_ids

    def __exit__(self, *exc):
        self.requests.put(_STOP)
        self._process.join(timeout=60)
        if self._process.is_alive():
            self._process.terminate()
        return False


def connect(local_setup, client_args) -> RemoteSentiment:
    """In a worker: claim a client id and return its RemoteSentiment."""
    requests, replies, client_ids = client_args
    client_id = client_ids.get()
    return RemoteSentiment(local_setup, requests, replies[client_id], client_id)
