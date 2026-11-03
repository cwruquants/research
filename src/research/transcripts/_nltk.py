"""NLTK data the pipeline needs, fetched the first time it is needed.

Steps that start threads or worker processes call ensure_nltk_data() first, because
concurrent nltk.download() calls overwrite the same zip and fail.
"""

import threading

import nltk

# NLTK resource path for each downloadable package: punkt_tab splits sentences and
# cmudict counts syllables for the textstat readability scores
RESOURCES = {"punkt_tab": "tokenizers/punkt_tab", "cmudict": "corpora/cmudict"}

_lock = threading.Lock()
_ready = set()


def ensure_nltk_data(packages=tuple(RESOURCES)) -> None:
    """Download any of the packages that are missing; safe to call from several threads."""
    with _lock:
        for package in packages:
            if package in _ready:
                continue
            try:
                nltk.data.find(RESOURCES[package])
            except LookupError:
                if not nltk.download(package, quiet=True):
                    raise RuntimeError(f"Could not download the NLTK package {package!r}")
            _ready.add(package)


def sent_tokenize(text: str) -> list[str]:
    if "punkt_tab" not in _ready:
        ensure_nltk_data(("punkt_tab",))
    return nltk.sent_tokenize(text)
