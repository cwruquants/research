"""NLTK sentence splitting, fetching the punkt_tab model the first time it is needed."""

import nltk

_ready = False


def sent_tokenize(text: str) -> list[str]:
    global _ready
    if not _ready:
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        _ready = True
    return nltk.sent_tokenize(text)
