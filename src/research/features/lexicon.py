"""Dictionary tone of a text: Loughran-McDonald (LM) and the Garcia et al. (2023) ML unigrams.

Each dictionary is a positive and a negative word list. A text's score is the number of its
words on the positive list minus the number on the negative list, so it is a count that grows
with the text's length (documents.lm_per_1000_words and ml_per_1000_words divide it out).

Words are matched exactly, lower-cased, with no stemming and no stopword list:

- LM: the repo's LM_Positive.csv and LM_Negative.csv, which already list every inflection of
  each word. Since 2026-11-01 the negative list leaves out "question" and "ill", which in
  earnings calls are analyst Q&A boilerplate and "I'll". (The pysentiment2 package this
  replaces stemmed each word first, so "quite" counted as negative "quit", dropped "good" and
  "strong" as surnames, and used its own LM copy that still had "question".)
- ML: the Garcia unigram lists. The paper counts each listed word unstemmed and divides by
  the document's word count (Garcia, Hu and Rohrer 2023, eq. 1). Its bigram lists are a
  separate dictionary and are not scored here.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Optional

from research import config

_WORD = re.compile(r"[a-z]+")


def read_word_list(path: Path) -> FrozenSet[str]:
    """One word per line (any case); blank lines are ignored."""
    with open(path, encoding="utf-8") as f:
        return frozenset(line.strip().lower() for line in f if line.strip())


@dataclass(frozen=True)
class Dictionary:
    positive: FrozenSet[str]
    negative: FrozenSet[str]

    def counts(self, words) -> Dict[str, int]:
        positive = negative = 0
        for word in words:
            positive += word in self.positive
            negative += word in self.negative
        return {"positive": positive, "negative": negative}


# dictionary -> (positive list, negative list), under data/word_sets
DICTIONARY_FILES = {
    "LM": ("LM_Positive.csv", "LM_Negative.csv"),
    "ML": ("Garcia_MLWords_ML_positive_unigram.csv", "Garcia_MLWords_ML_negative_unigram.csv"),
}


def _default_dictionaries(word_sets_dir: Path) -> Dict[str, Dictionary]:
    return {
        name: Dictionary(read_word_list(word_sets_dir / positive), read_word_list(word_sets_dir / negative))
        for name, (positive, negative) in DICTIONARY_FILES.items()
    }


class LexiconScorer:
    """Scores texts against the LM and ML dictionaries; build once and reuse."""

    def __init__(self, word_sets_dir: Optional[Path] = None):
        self.word_sets_dir = Path(word_sets_dir or config.WORD_SETS_DIR)
        self.dictionaries = _default_dictionaries(self.word_sets_dir)

    def files(self) -> Dict[str, Dict[str, Path]]:
        """The word list files behind each dictionary."""
        return {
            name: {"positive": self.word_sets_dir / positive, "negative": self.word_sets_dir / negative}
            for name, (positive, negative) in DICTIONARY_FILES.items()
        }

    def counts(self, text: str) -> Dict[str, Dict[str, int]]:
        """Positive and negative word counts for each dictionary."""
        words = _WORD.findall(text.lower())
        return {name: dictionary.counts(words) for name, dictionary in self.dictionaries.items()}

    def scores(self, text: str) -> Dict[str, Optional[float]]:
        """Positive minus negative count for each dictionary; None for a text with no words."""
        counts = self.counts(text)
        if not _WORD.search(text.lower()):
            return {name: None for name in counts}
        return {name: float(c["positive"] - c["negative"]) for name, c in counts.items()}
