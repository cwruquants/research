import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Dict
from pathlib import Path
from functions import extract_text

"""transcript_analysis.py
-------------------------------------------------------------
Compute basic analytics over an earnings‑call transcript.
The helper functions are intentionally simple and dependency‑light so
that they remain fast even for large files.
-------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# Ensure required NLTK models -------------------------------------------------
# ---------------------------------------------------------------------------
# Depending on your NLTK version, sentence/word tokenization relies on either
#   tokenizers/punkt/...          (pre‑3.8)
#   tokenizers/punkt_tab/...      (>=3.8)
# We defensively check for both.  If missing we download quietly at runtime.

REQUIRED_RESOURCES = [
    "tokenizers/punkt",            # classic location (English)
    "tokenizers/punkt_tab/english" # new location introduced in NLTK 3.8
]

for res in REQUIRED_RESOURCES:
    try:
        nltk.data.find(res)
    except LookupError:
        # The resource name that `nltk.download` expects is just the last part
        # after the slash ("punkt" / "punkt_tab"), so we split on '/'.
        nltk.download(res.split("/")[-1], quiet=True)

# ---------------------------------------------------------------------------
# Pre‑compiled regular expressions & constants --------------------------------
# ---------------------------------------------------------------------------

number_regex = re.compile(r"\b\d+(?:[\.,]\d+)?\b")  # integers & decimals
question_regex = re.compile(r"\?")

plural_pronouns = {
    "we", "us", "our", "ours", "they", "them", "their", "theirs",
}

singular_pronouns = {
    "i", "me", "my", "mine", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its",
}

# Analyst introduction lines often follow one of these styles, among others:
#   John Smith — Foo Capital — Analyst
#   Jane Doe, Bar & Co. - Analyst
analyst_pattern = re.compile(
    r"""
    ^\s*                           # start of line + optional whitespace
    (?P<name>[A-Za-z\s'.-]+?)      # analyst name (non‑greedy capture)
    \s*[—,-]\s*                   # dash or comma separator (various dashes)
    .+?                             # company / affiliation (lazy)
    \s*[—,-]\s*                   # second separator
    .*Analyst                       # title containing the word "Analyst"
    """,
    re.MULTILINE | re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Metric helpers --------------------------------------------------------------
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Return the number of word tokens in *text*."""
    return len(word_tokenize(text))


def count_sentences(text: str) -> int:
    """Return the number of sentence tokens in *text*."""
    return len(sent_tokenize(text))


def number_to_words_ratio(text: str) -> float:
    """Ratio of numeric tokens (e.g. 42, 3.14) to total word tokens."""
    words = word_tokenize(text)
    if not words:
        return 0.0
    numbers = number_regex.findall(text)
    return len(numbers) / len(words)


def proportion_plural_pronouns(text: str) -> float:
    """Proportion of plural pronouns among all personal pronouns."""
    words = [w.lower() for w in word_tokenize(text)]
    plural = sum(1 for w in words if w in plural_pronouns)
    singular = sum(1 for w in words if w in singular_pronouns)
    total = plural + singular
    return plural / total if total else 0.0


def count_analysts(text: str) -> int:
    """Return the number of *unique* analysts identified in the transcript."""
    names = {
        match.group("name").strip().lower()
        for match in analyst_pattern.finditer(text)
    }
    return len(names)


def count_questions(text: str) -> int:
    """Naïve count of question marks as a proxy for questions."""
    return len(question_regex.findall(text))

# ---------------------------------------------------------------------------
# Public API ------------------------------------------------------------------
# ---------------------------------------------------------------------------

def analyze_transcript(file_path: str) -> Dict[str, float]:
    """Extract text from *file_path* and compute all metrics."""
    text = extract_text(file_path)
    return {
        "word_count": count_words(text),
        "sentence_count": count_sentences(text),
        "number_to_words_ratio": number_to_words_ratio(text),
        "proportion_plural_pronouns": proportion_plural_pronouns(text),
        "analyst_count": count_analysts(text),
        "question_count": count_questions(text),
    }

# ---------------------------------------------------------------------------
# Command‑line entry ----------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    default_path = Path("src/data/earnings_calls/ex1.xml")
    print(analyze_transcript(default_path))
