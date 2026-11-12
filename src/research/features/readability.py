"""Readability scores for a transcript's text (textstat).

Linsear Write and textstat's text_standard are not computed: Linsear Write reads only a
text's first 100 words, and text_standard is a vote that includes it (dropped 2026-11-11).
"""

from typing import Any, Dict

import textstat


def readability_scores(text: str) -> Dict[str, Any]:
    """The textstat measures stored on public.documents, keyed as the uploader expects."""
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "difficult_words": textstat.difficult_words(text),
        "gunning_fog": textstat.gunning_fog(text),
    }
