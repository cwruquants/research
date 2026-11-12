import pytest

from research.features.lexicon import LexiconScorer


@pytest.fixture(scope="module")
def scorer():
    return LexiconScorer()


def test_lm_counts_words_exactly(scorer):
    counts = scorer.counts("Results were strong and good. Margins declined; losses widened.")["LM"]
    # strong and good used to be dropped as surnames; declined is matched without stemming
    assert counts == {"positive": 2, "negative": 2}


@pytest.mark.parametrize(
    "text",
    [
        "Thank you. Our next question comes from the line of Sam. Two questions, please.",
        "I'll take that one, and I think it is quite clear.",  # 'quite' used to stem to negative 'quit'
        "We issued a press release and signed a new contract.",
    ],
)
def test_lm_ignores_call_boilerplate(scorer, text):
    assert scorer.counts(text)["LM"]["negative"] == 0


def test_ml_counts_single_words(scorer):
    # A listed word scores on its own; the old rule needed two listed words side by side
    lists = scorer.dictionaries["ML"]
    positive = sorted(lists.positive)[0]
    negative = sorted(lists.negative)[0]
    counts = scorer.counts(f"It was {positive}. Then it was {negative} and {negative} again.")["ML"]
    assert counts == {"positive": 1, "negative": 2}


def test_scores_are_net_counts_and_none_for_no_words(scorer):
    assert scorer.scores("Strong demand, good margins, weak pricing.")["LM"] == 1.0
    assert scorer.scores("") == {"LM": None, "ML": None}
    assert scorer.scores("  --  123 ") == {"LM": None, "ML": None}
