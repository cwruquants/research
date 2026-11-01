"""Checks multi-word keywords in direct matching and section cleaning in load_document."""

import pytest

from src.analysis.match_extraction.matching_agent import MatchingAgent
from src.document.abstract_classes.attribute import DocumentAttr
from src.document.decompose_transcript import load_document


def _agent(tmp_path, keywords, **kwargs):
    path = tmp_path / "keywords.csv"
    path.write_text("\n".join(keywords) + "\n", encoding="utf-8")
    return MatchingAgent(keywords_path=str(path), **kwargs)


def _match(agent, text):
    agent.document = DocumentAttr(document=text)
    results = agent._direct_match()
    return {
        keyword: [(m.matched_text, m.position) for m in km.direct_matches]
        for keyword, km in results.keyword_matches.items()
        if km.direct_matches
    }


def test_phrase_matches_consecutive_words_case_insensitively(tmp_path):
    agent = _agent(tmp_path, ["greenhouse gas"])
    assert _match(agent, "Our Greenhouse Gas output. Gas greenhouse is not one.") == {
        "greenhouse gas": [("Greenhouse Gas", 1)]
    }


def test_single_words_still_match(tmp_path):
    agent = _agent(tmp_path, ["risk", "carbon tax"])
    assert _match(agent, "The risk of a carbon tax.") == {
        "risk": [("risk", 1)],
        "carbon tax": [("carbon tax", 4)],
    }


def test_exact_mode_does_not_cross_stopwords_or_plurals(tmp_path):
    agent = _agent(tmp_path, ["want make", "carbon emission"])
    assert _match(agent, "We want to make fewer carbon emissions.") == {}


def test_normalized_mode_crosses_stopwords_and_plurals(tmp_path):
    agent = _agent(tmp_path, ["want make", "carbon emission"], normalize_phrases=True)
    assert _match(agent, "We want to make fewer carbon emissions.") == {
        "want make": [("want to make", 1)],
        "carbon emission": [("carbon emissions", 5)],
    }


def test_normalized_mode_keeps_phrases_with_stopwords_exact(tmp_path):
    agent = _agent(tmp_path, ["able to"], normalize_phrases=True)
    assert _match(agent, "We are able to grow.") == {"able to": [("able to", 2)]}


RULE = "-" * 70
BAR = "=" * 70


@pytest.mark.parametrize("gap", ["", "\n"], ids=["tight", "spaced"])
def test_load_document_keeps_answers_and_drops_speakers(tmp_path, gap):
    # Tight files have no blank lines around one-line turns, which used to lose the answers
    def turn(name, said):
        return f"{RULE}\n{name}\n{RULE}\n{gap}{said}\n{gap}"

    body = (
        f"{BAR}\nPresentation\n{RULE}\nOperator   [1]\n{RULE}\n{gap}Welcome to the call.\n{gap}"
        f"{BAR}\nQuestions and Answers\n"
        + turn("Operator   [1]", "First question please.")
        + turn("Ann Lee, Bank - Analyst   [2]", "How are margins?")
        + turn("Bo Ray, Co - CFO   [3]", "Margins are improving.")
    )
    path = tmp_path / "1_T.xml"
    path.write_text(f"<Event><EventStory><Body>{body}</Body></EventStory></Event>", encoding="utf-8")

    text = load_document(str(path)).text
    assert "Welcome to the call." in text
    assert "How are margins?" in text
    assert "Margins are improving." in text
    assert "Ann Lee" not in text and "Bo Ray" not in text and "[" not in text
