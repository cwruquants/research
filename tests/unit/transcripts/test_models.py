from research.transcripts.models import DocumentAttr, ParagraphAttr, SentenceAttr, WordAttr
from research.transcripts.tokenize import (
    document_to_bigram,
    document_to_sentence,
    document_to_word,
    is_bigram,
)

TEXT = "Revenue grew this quarter. Costs fell.\n\nWe expect growth next year."


def test_document_counts():
    doc = DocumentAttr(TEXT, store_paragraphs=True, store_sentences=True)
    assert doc.num_paragraphs == 2
    assert doc.num_sentences == 3
    assert doc.num_words == 11
    assert [p.text for p in doc.paragraphs][1] == "We expect growth next year."


def test_paragraph_stores_sentences_only_when_asked():
    assert ParagraphAttr("One. Two.", store_sentences=False).sentences == []
    assert [s.text for s in ParagraphAttr("One. Two.", store_sentences=True).sentences] == ["One.", "Two."]


def test_scores_default_to_zero_and_serialise():
    sentence = SentenceAttr("A test sentence.")
    sentence.HIV4 = 0.2
    data = sentence.to_dict()
    assert data["text"] == "A test sentence." and data["HIV4"] == 0.2 and data["sentiment"] == 0.0
    assert WordAttr("hello").text == "hello"


def test_tokenizers():
    doc = DocumentAttr("Risk rose. Trade war fears grew.")
    assert document_to_word(doc) == ["Risk", "rose", "Trade", "war", "fears", "grew"]
    assert document_to_sentence(doc) == ["Risk rose.", "Trade war fears grew."]
    assert "Trade war" in document_to_bigram(doc)


def test_matching_and_counting_split_sentences_the_same_way():
    # A section that ends without a full stop must not run into the next one
    text = "Revenue grew. Thank you all\n\nOur first question is on margins. Go ahead"
    counted = DocumentAttr(text, store_paragraphs=True, store_sentences=True)
    assert document_to_sentence(DocumentAttr(text)) == [
        "Revenue grew.",
        "Thank you all",
        "Our first question is on margins.",
        "Go ahead",
    ]
    assert counted.num_sentences == len(document_to_sentence(counted)) == 4
    assert is_bigram("trade war") and not is_bigram("trade")
