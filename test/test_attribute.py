from typing import Dict, Any
from ..src.document.decompose_text import par_to_sentence, sentence_to_word, sentence_to_bigram


from ..src.document.abstract_classes.attribute import Attr, WordAttr, SentenceAttr, BigramAttr, ParagraphAttr, DocumentAttr

def test_attr_basic():
    print("Testing Attr...")
    a = Attr("Sample text")
    a.sentiment = 1.0
    a.ML = 0.5
    print(a.to_dict())

def test_word_attr():
    print("\nTesting WordAttr...")
    w = WordAttr("hello")
    w.LM = 0.8
    print(w.text)
    print(w.to_dict())

def test_sentence_attr():
    print("\nTesting SentenceAttr...")
    s = SentenceAttr("A test sentence.")
    s.HIV4 = 0.2
    print(s.text)
    print(s.to_dict())

def test_bigram_attr():
    print("\nTesting BigramAttr...")
    b = BigramAttr("big ram")
    b.sentiment = -0.3
    print(b.text)
    print(b.to_dict())

def test_paragraph_attr_store_sentences_false():
    print("\nTesting ParagraphAttr with store_sentences=False...")
    para = "Sentence one. Sentence two. Sentence three."
    p = ParagraphAttr(para, store_sentences=False)
    print(p.text)
    print("Sentences attribute:", p.sentences)
    print(p.to_dict())

def test_paragraph_attr_store_sentences_true():
    print("\nTesting ParagraphAttr with store_sentences=True...")
    para = "Sentence one. Sentence two. Sentence three."
    p = ParagraphAttr(para, store_sentences=True)
    print(p.text)
    print("Sentences attribute:", [s.text for s in p.sentences])
    print(p.to_dict())

def test_document_attr_store_paragraphs_false():
    print("\nTesting DocumentAttr with store_paragraphs=False...")
    doc = "Para one sentence. Second sentence.\n\nSecond paragraph."
    d = DocumentAttr(doc, store_paragraphs=False)
    print(d.text)
    print("Paragraphs attribute:", d.paragraphs)
    print(d.to_dict())

def test_document_attr_store_paragraphs_true():
    print("\nTesting DocumentAttr with store_paragraphs=True...")
    doc = "Para one sentence. Another sentence.\n\nSecond para here. And another."
    d = DocumentAttr(doc, store_paragraphs=True)
    print(d.text)
    print("Paragraphs attribute:", [p.text for p in d.paragraphs])
    print(d.to_dict())

def test_nested_usage():
    print("\nTesting nested usage: Document with Paragraphs storing Sentences...")
    doc = (
        "First para. Has two sentences.\n\n"
        "Second paragraph written. Next written here."
    )
    d = DocumentAttr(doc, store_paragraphs=True)
    # Now, for each paragraph, enable sentence storage
    for p in d.paragraphs:
        p.store_sentences = True
        p.sentences = [SentenceAttr(s.strip()) for s in p.text.split('.') if s.strip()]
    print(d.to_dict())

if __name__ == "__main__":
    test_attr_basic()
    test_word_attr()
    test_sentence_attr()
    test_bigram_attr()
    test_paragraph_attr_store_sentences_false()
    test_paragraph_attr_store_sentences_true()
    test_document_attr_store_paragraphs_false()
    test_document_attr_store_paragraphs_true()
    test_nested_usage()