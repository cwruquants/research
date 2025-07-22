import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class Attr:
    def __init__(self, text):
        self.text = text
        self.sentiment = 0.0
        self.ML = 0.0
        self.LM = 0.0
        self.HIV4 = 0.0

    def to_dict(self):
        return {
            "text": self.text,
            "sentiment": self.sentiment,
            "ML": self.ML,
            "LM": self.LM,
            "HIV4": self.HIV4
        }

class WordAttr(Attr):
    def __init__(self, word=""):
        super().__init__(word)

class SentenceAttr(Attr):
    def __init__(self, sentence="", store_words=False):
        super().__init__(sentence)
        self.store_words = store_words
        self.words = None

        if self.store_words:
            # Stub for word-level parsing (you can add your tokenizer here)
            tokens = sentence.split()
            self.words = [WordAttr(w) for w in tokens]

    def to_dict(self):
        dt = super().to_dict()
        if self.words:
            dt["words"] = [w.to_dict() for w in self.words]
        return dt

class BigramAttr(Attr):
    def __init__(self, bigram=""):
        super().__init__(bigram)

class ParagraphAttr(Attr):
    def __init__(self, paragraph: str = "", store_sentences: bool = False, store_words: bool = False):
        super().__init__(paragraph)
        self.sentences = None
        self.store_sentences = store_sentences
        self.store_words = store_words

        if self.store_sentences:
            raw_sentences = sent_tokenize(paragraph)
            self.sentences = [SentenceAttr(s.strip(), store_words=self.store_words) for s in raw_sentences if s.strip()]

    def to_dict(self):
        dt = {
            "paragraph": self.text,
            "sentiment": self.sentiment,
            "ML": self.ML,
            "LM": self.LM,
            "HIV4": self.HIV4
        }
        if self.sentences:
            dt["sentences"] = [s.to_dict() for s in self.sentences]
        return dt

class DocumentAttr(Attr):
    def __init__(self, document: str = "", store_paragraphs: bool = False, store_sentences: bool = False, store_words: bool = False):
        super().__init__(document)
        self.paragraphs = None
        self.store_paragraphs = store_paragraphs
        self.store_sentences = store_sentences
        self.store_words = store_words

        if self.store_paragraphs:
            raw_paragraphs = document.split('\n\n')
            self.paragraphs = [
                ParagraphAttr(p.strip(), store_sentences=self.store_sentences, store_words=self.store_words)
                for p in raw_paragraphs if p.strip()
            ]

    def get_paragraph_texts(self):
        return [p.text for p in self.paragraphs] if self.paragraphs else []

    def to_dict(self):
        dt = {
            "document": self.text,
            "sentiment": self.sentiment,
            "ML": self.ML,
            "LM": self.LM,
            "HIV4": self.HIV4
        }
        if self.paragraphs:
            dt["paragraphs"] = [p.to_dict() for p in self.paragraphs]
        return dt
