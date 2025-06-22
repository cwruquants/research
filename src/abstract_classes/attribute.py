from typing import Dict, Any

class Attr:
    def __init__(self, text):
        self.text=text
        self.sentiment = 0.0
        self.ML = 0.0
        self.LM = 0.0
        self.HIV4 = 0.0
        pass

    def to_dict(self):
        return {
              "sentiment": self.sentiment
            , "ML": self.ML
            , "LM": self.LM
            , "HIV4": self.HIV4
        }
    
class WordAttr(Attr):
    def __init__(self, word=""):
        """ 
            Constructor for word
        """
        super().__init__(word)


class SentenceAttr(Attr):
    def __init__(self, sentence=""):
        """
            Constructor for sentence
        """
        super().__init__(sentence)


class BigramAttr(Attr):
    def __init__(self, bigram=""):
        """
            Constructor for bigram
        """
        super().__init__(bigram)

class ParagraphAttr(Attr):
    def __init__(self, paragraph: str = "", store_sentences: bool = False):
        """
        Constructor for ParagraphAttr.
        """
        super().__init__(paragraph)
        self.sentences = None
        if store_sentences:
            # sentence split by .
            self.sentences = [SentenceAttr(s.strip()) for s in paragraph.split('.') if s.strip()]

    def to_dict(self):
        dt = {
            "paragraph": self.text,
            "sentiment": self.sentiment,
            "ML": self.ML,
            "LM": self.LM,
            "HIV4": self.HIV4
        }
        if self.sentences:
            dt["sentences"] = [sent.to_dict() for sent in self.sentences]
        return dt
    
class DocumentAttr(Attr):
    def __init__(self, document: str = "", store_paragraphs: bool = False):
        """
        Constructor for DocumentAttr.
        """
        super().__init__(document)
        self.paragraphs = None
        if store_paragraphs:
            # split by \n newline
            self.paragraphs = [ParagraphAttr(p.strip()) for p in document.split('\n\n') if p.strip()]

    def to_dict(self):
        dt = {
            "document": self.text,
            "sentiment": self.sentiment,
            "ML": self.ML,
            "LM": self.LM,
            "HIV4": self.HIV4
        }
        if self.paragraphs:
            dt["paragraphs"] = [par.to_dict() for par in self.paragraphs]
        return dt
        