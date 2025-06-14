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

    
        