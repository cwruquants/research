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
    def __init__(self, word : str = ""):
        """ 
            Constructor for word
        """
        super().__init__(word)

    def to_dict(self):
        return {
            "word" : self.text
            ,"sentiment": self.sentiment
            , "ML": self.ML
            , "LM": self.LM
            , "HIV4": self.HIV4
        }

class SentenceAttr(Attr):
    def __init__(self, sentence="", store_words : bool = False):
        """
            Constructor for sentence
        """
        super().__init__(sentence)
        self.words = None
        if store_words:
            self.words = [WordAttr(word) for word in sentence.split()]

    
    def to_dict(self):
        d =  {
            "sentence" : self.text
            ,"sentiment": self.sentiment
            , "ML": self.ML
            , "LM": self.LM
            , "HIV4": self.HIV4

        }
        if self.words:
            d["words"] = [word.to_dict() for word in self.words]
        
        return d


class BigramAttr(Attr):
    def __init__(self, bigram=""):
        """
            Constructor for bigram
        """
        super().__init__(bigram)

    
s = SentenceAttr("Hello i am bored", True)
print(s.to_dict())
s2 = SentenceAttr("Hello i am bored", False)
print(s2.to_dict())