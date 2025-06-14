from typing import Dict, Any

class Attr:
    def __init__(self):
        pass

    def to_dict(self):
        return self.__dict__
    
class WordAttr(Attr):
    def __init__(self, word : str = "", sentiment : float = None, ML : float = None, LM : float = None, HIV4 : float = None):
        """ 
            This function will need to set up the features for the word.

            Args:
                - word (str)
                - sentiment (float or None)
                - ML (float or None)
                - LM (float or None)
                - HIV4 (float or None)
        """
        super().__init__()
        self.word = word
        self.sentiment = sentiment
        self.ML = ML
        self.LM = LM
        self.HIV4 = HIV4
    
    """
    Adds a property named "text" required for processing
    """
    @property
    def text(self):
        return self.word
    
    def to_dict(self):
        """
        Overrides the to_dict function from the abstract class. Returns a dictionary containing the word and its features

        Returns:
            dict: containing the word itself and its features
                - word (str)
                - setiment (float or None)
                - ML (float or None)
                - LM (float or None)
                - HIV4 (float or None)
        """
        return {"word" : self.word, "sentiment" : self.sentiment, "ML" : self.ML, "LM" : self.LM, "HIV4" : self.HIV4}
        


class SentenceAttr(Attr):
    def __init__(self, words: list[WordAttr], sentence : str = "", sentiment : float = None, ML : float = None, LM : float = None, HIV4 : float = None):
        """ 
            This function will need to set up the features for the sentence.

            Args:
                - words (list)
                - sentence (str)
                - sentiment (float or None)
                - ML (float or None)
                - LM (float or None)
                - HIV4 (float or None)
        """
        super().__init__()
        self.words = words
        self.sentence = sentence
        self.sentiment = sentiment
        self.ML = ML
        self.LM = LM
        self.HIV4 =HIV4

        """
        Checks if the list of words passed in make up the sentence
        """
        words_joined = " ".join(w.word for w in words).strip()
        if words_joined != self.sentence.strip():
            raise ValueError("Word list does not make up the sentence")

    """
    Adds a property named "text" required for processing
    """
    @property
    def text(self):
        return self.sentence

    def to_dict(self):
        """
        Overrides the to_dict function from the abstract class. Returns a dictionary containing the senttence and its features

        Returns:
            dict: containing the word itself and its features
                - words (list)
                - sentence (str)
                - setiment (float or None)
                - ML (float or None)
                - LM (float or None)
                - HIV4 (float or None)
        """
        return {"words" : [word.to_dict() for word in self.words], "sentence" : self.sentence, "sentiment" : self.sentiment, "ML" : self.ML, "LM" : self.LM, "HIV4" : self.HIV4}


class ParagraphAttr(Attr):
    def __init__(self, paragraph=""):
        """ 
            This function will need to set up the features for the sentence.
        """
        super().__init__()
        pass


class DocumentAttr(Attr):
    def __init__(self, document=""):
        """ 
            This function will need to set up the features for the paragraph.
        """
        super().__init__()
        pass

