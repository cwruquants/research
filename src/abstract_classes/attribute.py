from typing import Dict, Any

class Attr:
    def __init__(self):
        pass

    def to_dict(self):
        return self.__dict__
    
class WordAttr(Attr):
    def __init__(self, word=""):
        """ 
            This function will need to set up the features for the word.
        """
        super().__init__()
        # self.sentiment = sentiment
        # self.ML = ML
        # self.LM = LM
        # self.HIV4 = HIV4
        pass


class SentenceAttr(Attr):
    def __init__(self, List[WordAttr]):
        """ 
            This function will need to set up the features for the sentence.
        """
        super().__init__()
        pass


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

