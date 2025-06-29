class ExposureResults:
    """
        This class stores the results a unique (keyword document, DocumentAttr) pair.

        The results will be stored in a result dictionary
    """
    def __init__(self, keyword_doc, earnings_call ):
        
        self.keyword_doc = keyword_doc
        self.earnings_call = earnings_call
        # Result Dictionary to store the results (do we want to have a different data structure for this?)
        self.resultDict_direct = {}
        self.resultDict_cosine = {}

    def getDict(self, choice: str = ""):
        if str != "direct" or str != "cosine":
            raise ValueError("Must provide dictionary choice, either 'direct' or 'cosine'")
        if str == "direct":
            if self.resultDict_direct:
                return self.resultDict_direct
            else:
                raise ValueError("Direct dictionary has not been initialized yet.")
        else:
            if self.resultDict_cosine:
                return self.resultDict_cosine
            else:
                raise ValueError("Cosine dictionary has not been initialized yet.")
            
    def initializeDict(self):
        """
            Ideally, the parameters here would set up 
        """
        pass

    def export(self, path: str = ""):
        """
            Calling export will export the result dictionary to a path of choice.
        """
        pass

