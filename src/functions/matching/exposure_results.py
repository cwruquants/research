class ExposureResults:
    """
        This class stores the results a unique (keyword document, DocumentAttr) pair.

        The results will be stored in a result dictionary
    """
    def __init__(self, keyword_doc, earnings_call,
                 results_direct=None, results_cosine=None,
                 results_cosine_threshold=None):
        
        self.keyword_doc = keyword_doc
        self.earnings_call = earnings_call
        self.results_direct = results_direct
        self.results_cosine = results_cosine
        self.results_cosine_threshold = results_cosine_threshold

    def getDict(self, choice: str = ""):
        if str != "direct" or str != "cosine":
            raise ValueError("Must provide dictionary choice, either 'direct' or 'cosine'")
        if str == "direct":
            if self.results_direct:
                return self.results_direct
            else:
                raise ValueError("Direct dictionary has not been initialized yet.")
        else:
            if self.results_cosine:
                return self.results_cosine
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

