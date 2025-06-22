from ....src.abstract_classes.attribute import DocumentAttr

class CosineSimAgent:
    def __init__(self, threshold = 0.7, csv_path = None):
        """
            Path is going to be a csv file of the exposure words.

            Ideally we would want our agent to *specialize* in this csv file. And be able to 
            analyze any earnings call document (DocAttr) that we give it.

            What does this agent need to be able to do?
            - take in csv
                - convert the csv file into a list that is usable by the agent
            - be able to take in any DocAttr, and return the DocAttr's exposure to its self.exposure_list
                - you should have a parameter, where you are able to *retrieve* the dictionary of WordAttr,
                    SentenceAttr, etc. 
            
        """

        self.exposure_list = None
        self.threshold = threshold

        pass

    def analyze(self, document: DocumentAttr = None):
        """
            This function should take in a DocumentAttr (which should already contain a list of ParAttrs, etc.) and
            return an ExposureResults class object that contains all of the information that we want to know about
            the analysis.

            So information that we want to know could be:
            - count of words matching our self.exposure_list over our given self.threshold.
            - list the sentences containing 
        """
        if not document:
            raise ValueError("Please provide a valid document attribute.")
        
        



class ExposureResults:
    def __init__():
        pass