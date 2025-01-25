'''
    CWRU Quants Research Vertical Exposure Project
'''

# Import statements


# Functions
def extract_text() -> str:
    """
        This function takes in an earnings transcript as an input, and extracts the words from the transcript.

        Input: .xml file
        Output: str
    """

    return ""


def extract_exposure(exposure_words, txt_string, buffer) -> dict:
    """
        This takes in the str returned from extract_text, and extracts regions (+- buffer) where the exposure
        words exist. 

        For example, if buffer = 5, then wherever we identify an exposure word, we take the substring of words beginning
        5 words before exposure word, and 5 words after the exposure words. This would create a string with 11 words. 
        We would then add this to our return dict.

        Input: 
        - exposure_words: csv
        - txt_string: str
        - buffer: int

        Output:
        dict containing strings of words containing exposure words. These will later be analyzed with sentiment analysis processes
    """

    return {}

def sentiment_score():
    """
        Returns sentiment score using roBERTa method for positive/negative/neutral sentiment surrounding our exposure words.
    """

    return 0


