'''
    CWRU Quants Research Vertical Exposure Project
'''

# Import statements
import xml.etree.ElementTree as ET
import re
import os

# Functions
def extract_text(file_path: str) -> str:
    """

    This function takes in an earnings transcript as an input, and extracts the words from the transcript.

    Input: .xml file
    Output: str
    """
    
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract all text content from XML elements, including CDATA
    text_parts = []
    
    # Look specifically for Body and Headline elements which contain CDATA
    for event_story in root.findall('.//EventStory'):
        # Extract headline
        headline = event_story.find('Headline')
        if headline is not None and headline.text:
            text_parts.append(headline.text.strip())
        
        # Extract body
        body = event_story.find('Body')
        if body is not None and body.text:
            text_parts.append(body.text.strip())
    
    # Join all text parts with spaces
    raw_text = ' '.join(text_parts)
    
    # Clean up the text
    # Remove multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', raw_text)
    # Remove special characters and normalize whitespace
    cleaned_text = re.sub(r'[^\w\s.,?!-]', '', cleaned_text)
    # Trim leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text





    




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


