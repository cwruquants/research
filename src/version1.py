'''
    CWRU Quants Research Vertical Exposure Project
'''

# Import statements
import xml.etree.ElementTree as ET
import re
import os
import csv

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

def csv_to_list(filepath):
    """
        This takes in a csv file consisting of political bigrams and returns a list of political bigrams with the underscore removed
"""
    political_list = []
    with open (filepath,'r') as file:
        f = csv.reader(file)
    for row in f:
      political_list.append(row[0])
    del political_list[0]
    for i in range(len(political_list)):
        political_list[i] = political_list[i].split("_")[0] + " " + political_list[i].split("_")[1]
    return political_list

def extract_exposure(text, keywords, window=10) -> dict :
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
        dictionary
        
    """
    words = re.findall(r'\w+', text)
    contexts = {}

    for index, word in enumerate(words):
        if word.lower() in keywords:
            start = max(0, index - window)
            end = min(len(words), index + window + 1)
            context = " ".join(words[start:end])
            contexts[word] = context

    return contexts


def sentiment_score(dicionary):
    """
        Returns sentiment score using roBERTa method for positive/negative/neutral sentiment surrounding our exposure words.

        Input:
        - dictionary: dict

        Output:
        - score: double?
    """

    return 0


