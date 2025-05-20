'''
    CWRU Quants Research Vertical Exposure Project
'''

# Import statements
import xml.etree.ElementTree as ET
import re
import os
import csv
import inflect
from sklearn.feature_extraction.text import TfidfVectorizer 
from bs4 import BeautifulSoup

###### TEXT EXTRACTION FUNCTIONS ######
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

def extract_company_info(file_path: str) -> list:
    '''
        This function takes in an earnings transcript as an input, and extracts the company infomation from the transcript

        Input: file_path for earnings call

        Output: List[company name, ticker, earnings call date, city]
    '''
    from datetime import datetime
    from dateutil import parser

    tree = ET.parse(file_path)
    root = tree.getroot()

    company_name = root.find('companyName').text
    ticker = root.find('companyTicker').text
    date = root.find('startDate')
    city = root.find('city').text

    date_str = date.text.strip()

    dt = parser.parse(date_str)
    
    date = dt.strftime("%m-%d-%Y")


    return [company_name, ticker, date, city]


def csv_to_list(filepath):
    """
    This takes in a CSV file consisting of political bigrams and returns
    a list of bigrams with the underscore removed (e.g., "democratic_party" -> "democratic party").
    """
    political_list = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # Convert the CSV rows to a list so we can manipulate them outside the 'with' block
        rows = list(reader)
    
    # Remove the first row if it's a header
    if rows:
        rows.pop(0)
    
    # Process each row, replacing underscores with spaces
    for row in rows:
        # Ensure row is not empty
        if row and len(row) > 0:
            bigram = row[0]
            # Replace the first underscore with a space if you only expect two parts
            parts = bigram.split("_")
            if len(parts) == 2:
                bigram = parts[0] + " " + parts[1]
            political_list.append(bigram.lower())
    
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

    expanded_keywords = add_plurals(keywords)

    for index, word in enumerate(words):
        if word.lower() in expanded_keywords:
            start = max(0, index - window)
            end = min(len(words), index + window + 1)
            context = " ".join(words[start:end])
            
            # If the word is already in contexts, append the new context
            if word.lower() in contexts:
                contexts[word.lower()].append(context)
            # Otherwise, create a new list with the context
            else:
                contexts[word.lower()] = [context]
    return contexts

def extract_exposure2(text_string, seed_words, buffer):
    """
    Extracts regions around seed words and their similar words using KeyBERT.
    
    Args:
        seed_words (list): List of seed words to search for in the text.
        text_string (str): The text to analyze.
        buffer (int): gives the number of words to extract left and right of the considered word
        
    Returns:
        dict: Dictionary with seed words and similar words as keys, 
              and the surrounding words as values.
    """
    from keybert import KeyBERT

    all_words = re.findall(r'\b\w+\b', text_string.lower())

    expanded_keywords = add_plurals(seed_words)

    kw_model = KeyBERT()

    # Use KeyBERT to extract related words based on the full text
    keywords = kw_model.extract_keywords(text_string, keyphrase_ngram_range=(1, 2), 
                                         stop_words='english', top_n=10, use_mmr=True, diversity=0.5)
    
    # Extract just the words from the KeyBERT results
    similar_words = set(word.lower() for word, _ in keywords)
    
    # Include both seed words and their similar words
    search_words = set(expanded_keywords) | similar_words

    results = {}

    for i, word in enumerate(all_words):
        normalized_word = word.strip()
        if normalized_word in search_words:
            start_idx, end_idx = max(0, i - buffer), min(len(all_words), i + buffer + 1)
            surrounding_words = ' '.join(all_words[start_idx:end_idx])

            if normalized_word not in results:
                results[normalized_word] = []

            results[normalized_word].append(surrounding_words)

    return results

def add_plurals(word_list):
    p = inflect.engine()
    all_words = list(word_list)
    for word in word_list:
        all_words.append(p.plural(word))
    return all_words

def calculate_risk_word_percentage(data_dict, risk_words_csv_path):
    """
    Calculate what percentage of key-value pairs in `data_dict` contain
    at least one risk word from the CSV file at `risk_words_csv_path`.

    :param data_dict: Dictionary where values are strings to be checked.
    :param risk_words_csv_path: Path to CSV file containing a single column of risk words.
    :return: List containing:
             [0]: Count of risk word appearances.
             [1]: Floating-point percentage of dictionary entries containing risk words.
    """
    risk_words = csv_to_list(risk_words_csv_path)
    count_with_risk = 0

    for key, text_value in data_dict.items():
        lower_text = text_value.lower()
        if any(risk_word in lower_text for risk_word in risk_words):
            count_with_risk += 1

    total_entries = len(data_dict)
    if total_entries == 0:
        return [0, 0.0]  # Always return a list with two elements.

    percentage = (count_with_risk / total_entries) * 100
    return [count_with_risk, percentage]


def calculate_risk_word_percentage2(data_dict, risk_words_csv_path):
    """
    Calculate what percentage of key-value pairs in `data_dict` contain
    at least one risk word from the CSV file at `risk_words_csv_path`.

    TODO:
        - ADD USAGE OF KEYBERT WITH THIS FUNCTION

    :param data_dict: Dictionary where values are strings to be checked.
    :param risk_words_csv_path: Path to CSV file containing a single column of risk words.
    :return: Floating-point percentage of dictionary entries containing risk words.
    """
    
    return 0.0




def sentiment_score(text_dict, sentiment_analyzer=None):
    """
    Returns sentiment scores for each string in text_dict using RoBERTa-based
    sentiment analysis for positive/negative/neutral sentiment.
    
    Args:
        text_dict (dict): Dictionary of text strings to analyze
        sentiment_analyzer: Optional pre-initialized sentiment analyzer pipeline
    """
    if sentiment_analyzer is None:
        from transformers import pipeline
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

    results = {}

    for key, text in text_dict.items():
        prediction = sentiment_analyzer(text)[0]
        
        label = prediction["label"].lower() 
        score = prediction["score"]     

        if label == "positive":
            numeric_score = score
        elif label == "negative":
            numeric_score = -score
        else:
            numeric_score = 0

        results[key] = {
            "text": text,
            "label": label,
            "score": score,
            "numeric_score": numeric_score
        }

    return results

def tf_idf(*args):
    '''
        Input: 
        Output:
    '''
    documents = list(args)
    if not all(isinstance(doc, str) for doc in documents):
        raise ValueError("All inputs must be strings.")

    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(documents)

    print('\nIDF values:')
    for word, idf in zip(tfidf.get_feature_names_out(), tfidf.idf_):
        print(word, ':', idf)

    return result


