'''
    CWRU Quants Research Vertical Exposure Project
'''

# Import statements
import xml.etree.ElementTree as ET


# Functions
def extract_text(xml_file) -> str:
    """
        This function takes in an earnings transcript as an input, and extracts the words from the transcript.

        Input: .xml file
        Output: str
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # find <body> tag and extract
        body = root.find(".//Body")
        if body is not None and body.text:
            # clean and return
            return body.text.strip()
        else:
            return "No body section found."

    except ET.ParseError as e:
        return f"Error parsing XML: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# test
# spoken_words = extract_text("src/data/example.xml")
# print(spoken_words)

def extract_exposure(exposure_words, txt_string, buffer) -> dict:
    """
    This takes in the str returned from extract_text, and extracts regions (+- buffer) where the exposure
    words exist.

    For example, if buffer = 5, then wherever we identify an exposure word, we take the substring of words beginning
    5 words before the exposure word, and 5 words after the exposure word. This would create a string with 11 words.
    We would then add this to our return dict.

    Input: 
    - exposure_words: csv string of words (e.g., "smoke, pollution, radiation")
    - txt_string: str containing the text to search in
    - buffer: int number of words before and after the exposure word to include in the context

    Output:
    dict containing extracted context strings around each occurrence of an exposure word.
    The keys are generated identifiers (e.g., "exposure_0", "exposure_1", etc.)
    """
    
    # split the exposure_words CSV string into a list
    exposure_list = [word.strip().lower() for word in exposure_words.split(',') if word.strip()]
    
    words = txt_string.split()
    
    exposure_contexts = {}
    context_count = 0
    
    for i, word in enumerate(words):
        if word.lower() in exposure_list:
            start = max(0, i - buffer)
            end = min(len(words), i + buffer + 1) 
            context = " ".join(words[start:end])
            exposure_contexts[f"exposure_{context_count}"] = context
            context_count += 1

    return exposure_contexts


def sentiment_score(text_dict):
    """
    Returns sentiment scores for each string in text_dict using RoBERTa-based
    sentiment analysis for positive/negative/neutral sentiment.
    """
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





