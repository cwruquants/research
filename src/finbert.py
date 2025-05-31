import nltk
nltk.download('punkt_tab')  # Download sentence tokenizer
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load FinBERT sentiment analysis pipeline
model_name = "yiyanghkust/finbert-tone"
finbert = pipeline(
    "sentiment-analysis",
    model=AutoModelForSequenceClassification.from_pretrained(model_name),
    tokenizer=AutoTokenizer.from_pretrained(model_name),
    return_all_scores=True

)

# Helper function for single-sentence sentiment
def get_sentiment(sentence):
    """
    Analyzes the sentiment of a single sentence using FinBERT.

    Args:
        sentence (str): A sentence from financial text.

    Returns:
        str: Sentiment label ('positive', 'negative', or 'neutral').
    """
    result = finbert(sentence)[0]
    return result['label']


def get_sentiment_score(sentence):
    """
    Returns the overall sentiment score for the text.Score is in range [-1, 1], where -1 is negative and 1 is positive and 0 is neutral.
    """
    #runnign the model on the text
    result = finbert(sentence)[0]  

    positive_score = 0.0
    negative_score = 0.0

    for i in result:
        label = i['label'].lower()
        score = i['score']
        if label == 'positive':
            positive_score = score
        elif label == 'negative':
            negative_score = score

    #calculating and returning sentiment score
    sentiment_score = positive_score - negative_score
    return sentiment_score