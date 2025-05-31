import nltk
nltk.download('punkt')  # Download sentence tokenizer
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load FinBERT sentiment analysis pipeline
model_name = "yiyanghkust/finbert-tone"
finbert = pipeline(
    "sentiment-analysis",
    model=AutoModelForSequenceClassification.from_pretrained(model_name),
    tokenizer=AutoTokenizer.from_pretrained(model_name)
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

# Main function to get sentiment proportions
# Main function to get sentiment proportions
def get_proportions(text):
    """
    Splits a paragraph of financial text into sentences, runs FinBERT sentiment analysis on each,
    and calculates the proportion of positive, negative, and neutral sentiments.

    Args:
        text (str): A paragraph or document containing financial text.

    Returns:
        tuple: A tuple of floats (positive_proportion, negative_proportion, neutral_proportion),
               where each value is the proportion of that sentiment among all sentences.
    """
    sentences = sent_tokenize(text)
    positive, negative, neutral = 0, 0, 0

    print("Sentences:", sentences)

    for sentence in sentences:
        sentiment = get_sentiment(sentence)
        label = sentiment['label'].lower()  # ✅ Fix here
        print(sentence)
        print(label)

        if label == 'positive':
            positive += 1
        elif label == 'negative':
            negative += 1
        else:
            neutral += 1

    s = positive + negative + neutral
    if s == 0:
        return 0, 0, 0  # Avoid division by zero

    positive_proportion = positive / s
    negative_proportion = negative / s
    neutral_proportion = neutral / s

    return positive_proportion, negative_proportion, neutral_proportion