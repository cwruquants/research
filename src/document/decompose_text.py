from src.document.abstract_classes.attribute import SentenceAttr, ParagraphAttr, DocumentAttr
import nltk

# Only download punkt if it's not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.util import bigrams

__all__ = [
    "par_to_sentence",
    "sentence_to_word",
    "sentence_to_bigram",
    "document_to_word",
    "document_to_sentence",
    "document_to_bigram"
]


def par_to_sentence(par: ParagraphAttr):
    """
        Input: 
        - par: ParagraphAttr

        Output:
        - sentence_strings: List[String]
    """
    sentences = sent_tokenize(par.text)
    return sentences

def sentence_to_word(sentence: SentenceAttr):
    """
        Input:
        - sentence: SentenceAttr

        Output:
        - tokens: List[String] # List of words with all of the punctuation removed
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence.text)

    return tokens


def sentence_to_bigram(sentence: SentenceAttr):
    """
        Input:
        - sentence: SentenceAttr

        Output:
        - bigram_strings: List[String]

        Returns a list of bigram strings for us to initialize as BigramAttr objects
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence.text)

    bigram_tuples = list(bigrams(tokens))
    bigram_strings = [" ".join(pair) for pair in bigram_tuples]

    return bigram_strings # List[string]

def document_to_word(document: DocumentAttr):
    """
        Input:
        - document: DocumentAttr

        Output:
        - word_strings: List[String]
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document.text)

    return tokens

def document_to_sentence(document: DocumentAttr):
    """
        Input:
        - document: DocumentAttr

        Output:
        - sentence_strings: List[String]
    """
    sentences = sent_tokenize(document.text)
    return sentences

def document_to_bigram(document: DocumentAttr):
    """
        Input:
        - document: DocumentAttr

        Output:
        - bigram_strings: List[String]

        Returns a list of bigram strings from the document text
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document.text)

    bigram_tuples = list(bigrams(tokens))
    bigram_strings = [" ".join(pair) for pair in bigram_tuples]

    return bigram_strings

def is_bigram(text: str) -> bool:
    """
    Checks if the input text is a bigram (i.e., two words separated by whitespace).

    Args:
        text (str): The input string to check.

    Returns:
        bool: True if the text is a bigram, False otherwise.
    """
    # Split by whitespace and check if there are exactly two non-empty words
    words = text.strip().split()
    return len(words) == 2 and all(word.isalnum() for word in words)
