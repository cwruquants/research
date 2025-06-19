from attribute import SentenceAttr, ParagraphAttr
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import bigrams
import string

__all__ = [
    "par_to_sentence",
    "sentence_to_word",
    "sentence_to_bigram"
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
    string_raw = sentence.text
    tokens = word_tokenize(string_raw) # List[strings]

    tokens = [t for t in tokens if t not in string.punctuation] # Removes ['.']
    return tokens


def sentence_to_bigram(sentence: "SentenceAttr"):
    """
        Input:
        - sentence: SentenceAttr

        Output:
        - bigram_strings: List[String]

        Returns a list of bigram strings for us to initialize as BigramAttr objects
    """
    string_raw = sentence.text
    tokens = word_tokenize(string_raw) # List[strings]

    # remove punctuation
    tokens = [t for t in tokens if t not in string.punctuation] # Removes ['.']

    bigram_tuples = list(bigrams(tokens))
    bigram_strings = [" ".join(pair) for pair in bigram_tuples]

    return bigram_strings # List[string]


