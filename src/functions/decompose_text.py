from src.abstract_classes.attribute import SentenceAttr, ParagraphAttr, DocumentAttr
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.util import bigrams

__all__ = [
    "par_to_sentence",
    "sentence_to_word",
    "sentence_to_bigram",
    "document_to_word"
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
    


