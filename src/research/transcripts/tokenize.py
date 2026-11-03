from nltk.tokenize import RegexpTokenizer
from nltk.util import bigrams

from research.transcripts._nltk import sent_tokenize
from research.transcripts.models import DocumentAttr

__all__ = [
    "document_to_word",
    "document_to_sentence",
    "document_to_bigram",
    "is_bigram",
]


def document_to_word(document: DocumentAttr):
    """
    Input:
    - document: DocumentAttr

    Output:
    - word_strings: List[String]
    """
    tokenizer = RegexpTokenizer(r"\w+")
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
    tokenizer = RegexpTokenizer(r"\w+")
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
