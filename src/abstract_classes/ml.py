import pandas as pd
from typing import List
import unicodedata

def load_lexicon_from_excel(path: str, sheet_name: str) -> List[str]:
    """
    Loads a list of unigrams or bigrams from the first column (index 0) of an Excel sheet.

    - Normalizes Unicode
    - Strips whitespace
    - Lowercases everything

    Args:
        path (str): Path to the Excel file
        sheet_name (str): Name of the sheet to load

    Returns:
        List[str]: List of cleaned unigram or bigram strings
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    lexicon = df[0].dropna().tolist()

    # Clean and normalize
    lexicon = [
        unicodedata.normalize("NFKD", str(word)).strip().lower()
        for word in lexicon
    ]

    return lexicon

if __name__ == "__main__":
    # Load positive ML bigrams
    positive_bigrams = load_lexicon_from_excel(
        path="Garcia_MLWords.xlsx",
        sheet_name="ML_positive_bigram"
    )

    # Load negative ML unigrams
    negative_unigrams = load_lexicon_from_excel(
        path="Garcia_MLWords.xlsx",
        sheet_name="ML_negative_unigram"
    )

    print(positive_bigrams[:5])  # Show first few
