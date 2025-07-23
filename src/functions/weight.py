from sklearn.feature_extraction.text import TfidfVectorizer
from src.functions.decompose_transcript import extract_presentation_section, extract_qa_section, clean_spoken_content
from pathlib import Path
# import glob
# import xml.etree.ElementTree as ET


# def extract_text_from_xml(path):
#     tree = ET.parse(path)
#     root = tree.getroot()
#     texts = []
#     for elem in root.iter():
#         if elem.text:
#             texts.append(elem.text.strip())
#     return " ".join(texts)


class TFIDFWeight:
    """
    Class to compute and store TF-IDF weights for a collection of XML files.
    """
    def __init__(self, file_path, max_features=100, ngram_range=(1, 2), min_df=1, threshold=None):
        """
        Initialize the TFIDFWeight instance.

        Args:
            directory_path (str): Path to directory containing XML files.
            max_features (int, optional): Maximum number of features for TF-IDF.
            ngram_range (tuple, optional): Range of n-values for n-grams.
            min_df (int, optional): Minimum document frequency.
            threshold (float, optional): Minimum TF-IDF weight to retain a term.
        """
        # self.directory_path = directory_path
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.threshold = threshold

        # Discover XML files
        # self.text_files = glob.glob(f"{self.directory_path}/*.xml")
        # self.titles = [Path(fp).stem for fp in self.text_files]

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            # max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df
        )
        self.weights = {}  # maps term -> list of weights matching self.titles order

    def fit(self):
        """
        Fit the TF-IDF vectorizer, store weights in a dict.
        """
        presentation_text = extract_presentation_section(self.file_path)
        qa_text = extract_qa_section(self.file_path)
        full_text = presentation_text + "\n\n" + qa_text
        cleaned_text = clean_spoken_content(full_text)
        matrix = self.vectorizer.fit_transform(cleaned_text)
        feature_names = self.vectorizer.get_feature_names_out()

        # Build term -> weights mapping
        for idx, term in enumerate(feature_names):
            row = matrix[:, idx].toarray().flatten().tolist()
            if self.threshold is None or max(row) >= self.threshold:
                self.weights[term] = row
        return self.weights

    def get_term_weight(self, term):
        """
        Return list of TF-IDF weights for the given term across documents.

        Args:
            term (str): Term or n-gram to look up.

        Returns:
            list of float: Weight values in same order as self.titles.
        """
        if not self.weights:
            self.fit()
        try:
            return self.weights[term]
        except KeyError:
            raise KeyError(f"Term '{term}' not found or below threshold.")

    def get_all_weights(self):
        """
        Return the entire term -> weights mapping.

        Returns:
            dict: Keys are terms, values are lists of weights.
        """
        if not self.weights:
            self.fit()
        return self.weights

    def get_titles(self):
        """
        Return the ordered list of document titles corresponding to weight positions.
        """
        return self.titles


if __name__ == "__main__":
    # Example usage
    directory = "/path/to/your/xml_files"
    tfidf = TFIDFWeight(directory, threshold=0.1)
    weights = tfidf.fit()
    print("Documents:", tfidf.get_titles())
    print("Weights for 'climate change':", tfidf.get_term_weight('climate change'))
    # All term weights
    all_weights = tfidf.get_all_weights()
    print(len(all_weights), "terms stored.")
