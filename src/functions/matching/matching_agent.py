from src.abstract_classes.attribute import DocumentAttr
import os, csv
from word_forms.word_forms import get_word_forms
from sentence_transformers import SentenceTransformer, util
import torch


class MatchingAgent:
    def __init__(self, keywords_file=None, document: DocumentAttr = None, cos_threshold: float = 0.7):
        """
        Initialize a MatchingAgent that analyzes document exposure based on keywords.

        Args:
            document (DocumentAttr): Document containing the keywords to match against
            keywords_file (str, optional): Path to CSV file containing exposure words
            cos_threshold (float): Similarity threshold for matching (default: 0.7)
        """
        self.document = document
        self.keywords_list = []
        self.cos_threshold = cos_threshold  # Default threshold for cosine similarity

        if keywords_file:
            self.load_keywords(keywords_file)
            print("LOADED KEYWORDS")
            self.load_keyword_variations(self.keywords_list)
            print("LIST", self.keywords_list)

    def load_keywords(self, keywords_file: str) -> None:
        """
        Load and process exposure words from a CSV file.

        Args:
            keywords_file (str): Path to the CSV file containing exposure words
        """
        if not os.path.exists(keywords_file):
            raise FileNotFoundError(f"Keywords file not found: {keywords_file}")

        self.keywords_list = []

        try:
            with open(keywords_file, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    for cell in row:
                        keyword = cell.strip()
                        if keyword:  # Only add non-empty keywords
                            self.keywords_list.append(keyword)
        except Exception as e:
            raise RuntimeError(f"Error reading keywords file: {e}")

        # Remove duplicates
        seen = set()
        self.keywords_list = [keyword for keyword in self.keywords_list
                              if keyword.lower() not in seen and not seen.add(keyword.lower())]

        # print(f"Loaded {len(self.keywords_list)} unique keywords from {keywords_file}")

    def load_keyword_variations(self, keywords: list) -> None:
        """
        Loads keyword variations for all keywords in a list.

        Args:
            keywords (list): List of keywords to process

        Returns:
            list: An updated list of keywords with variations.
        """
        all_variations = []
        for word in self.keywords_list:
            print("Processing word:", word)
            forms = get_word_forms(word)
            all_variations.extend(list(set().union(*forms.values())))
        self.keywords_list += all_variations

    def cos_similarity(self, match_type: str = "", threshold: float = None):
        """
        Calculate cosine similarity between documents and find matching instances.

        Args
            threshold (float, optional): Override default similarity threshold

        Returns:
            ExposureResults: Object containing match statistics and instances
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")
        document_text = self.document.text

        if match_type == "single":
            document_words = [word.strip('.,!?:;()[]{}"\'').lower()
                              for word in document_text.split()
                              if word.strip('.,!?:;()[]{}"\'')]
            # print("Document words:", document_words)
            corpus_embeddings = model.encode(document_words, convert_to_tensor=True)
            for keyword in self.keywords_list:
                query_embedding = model.encode(keyword, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
                # Format the hits to show the actual words
                formatted_hits = [
                    {
                        'word': document_words[hit['corpus_id']],
                        'score': round(hit['score'], 3)
                    } for hit in hits[0]
                ]
                print(f"Matches for keyword '{keyword}': {formatted_hits}")
    pass


def direct_match(self):
    """
    Find exact matches between keywords and document text.

    Returns:
        Number of matches and optionally match details
    """
    # TODO:
    # 1. Extract keywords from document_text
    # 2. Find exact matches in target document
    # 3. Count matches
    # 4. If return_instances, collect match context
    # 5. Return results
    pass
