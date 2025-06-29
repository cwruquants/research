from src.abstract_classes.attribute import DocumentAttr
from src.functions.matching.exposure_results import ExposureResults
import os, csv
from word_forms.word_forms import get_word_forms
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Union




class MatchingAgent:
    def __init__(self, keywords_file=None, document: DocumentAttr = None, cos_threshold: float = 0.5):
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
            self.load_keyword_variations(self.keywords_list)

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

    def cos_similarity(self, matching_type: str = "", threshold: float = None) -> ExposureResults:
        """
        Calculate cosine similarity between documents and find matching instances.

        Args
            threshold (float, optional): Override default similarity threshold

        Returns:
            ExposureResults: Object containing match statistics and instances
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")
        document_text = self.document.text

        if matching_type == "word":
            # Split document into individual words, remove punctuation, and lower case
            document_words = [word.strip('.,!?:;()[]{}"\'').lower()
                              for word in document_text.split()
                              if word.strip('.,!?:;()[]{}"\'')]
            corpus_embeddings = model.encode(document_words, convert_to_tensor=True)

            all_hits = []
            for keyword in self.keywords_list:
                query_embedding = model.encode(keyword, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
                all_hits.append(hits[0])  # hits[0] contains the list of dictionaries for this keyword

            if threshold is None:
                threshold = self.cos_threshold

            filtered_hits = self.filter_hits(all_hits, threshold)

            results = ExposureResults(
                keyword_doc=self.keywords_list,
                earnings_call=self.document, # Unsure about this
                results_cosine=filtered_hits,
                results_cosine_threshold=threshold
            )

            return results

            # print("\n" + "=" * 60)
            # print("COSINE SIMILARITY MATCHING RESULTS")
            # print("=" * 60)
            #
            # for i, (keyword, keyword_hits) in enumerate(zip(self.keywords_list, filtered_hits)):
            #     if keyword_hits:  # Only show keywords that have matches above threshold
            #         print(f"\nKeyword: '{keyword}'")
            #         print("-" * 40)
            #         for hit in keyword_hits:
            #             word = document_words[hit['corpus_id']]
            #             score = hit['score']
            #             print(f"  • '{word}' (similarity: {score:.3f})")
            #     else:
            #         print(f"\nKeyword: '{keyword}' - No matches above threshold")
            #
            # print("\n" + "=" * 60)

            # return filtered_hits

    def filter_hits(self, hits: List[List[Dict[str, Union[int, float]]]], threshold: float = None) -> List[
        List[Dict[str, Union[int, float]]]]:
        """
        Filter hits based on a similarity threshold.

        Args:
            hits (List[List[Dict[str, Union[int, float]]]]): List of hit lists from semantic search
            threshold (float, optional): Similarity threshold to filter hits

        Returns:
            List[List[Dict[str, Union[int, float]]]]: Filtered list of hits
        """

        if threshold is None:
            threshold = self.cos_threshold

        filtered_hits = []
        for hit_list in hits:
            # Filter each list of hits for a keyword
            filtered_list = [hit for hit in hit_list if hit['score'] >= threshold]
            filtered_hits.append(filtered_list)

        return filtered_hits

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