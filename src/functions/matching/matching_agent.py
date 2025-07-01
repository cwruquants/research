from src.abstract_classes.attribute import DocumentAttr
from src.functions.matching.exposure_results import ExposureResults, MatchInstance, KeywordMatches
import os, csv, re
from word_forms.word_forms import get_word_forms
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Union

class MatchingAgent:
    def __init__(self, keywords_file: str | None = None, document: DocumentAttr | None = None, cos_threshold: float = 0.7):
        """
        Initialize a MatchingAgent that analyzes document exposure based on keywords.

        Args:
            document (DocumentAttr): Document containing the text to match against
            keywords_file (str, optional): Path to CSV file containing exposure words
            cos_threshold (float): Similarity threshold for matching (default: 0.5)
        """
        self.document = document
        self.keywords_list = []
        self.cos_threshold = cos_threshold
        self.model = None

        if keywords_file:
            self.load_keywords(keywords_file)
            self.load_keyword_variations(self.keywords_list)

    def _get_model(self):
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

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

    def load_keyword_variations(self, keywords: list) -> None:
        """
        Loads keyword variations for all keywords in a list.

        Args:
            keywords (list): List of keywords to process
        """
        all_variations = []
        for word in self.keywords_list:
            print(f"Processing word: {word}")
            try:
                forms = get_word_forms(word)
                variations = list(set().union(*forms.values()))
                # Filter out the original word to avoid duplicates
                variations = [v for v in variations if v.lower() != word.lower()]
                all_variations.extend(variations)
            except Exception as e:
                print(f"Warning: Could not get variations for '{word}': {e}")
        
        # Add variations and remove duplicates
        self.keywords_list.extend(all_variations)
        seen = set()
        self.keywords_list = [keyword for keyword in self.keywords_list
                              if keyword.lower() not in seen and not seen.add(keyword.lower())]

    def _get_context(self, position: int, context_size: int = 50) -> str:
        """
        Extract context around a match position.

        Args:
            position (int): Position of the match
            context_size (int): Number of characters to include on each side

        Returns:
            str: Context string with the match highlighted
        """
        if not self.document or not self.document.text:
            raise ValueError("No document provided for matching")

        text = self.document.text
        start = max(0, position - context_size)
        end = min(len(text), position + context_size)
        return text[start:end].strip()

    def direct_match(self) -> ExposureResults:
        """
        Find exact matches between keywords and document text.

        Returns:
            ExposureResults: Object containing direct match results
        """
        if not self.document or not self.document.text:
            raise ValueError("No document provided for matching")

        document_text = self.document.text
        results = ExposureResults(
            keyword_doc=self.keywords_list,
            earnings_call=self.document,
            cosine_threshold=None
        )

        # Process each keyword
        for keyword in self.keywords_list:
            direct_matches = []
            
            # Find all occurrences of the keyword (case-insensitive)
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            for match in pattern.finditer(document_text):
                context = self._get_context(match.start())
                match_instance = MatchInstance(
                    matched_text=match.group(),
                    context=context,
                    similarity_score=1.0,
                    position=match.start()
                )
                direct_matches.append(match_instance)

            results.add_direct_matches(keyword, direct_matches)

        return results

    def cos_similarity(self, match_type: str = "word", threshold: float | None = None) -> ExposureResults:
        """
        Calculate cosine similarity between keywords and document text.

        Args:
            match_type (str): Type of matching ("word" or "bigram")
            threshold (float, optional): Override default similarity threshold

        Returns:
            ExposureResults: Object containing cosine similarity match results
        """
        if not self.document or not self.document.text:
            raise ValueError("No document provided for matching")

        model = self._get_model()
        document_text = self.document.text
        
        if threshold is None:
            threshold = self.cos_threshold

        results = ExposureResults(
            keyword_doc=self.keywords_list,
            earnings_call=self.document,
            cosine_threshold=threshold
        )

        if match_type == "word":
            # Split document into individual words, potentially use a library for this
            words_with_positions = []
            word_pattern = re.compile(r'\b\w+\b')
            
            for match in word_pattern.finditer(document_text):
                word = match.group().lower()
                position = match.start()
                words_with_positions.append((word, position, match.group()))  # (cleaned, position, original)

            if not words_with_positions:
                return results

            # Extract just the words for encoding
            document_words = [item[0] for item in words_with_positions]
            corpus_embeddings = model.encode(document_words, convert_to_tensor=True)

            # Process each keyword
            for keyword in self.keywords_list:
                cosine_matches = []
                
                try:
                    query_embedding = model.encode(keyword.lower(), convert_to_tensor=True)
                    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(document_words))
                    
                    # Filter hits by threshold and create match instances
                    for hit in hits[0]:  # hits[0] contains the list for this keyword
                        if hit['score'] >= threshold:
                            word_idx = hit['corpus_id']
                            _, position, original_word = words_with_positions[int(word_idx)]
                            context = self._get_context(position)
                            
                            match_instance = MatchInstance(
                                matched_text=original_word,
                                context=context,
                                similarity_score=float(hit['score']),
                                position=position
                            )
                            cosine_matches.append(match_instance)
                
                except Exception as e:
                    print(f"Warning: Error processing keyword '{keyword}': {e}")
                    cosine_matches = []

                results.add_cosine_matches(keyword, cosine_matches)

        else:
            raise ValueError(f"Unsupported match_type: {match_type}. Use 'word'.")

        return results