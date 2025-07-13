from src.abstract_classes.attribute import DocumentAttr
from src.functions.matching.exposure_results import ExposureResults, MatchInstance
import os, csv
from word_forms.word_forms import get_word_forms
from sentence_transformers import SentenceTransformer, util
from typing import List
from src.functions.decompose_text import document_to_word, document_to_sentence, document_to_bigram
from nltk.tokenize import RegexpTokenizer
import torch

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if keywords_file:
            self.load_keywords(keywords_file)
            # self.load_keyword_variations(self.keywords_list)

    def _get_model(self):
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.model.to(self.device)
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

    def _get_context(self, word_index: int, sentences: List[str], context_size: int = 0) -> str:
        """
        Extract sentence-based context around a match.
        Args:
            word_index (int): Index of the matched word in the document's word list.
            sentences (List[str]): The list of sentences in the document.
            context_size (int): Number of sentences to include on each side.
        Returns:
            str: Context string.
        """
        if not sentences:
            return ""

        # Find which sentence the word belongs to
        word_count = 0
        sentence_index = -1
        tokenizer = RegexpTokenizer(r'\w+')
        for i, sentence in enumerate(sentences):
            sentence_word_count = len(tokenizer.tokenize(sentence))
            if word_index < word_count + sentence_word_count:
                sentence_index = i
                break
            word_count += sentence_word_count
        
        if sentence_index == -1:
            return ""

        # Get context sentences
        start = max(0, sentence_index - context_size)
        end = min(len(sentences), sentence_index + context_size + 1)
        
        context_sentences = sentences[start:end]
        
        return " ".join(context_sentences).strip()

    def direct_match(self) -> ExposureResults:
        """
        Find exact matches between keywords and document text.
        Returns:
            ExposureResults: Object containing direct match results
        """
        if not self.document or not self.document.text:
            raise ValueError("No document provided for matching")

        document_words = document_to_word(self.document)
        sentences = document_to_sentence(self.document)
        results = ExposureResults(
            keyword_doc=self.keywords_list,
            earnings_call=self.document,
            cosine_threshold=None
        )

        # Process each keyword
        for keyword in self.keywords_list:
            direct_matches = []
            
            # Find all occurrences of the keyword (case-insensitive)
            for i, word in enumerate(document_words):
                if keyword.lower() == word.lower():
                    context = self._get_context(i, sentences)
                    match_instance = MatchInstance(
                        keyword=keyword,
                        matched_text=word,
                        context=context,
                        position=i
                    )
                    direct_matches.append(match_instance)

            results.add_direct_matches(keyword, direct_matches)
        return results

    def cos_similarity(self, match_type: str = "word", threshold: float | None = None, exclude_duplicates: bool = True) -> ExposureResults:
        """
        Calculate cosine similarity between keywords and document text.
        Args:
            match_type (str): Type of matching ("word", "bigram", or "hybrid")
            threshold (float, optional): Override default similarity threshold
            exclude_duplicates (bool): Exclude duplicate matches (matches with same position)
        Returns:
            ExposureResults: Object containing cosine similarity match results
        """
        if not self.document or not self.document.text:
            raise ValueError("No document provided for matching")

        model = self._get_model()
        
        if threshold is None:
            threshold = self.cos_threshold

        results = ExposureResults(
            keyword_doc=self.keywords_list,
            earnings_call=self.document,
            cosine_threshold=threshold
        )

        if match_type == "word":
            document_words_original = document_to_word(self.document)
            sentences = document_to_sentence(self.document)
            if not document_words_original:
                return results

            document_words = [word.lower() for word in document_words_original]
            corpus_embeddings = model.encode(document_words, convert_to_tensor=True, device=self.device)

            # Process each keyword
            for keyword in self.keywords_list:
                direct_matches = []
                cosine_matches = []
                
                try:
                    query_embedding = model.encode(keyword.lower(), convert_to_tensor=True, device=self.device)
                    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(document_words))
                    
                    # Filter hits by threshold and create match instances
                    for hit in hits[0]:  # hits[0] contains the list for this keyword
                        if hit['score'] >= threshold:
                            word_idx = int(hit['corpus_id'])
                            original_word = document_words_original[word_idx]
                            context = self._get_context(word_idx, sentences)
                            
                            if hit['score'] > 0.99:
                                match_instance = MatchInstance(
                                    keyword=keyword,
                                    matched_text=original_word,
                                    context=context,
                                    position=word_idx,
                                    similarity_score=1.0
                                )
                                direct_matches.append(match_instance)
                            else:
                                match_instance = MatchInstance(
                                    keyword=keyword,
                                    matched_text=original_word,
                                    context=context,
                                    similarity_score=float(hit['score']),
                                    position=word_idx
                                )
                                cosine_matches.append(match_instance)
                
                except Exception as e:
                    print(f"Warning: Error processing keyword '{keyword}': {str(e)}")
                    direct_matches = []
                    cosine_matches = []

                if direct_matches:
                    results.add_direct_matches(keyword, direct_matches)

                if cosine_matches:
                    results.add_cosine_matches(keyword, cosine_matches)
        elif match_type == "bigram":
            document_bigrams_original = document_to_bigram(self.document)
            sentences = document_to_sentence(self.document)
            if not document_bigrams_original:
                return results
            
            document_bigrams = [bigram.lower() for bigram in document_bigrams_original]
            corpus_embeddings = model.encode(document_bigrams, convert_to_tensor=True, device=self.device)

            # Process each keyword
            for keyword in self.keywords_list:
                direct_matches = []
                cosine_matches = []
                
                try:
                    query_embedding = model.encode(keyword.lower(), convert_to_tensor=True, device=self.device)
                    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(document_bigrams))
                    
                    # Filter hits by threshold and create match instances
                    for hit in hits[0]:
                        if hit['score'] >= threshold:
                            bigram_idx = int(hit['corpus_id'])
                            original_bigram = document_bigrams_original[bigram_idx]
                            # The position of the bigram is the position of its first word
                            context = self._get_context(bigram_idx, sentences)
                            
                            if hit['score'] > 0.99:
                                match_instance = MatchInstance(
                                    keyword=keyword,
                                    matched_text=original_bigram,
                                    context=context,
                                    position=bigram_idx,
                                    similarity_score=1.0
                                )
                                direct_matches.append(match_instance)
                            else:
                                match_instance = MatchInstance(
                                    keyword=keyword,
                                    matched_text=original_bigram,
                                    context=context,
                                    similarity_score=float(hit['score']),
                                    position=bigram_idx
                                )
                                cosine_matches.append(match_instance)
                
                except Exception as e:
                    print(f"Warning: Error processing keyword '{keyword}': {str(e)}")
                    direct_matches = []
                    cosine_matches = []

                if direct_matches:
                    results.add_direct_matches(keyword, direct_matches)

                if cosine_matches:
                    results.add_cosine_matches(keyword, cosine_matches)
        elif match_type == "hybrid":
            document_words_original = document_to_word(self.document)
            document_bigrams_original = document_to_bigram(self.document)
            sentences = document_to_sentence(self.document)
            if not document_words_original and not document_bigrams_original:
                return results
            
            num_words = len(document_words_original)

            combined_corpus_original = document_words_original + document_bigrams_original
            combined_corpus_lower = [item.lower() for item in combined_corpus_original]
            corpus_embeddings = model.encode(combined_corpus_lower, convert_to_tensor=True, device=self.device)

            # Process each keyword
            for keyword in self.keywords_list:
                direct_matches = []
                cosine_matches = []
                
                try:
                    query_embedding = model.encode(keyword.lower(), convert_to_tensor=True, device=self.device)
                    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(combined_corpus_original))
                    
                    # Filter hits by threshold and create match instances
                    for hit in hits[0]:
                        if hit['score'] >= threshold:
                            corpus_idx = int(hit['corpus_id'])
                            
                            if corpus_idx < num_words:
                                # Word match
                                word_idx = corpus_idx
                                original_text = document_words_original[word_idx]
                            else:
                                # Bigram match
                                bigram_list_idx = corpus_idx - num_words
                                word_idx = bigram_list_idx
                                original_text = document_bigrams_original[bigram_list_idx]

                            context = self._get_context(word_idx, sentences)
                            
                            if hit['score'] > 0.99:
                                match_instance = MatchInstance(
                                    keyword=keyword,
                                    matched_text=original_text,
                                    context=context,
                                    position=word_idx,
                                    similarity_score=1.0
                                )
                                direct_matches.append(match_instance)
                            else:
                                match_instance = MatchInstance(
                                    keyword=keyword,
                                    matched_text=original_text,
                                    context=context,
                                    position=word_idx,
                                    similarity_score=float(hit['score']),
                                )
                                cosine_matches.append(match_instance)
                
                except Exception as e:
                    print(f"Warning: Error processing keyword '{keyword}': {str(e)}")
                    direct_matches = []
                    cosine_matches = []

                if direct_matches:
                    results.add_direct_matches(keyword, direct_matches)

                if cosine_matches:
                    results.add_cosine_matches(keyword, cosine_matches)
        else:
            raise ValueError(f"Unsupported match_type: {match_type}. Use 'word', 'bigram', or 'hybrid'.")

        if exclude_duplicates:
            best_matches_by_position = {} # key: position, value: match_instance

            # Collect all matches and find the best one for each position
            for keyword, km in results.keyword_matches.items():
                # TODO: for cosine bigram matches, check if the setnence is already in the direct matches first.
                # Direct matches
                for match in km.direct_matches:
                    if match.position is not None:
                        if match.position not in best_matches_by_position or score > best_matches_by_position[match.position].similarity_score:
                            best_matches_by_position[match.position] = match
                
                # Cosine matches
                for match in km.cosine_matches:
                    if match.position is not None and match.similarity_score is not None:
                        score = match.similarity_score
                        if match.position not in best_matches_by_position or score > best_matches_by_position[match.position].similarity_score:
                            best_matches_by_position[match.position] = match

            filtered_results = ExposureResults(
                keyword_doc=self.keywords_list,
                earnings_call=self.document,
                cosine_threshold=threshold
            )

            for position, match in best_matches_by_position.items():
                if match.similarity_score and match.similarity_score > 0.99:
                    filtered_results.add_direct_matches(match.keyword, [match])
                else:
                    filtered_results.add_cosine_matches(match.keyword, [match])
            
            return filtered_results

        return results