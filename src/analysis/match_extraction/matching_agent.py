# from src.abstract_classes.attribute import DocumentAttr
from src.analysis.match_extraction.exposure_results import ExposureResults, MatchInstance
import os, csv
from word_forms.word_forms import get_word_forms
from sentence_transformers import SentenceTransformer, util
from typing import List
from document.decompose_text import document_to_word, document_to_sentence, document_to_bigram, is_bigram
from document.decompose_transcript import load_document
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import torch
import time
from typing import List, Tuple, Dict, Any

class MatchingAgent:
    def __init__(self, keywords_path: str, cos_threshold: float = 0.7, find_keyword_variations: bool = False):
        """
        Initialize a MatchingAgent that analyzes document exposure based on keywords.

        Args:
            keywords_file (str, optional): Path to CSV file containing exposure words
            cos_threshold (float): Similarity threshold for matching (default: 0.7)
        """
        self.keywords_path = keywords_path
        self._load_keywords(keywords_path)
        if find_keyword_variations:
            self._load_keyword_variations(self.keywords_list)
        # Document is provided later to processing methods and may be overwritten per call
        self.document_path = None
        self.document = None
        self.keyword_doc_name = os.path.basename(keywords_path)
        self.document_name = None
        self.cos_threshold = cos_threshold
        self.find_keyword_variations = find_keyword_variations
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")


    def _get_model(self):
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.model.to(self.device)
        return self.model

    def _load_keywords(self, keywords_path: str) -> None:
        """
        Load and process exposure words from a CSV file.
        Args:
            keywords_file (str): Path to the CSV file containing exposure words
        """
        if not os.path.exists(keywords_path):
            raise FileNotFoundError(f"Keywords file not found: {keywords_path}")

        self.keywords_list = []

        try:
            with open(keywords_path, 'r', encoding='utf-8') as file:
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

    def _load_keyword_variations(self, keywords: list) -> None:
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

    def _direct_match(self, show_progress: bool = False) -> ExposureResults:
        """
        Find exact matches between keywords and document text.
        Returns:
            ExposureResults: Object containing direct match results
        """
        if not self.document or not self.document.text:
            raise ValueError("No document provided for matching")

        import time
        start_time = time.time()

        document_words = document_to_word(self.document)
        sentences = document_to_sentence(self.document)
        total_sentences = len(sentences)

        results = ExposureResults(
            keyword_doc=self.keywords_list,
            keyword_doc_name=self.keyword_doc_name,
            earnings_call=self.document,
            earnings_call_name=self.document_name,
            total_sentences_in_document=total_sentences,
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
        
        end_time = time.time()
        results.runtime_seconds = end_time - start_time
        return results

    def _prepare_corpus(self, match_type: str) -> Tuple[List[str], int]:
        """Prepares the corpus for matching based on the match type."""
        document_words = document_to_word(self.document)
        if match_type == "word":
            return document_words, len(document_words)
        
        document_bigrams = document_to_bigram(self.document)
        if match_type == "bigram":
            return document_bigrams, 0

        if match_type == "hybrid":
            return document_words + document_bigrams, len(document_words)

        raise ValueError(f"Unsupported match_type: {match_type}. Use 'word', 'bigram', or 'hybrid'.")

    def _process_hits(self, hits: List[Dict[str, Any]], keyword: str, corpus: List[str], sentences: List[str], num_words: int, threshold: float) -> Tuple[List[MatchInstance], List[MatchInstance]]:
        """Processes search hits to create direct and cosine match instances."""
        direct_matches = []
        cosine_matches = []
        
        document_words = document_to_word(self.document) if num_words > 0 else []

        for hit in hits[0]:
            if hit['score'] < threshold:
                continue

            corpus_id = int(hit['corpus_id'])
            score = float(hit['score'])
            
            if num_words > 0 and corpus_id < num_words:
                # This is a word match
                position = corpus_id
                matched_text = document_words[position]
            else:
                # This is a bigram match
                position = corpus_id - num_words
                matched_text = corpus[corpus_id]

            context = self._get_context(position, sentences)
            
            match_instance = MatchInstance(
                keyword=keyword,
                matched_text=matched_text,
                context=context,
                position=position,
                similarity_score=score
            )

            if score > 0.99:
                match_instance.similarity_score = 1.0
                direct_matches.append(match_instance)
            else:
                cosine_matches.append(match_instance)
        
        return direct_matches, cosine_matches

    def _cos_similarity(self, match_type: str = "word", threshold: float | None = None, exclude_duplicates: bool = True, show_progress: bool = False) -> ExposureResults:
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

        start_time = time.time()
        model = self._get_model()
        
        final_threshold = threshold if threshold is not None else self.cos_threshold
        
        sentences = document_to_sentence(self.document)
        total_sentences = len(sentences)

        results = ExposureResults(
            keyword_doc=self.keywords_list,
            keyword_doc_name=self.keyword_doc_name,
            earnings_call=self.document,
            earnings_call_name=self.document_name,
            total_sentences_in_document=total_sentences,
            cosine_threshold=final_threshold
        )

        corpus, num_words = self._prepare_corpus(match_type)
        
        if not corpus:
            return results

        corpus_lower = [item.lower() for item in corpus]
        corpus_embeddings = model.encode(corpus_lower, convert_to_tensor=True, device=self.device)

        for keyword in self.keywords_list:
            try:
                query_embedding = model.encode(keyword.lower(), convert_to_tensor=True, device=self.device)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=len(corpus))
                
                direct_matches, cosine_matches = self._process_hits(
                    hits, keyword, corpus, sentences, num_words, final_threshold
                )

                if direct_matches:
                    results.add_direct_matches(keyword, direct_matches)
                if cosine_matches:
                    results.add_cosine_matches(keyword, cosine_matches)

            except Exception as e:
                print(f"Warning: Error processing keyword '{keyword}': {str(e)}")

        if exclude_duplicates:
            # The logic for excluding duplicates remains the same
            word_matches = {}
            bigram_matches = {}
            
            for km in results.keyword_matches.values():
                for match in km.direct_matches + km.cosine_matches:
                    if match.position is None:
                        continue
                    
                    score = match.similarity_score if match.similarity_score is not None else 1.0
                    
                    if is_bigram(match.matched_text):
                        if match.position not in bigram_matches or score > bigram_matches[match.position].similarity_score:
                            bigram_matches[match.position] = match
                    else:
                        if match.position not in word_matches or score > word_matches[match.position].similarity_score:
                            word_matches[match.position] = match

            filtered_bigram_matches = {
                bigram_pos: bigram_match
                for bigram_pos, bigram_match in bigram_matches.items()
                if not any(abs(bigram_pos - word_pos) <= 1 for word_pos in word_matches)
            }

            final_matches = {**word_matches, **filtered_bigram_matches}

            filtered_results = ExposureResults( 
                keyword_doc=self.keywords_list,
                keyword_doc_name=self.keyword_doc_name,
                earnings_call=self.document,
                earnings_call_name=self.document_name,
                total_sentences_in_document=total_sentences,
                cosine_threshold=final_threshold
            )

            for match in final_matches.values():
                if match.similarity_score and match.similarity_score > 0.99:
                    filtered_results.add_direct_matches(match.keyword, [match])
                else:
                    filtered_results.add_cosine_matches(match.keyword, [match])
            
            filtered_results.runtime_seconds = time.time() - start_time
            return filtered_results

        results.runtime_seconds = time.time() - start_time
        return results

    def single_processing(
        self,
        document_path: str,
        matching_function: str = "cosine",
        match_type: str = "word",
        threshold: float | None = None,
        exclude_duplicates: bool = True,
        print_results: bool = False,
        save_json: bool = False,
        export_format: str = "word",
        output_dir: str = "results",
        show_progress: bool = False,
    ) -> ExposureResults:
        """
        Process a single earnings call (the agent's current document) using either
        direct matching or cosine similarity, with options to print and/or save results.

        Args:
            document_path (str): Path to the earnings call XML to process.
            matching_function (str): "cosine" for cosine similarity or "direct" for exact matching.
            match_type (str): "word", "bigram", or "hybrid" (used only for cosine matching).
            threshold (float | None): Cosine threshold override (used only for cosine matching).
            exclude_duplicates (bool): Exclude duplicates for cosine matching.
            print_results (bool): Print results to stdout.
            save_json (bool): Save JSON to the results directory.
            export_format (str): "word" or "sentence" JSON structure.
            output_dir (str): Directory to save JSON into.

        Returns:
            ExposureResults
        """
        # Load the document for this call; this overwrites any prior document state
        self.document_path = document_path
        self.document = load_document(document_path)
        self.document_name = os.path.basename(document_path)

        if matching_function == "cosine":
            results = self._cos_similarity(match_type=match_type, threshold=threshold, exclude_duplicates=exclude_duplicates, show_progress=show_progress)
        elif matching_function == "direct":
            results = self._direct_match(show_progress=show_progress)
        else:
            raise ValueError(f"Unsupported matching_function: {matching_function}. Use 'direct' or 'cosine'.")

        if print_results:
            try:
                tqdm.write(str(results))
            except Exception:
                print(results)

        if save_json:
            # Name file with document base name for easier identification
            base_name = os.path.splitext(self.document_name)[0]
            filename = f"{base_name}.json"
            results.export_to_json(output_dir=output_dir, export_format=export_format, filename=filename)

        return results

    def batch_processing(
        self,
        folder_path: str,
        matching_function: str = "cosine",
        match_type: str = "word",
        threshold: float | None = None,
        exclude_duplicates: bool = True,
        print_results: bool = False,
        save_json: bool = False,
        export_format: str = "word",
        output_root_dir: str = "results",
        show_progress: bool = False,
    ) -> dict[str, ExposureResults]:
        """
        Process a folder of earnings calls with the same agent configuration.

        Args:
            folder_path (str): Path to a folder containing earnings call XML files.
            matching_function (str): "cosine" for cosine similarity or "direct" for exact matching.
            match_type (str): "word", "bigram", or "hybrid" (used only for cosine matching).
            threshold (float | None): Cosine threshold override (used only for cosine matching).
            exclude_duplicates (bool): Exclude duplicates for cosine matching.
            print_results (bool): Print results for every document.
            save_json (bool): Save per-document JSONs into a timestamped subfolder of output_root_dir.
            export_format (str): "word" or "sentence" JSON structure.
            output_root_dir (str): Root directory under which a timestamped batch folder is created.

        Returns:
            dict[str, ExposureResults]: Mapping of document filename to results.
        """
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Provided folder_path is not a directory: {folder_path}")

        # Prepare batch output directory if saving
        batch_output_dir = None
        if save_json:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            batch_output_dir = os.path.join(output_root_dir, f"exposure_run_{timestamp}")
            os.makedirs(batch_output_dir, exist_ok=True)

        results_by_file: dict[str, ExposureResults] = {}

        # List XML files in the directory (non-recursive)
        entries = [e for e in sorted(os.listdir(folder_path)) if e.lower().endswith(".xml") and os.path.isfile(os.path.join(folder_path, e))]

        # Progress bar for files
        for entry in tqdm(entries, desc="Batch files", unit="file"):
            file_path = os.path.join(folder_path, entry)

            # Load/overwrite current document context for each file
            self.document_path = file_path
            self.document = load_document(file_path)
            self.document_name = os.path.basename(file_path)

            if matching_function == "cosine":
                # Disable per-keyword progress in batch to avoid nested bars
                res = self._cos_similarity(match_type=match_type, threshold=threshold, exclude_duplicates=exclude_duplicates, show_progress=show_progress)
            elif matching_function == "direct":
                res = self._direct_match(show_progress=show_progress)
            else:
                raise ValueError(f"Unsupported matching_function: {matching_function}. Use 'direct' or 'cosine'.")

            if print_results:
                try:
                    tqdm.write(str(res))
                except Exception:
                    print(res)

            if save_json and batch_output_dir:
                base_name = os.path.splitext(self.document_name)[0]
                filename = f"{base_name}.json"
                res.export_to_json(output_dir=batch_output_dir, export_format=export_format, filename=filename)

            results_by_file[self.document_name] = res

        return results_by_file