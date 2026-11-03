from research.matching.cosine import CosineMatchingMixin
from research.matching.results import ExposureResults, MatchInstance
import os, csv, re
from bisect import bisect_right
from word_forms.word_forms import get_word_forms
from typing import List
from research.transcripts.tokenize import document_to_word, document_to_sentence
from research.transcripts.parse import load_document
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import time
from typing import Tuple, Dict

# Dropped from the document for normalized phrase matching. Negations stay: "impact not" is a real entry
PHRASE_STOPWORDS = frozenset(
    "a an the of to in on at for and or but is are was were be been it its this that these those "
    "we our i you they he she as by with from so very have has had will would can could do does did".split()
)

# Sentences that read as a legal safe-harbor statement, and the sentences around them
SAFE_HARBOR_RE = re.compile(
    r"forward[- ]looking|safe[- ]harbor|private securities|actual results|differ materially"
    r"|securities and exchange commission|sec filing|filings? with the sec|form 10-?[kq]"
    r"|annual report on form|risk factors|no obligation|undertakes? no",
    re.IGNORECASE,
)
SAFE_HARBOR_WINDOW = 2

def _strip_plural(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word

class MatchingAgent(CosineMatchingMixin):
    def __init__(self, keywords_path: str, cos_threshold: float = 0.7, find_keyword_variations: bool = False,
                 normalize_phrases: bool = False, exclude_phrases_path: str | None = None,
                 exclude_safe_harbor: bool = False):
        """
        Initialize a MatchingAgent that analyzes document exposure based on keywords.

        Args:
            keywords_file (str, optional): Path to CSV file containing exposure words
            cos_threshold (float): Similarity threshold for matching (default: 0.7)
            normalize_phrases (bool): In direct matching, also match multi-word keywords across
                stopwords and plural endings, for lists built from stripped text ("want make"
                matches "want to make"). Single-word keywords stay exact.
            exclude_phrases_path (str, optional): CSV of terms of art ("risk management"); in direct
                matching, words inside them don't count toward any keyword.
            exclude_safe_harbor (bool): In direct matching, skip safe-harbor sentences and the
                sentences around them.
        """
        self.normalize_phrases = normalize_phrases
        self.exclude_safe_harbor = exclude_safe_harbor
        self.excluded_phrases = set()
        if exclude_phrases_path:
            tokenizer = RegexpTokenizer(r'\w+')
            with open(exclude_phrases_path, 'r', encoding='utf-8') as file:
                self.excluded_phrases = {
                    tuple(token.lower() for token in tokenizer.tokenize(cell))
                    for row in csv.reader(file) for cell in row if cell.strip()
                }
        self.keywords_path = keywords_path
        self._load_keywords(keywords_path)
        if find_keyword_variations:
            self._load_keyword_variations(self.keywords_list)
        self._load_phrases()
        # Document is provided later to processing methods and may be overwritten per call
        self.document_path = None
        self.document = None
        self.keyword_doc_name = os.path.basename(keywords_path)
        self.document_name = None
        self.cos_threshold = cos_threshold
        self.find_keyword_variations = find_keyword_variations
        self.model = None
        self.device = None


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

    @staticmethod
    def _sentence_word_ends(sentences: List[str]) -> List[int]:
        """
        Cumulative word counts per sentence: entry i is the document word index
        just past the last word of sentence i.
        """
        tokenizer = RegexpTokenizer(r'\w+')
        ends = []
        word_count = 0
        for sentence in sentences:
            word_count += len(tokenizer.tokenize(sentence))
            ends.append(word_count)
        return ends

    def _get_context(self, word_index: int, sentences: List[str], context_size: int = 0,
                     sentence_ends: List[int] | None = None) -> str:
        """
        Extract sentence-based context around a match.
        Args:
            word_index (int): Index of the matched word in the document's word list.
            sentences (List[str]): The list of sentences in the document.
            context_size (int): Number of sentences to include on each side.
            sentence_ends (List[int], optional): Precomputed _sentence_word_ends(sentences),
                passed by callers that look up many matches in the same document.
        Returns:
            str: Context string.
        """
        if not sentences:
            return ""

        if sentence_ends is None:
            sentence_ends = self._sentence_word_ends(sentences)

        # Find which sentence the word belongs to
        sentence_index = bisect_right(sentence_ends, word_index)
        if sentence_index >= len(sentences):
            return ""

        # Get context sentences
        start = max(0, sentence_index - context_size)
        end = min(len(sentences), sentence_index + context_size + 1)
        
        context_sentences = sentences[start:end]
        
        return " ".join(context_sentences).strip()

    def _load_phrases(self) -> None:
        """Index multi-word keywords by their lowercase tokens, for _phrase_spans."""
        tokenizer = RegexpTokenizer(r'\w+')
        exact: Dict[Tuple[str, ...], str] = {}
        normalized: Dict[Tuple[str, ...], str] = {}
        for keyword in self.keywords_list:
            tokens = tuple(token.lower() for token in tokenizer.tokenize(keyword))
            if len(tokens) < 2:
                continue
            if self.normalize_phrases and not PHRASE_STOPWORDS.intersection(tokens):
                normalized.setdefault(tuple(_strip_plural(token) for token in tokens), keyword)
            else:
                exact.setdefault(tokens, keyword)
        self.exact_phrases = exact
        self.normalized_phrases = normalized

    def _phrase_spans(self, document_words: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Find multi-word keywords as consecutive document words (case-insensitive).
        Returns:
            Keyword -> list of (start, end) word index spans, end exclusive.
        """
        spans: Dict[str, List[Tuple[int, int]]] = {}

        def scan(words: List[str], positions: List[int], phrases: Dict[Tuple[str, ...], str]) -> None:
            for size in {len(phrase) for phrase in phrases}:
                for i in range(len(words) - size + 1):
                    keyword = phrases.get(tuple(words[i:i + size]))
                    if keyword is not None:
                        spans.setdefault(keyword, []).append((positions[i], positions[i + size - 1] + 1))

        lowered = [word.lower() for word in document_words]
        scan(lowered, list(range(len(lowered))), self.exact_phrases)
        if self.normalized_phrases:
            kept = [i for i, word in enumerate(lowered) if word not in PHRASE_STOPWORDS]
            scan([_strip_plural(lowered[i]) for i in kept], kept, self.normalized_phrases)
        return spans

    def _excluded_positions(self, document_words: List[str], sentences: List[str],
                            sentence_ends: List[int]) -> set:
        """Document word indexes that direct matching must skip."""
        excluded = set()
        lowered = [word.lower() for word in document_words]
        for size in {len(phrase) for phrase in self.excluded_phrases}:
            for i in range(len(lowered) - size + 1):
                if tuple(lowered[i:i + size]) in self.excluded_phrases:
                    excluded.update(range(i, i + size))

        if self.exclude_safe_harbor:
            starts = [0] + sentence_ends[:-1]
            for i, sentence in enumerate(sentences):
                if SAFE_HARBOR_RE.search(sentence):
                    for j in range(max(0, i - SAFE_HARBOR_WINDOW), min(len(sentences), i + SAFE_HARBOR_WINDOW + 1)):
                        excluded.update(range(starts[j], sentence_ends[j]))
        return excluded

    def _direct_match(self, show_progress: bool = False, document_words: List[str] | None = None,
                      sentences: List[str] | None = None) -> ExposureResults:
        """
        Find exact matches between keywords and document text.
        Args:
            document_words, sentences (optional): Precomputed document_to_word / document_to_sentence
                of self.document, so several keyword sets can share one tokenization.
        Returns:
            ExposureResults: Object containing direct match results
        """
        if not self.document or not self.document.text:
            raise ValueError("No document provided for matching")

        start_time = time.time()

        if document_words is None:
            document_words = document_to_word(self.document)
        if sentences is None:
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

        # One pass over the document: positions of every word that is a keyword (case-insensitive)
        keyword_set = {keyword.lower() for keyword in self.keywords_list}
        positions_by_keyword: Dict[str, List[int]] = {}
        for i, word in enumerate(document_words):
            lowered = word.lower()
            if lowered in keyword_set:
                positions_by_keyword.setdefault(lowered, []).append(i)

        sentence_ends = self._sentence_word_ends(sentences)
        excluded = self._excluded_positions(document_words, sentences, sentence_ends)
        phrase_spans = self._phrase_spans(document_words)

        # Process each keyword
        for keyword in self.keywords_list:
            direct_matches = [
                MatchInstance(
                    keyword=keyword,
                    matched_text=document_words[i],
                    context=self._get_context(i, sentences, sentence_ends=sentence_ends),
                    position=i
                )
                for i in positions_by_keyword.get(keyword.lower(), [])
                if i not in excluded
            ]
            direct_matches += [
                MatchInstance(
                    keyword=keyword,
                    matched_text=" ".join(document_words[start:end]),
                    context=self._get_context(start, sentences, sentence_ends=sentence_ends),
                    position=start
                )
                for start, end in phrase_spans.get(keyword, [])
                if start not in excluded
            ]

            results.add_direct_matches(keyword, direct_matches)
        
        end_time = time.time()
        results.runtime_seconds = end_time - start_time
        return results

    def match_document(self, document, document_name: str | None = None,
                       document_words: List[str] | None = None,
                       sentences: List[str] | None = None) -> ExposureResults:
        """
        Direct-match one already-loaded document (a DocumentAttr from load_document).
        Pass document_words / sentences from document_to_word / document_to_sentence to
        share one tokenization across several keyword sets.
        """
        self.document = document
        self.document_name = document_name
        return self._direct_match(document_words=document_words, sentences=sentences)

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
