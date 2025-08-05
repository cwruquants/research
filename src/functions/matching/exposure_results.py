import json
import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MatchInstance:
    """Represents a single match occurrence"""
    keyword: str # The keyword that is being searched for
    matched_text: str # The word that was found
    context: str # Sentence where match was found
    position: int | None = None # Word index in the text
    similarity_score: float | None = None # Cosine similarity score (0-1)


@dataclass
class KeywordMatches:
    """All matches for a specific keyword"""
    keyword: str
    direct_matches: List[MatchInstance]
    cosine_matches: List[MatchInstance]

    @property
    def total_matches(self) -> int:
        return len(self.direct_matches) + len(self.cosine_matches)


class ExposureResults:
    def __init__(self, keyword_doc, earnings_call, keyword_doc_name: str | None = None, earnings_call_name: str | None = None,
                 total_sentences_in_document: int | None = None, keyword_matches: Dict[str, KeywordMatches] | None = None,
                 cosine_threshold: float | None = None, runtime_seconds: float | None = None):
        self.keyword_doc = keyword_doc
        self.earnings_call = earnings_call
        self.keyword_doc_name = keyword_doc_name
        self.earnings_call_name = earnings_call_name
        self.total_sentences_in_document = total_sentences_in_document
        self.cosine_threshold = cosine_threshold
        self.runtime_seconds = runtime_seconds
        # Store the total number of keywords searched
        self.total_keywords_searched = len(keyword_doc) if keyword_doc else 0

        if keyword_matches is None:
            self.keyword_matches: Dict[str, KeywordMatches] = {}
        else:
            self.keyword_matches = keyword_matches

    @classmethod
    def load_json(cls, file_path: str):
        """
        Loads an ExposureResults object from a JSON file created by the export method.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            ExposureResults: An instance of the ExposureResults class.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
            raise
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file at {file_path}.")
            raise

        try:
            metadata = data['metadata']
            keyword_doc_name = metadata.get('keyword_doc_name')
            earnings_call_name = metadata.get('earnings_call_name')
            cosine_threshold = metadata.get('cosine_threshold')
            runtime_seconds = metadata.get('runtime_seconds')
            total_sentences_in_document = metadata.get('total_sentences_in_document')

            # Assumes 'matches' key exists for word-format JSON
            keyword_matches_data = data.get('matches', {})
            keyword_matches_obj = {}

            for keyword, matches in keyword_matches_data.items():
                direct_matches = [
                    MatchInstance(
                        keyword=keyword,
                        matched_text=m['matched_text'],
                        context=m['context'],
                        position=m.get('position')
                    ) for m in matches.get('direct_matches', [])
                ]
                cosine_matches = [
                    MatchInstance(
                        keyword=keyword,
                        matched_text=m['matched_text'],
                        context=m['context'],
                        position=m.get('position'),
                        similarity_score=m.get('similarity_score')
                    ) for m in matches.get('cosine_matches', [])
                ]
                keyword_matches_obj[keyword] = KeywordMatches(
                    keyword=keyword,
                    direct_matches=direct_matches,
                    cosine_matches=cosine_matches
                )

            instance = cls(
                keyword_doc=None,
                earnings_call=None,
                keyword_doc_name=keyword_doc_name,
                earnings_call_name=earnings_call_name,
                total_sentences_in_document=total_sentences_in_document,
                keyword_matches=keyword_matches_obj,
                cosine_threshold=cosine_threshold,
                runtime_seconds=runtime_seconds
            )

            instance.total_keywords_searched = metadata.get('total_keywords_searched', len(keyword_matches_obj))

            return instance
        except KeyError as e:
            print(f"Error: The JSON file is missing a required key: {e}")
            raise

    def add_keyword_matches(self, keyword: str, direct_match_list: List[MatchInstance],
                            cosine_match_list: List[MatchInstance]):
        """Add matches for a specific keyword"""
        self.keyword_matches[keyword] = KeywordMatches(
            keyword=keyword,
            direct_matches=direct_match_list,
            cosine_matches=cosine_match_list
        )
    
    def add_direct_matches(self, keyword: str, match_list: List[MatchInstance]):
        """Add a direct match for a specific keyword"""
        if keyword not in self.keyword_matches:
            self.keyword_matches[keyword] = KeywordMatches(keyword=keyword, direct_matches=[], cosine_matches=[])
        self.keyword_matches[keyword].direct_matches.extend(match_list)
    
    def add_cosine_matches(self, keyword: str, match_list: List[MatchInstance]):
        """Add a cosine match for a specific keyword"""
        if keyword not in self.keyword_matches:
            self.keyword_matches[keyword] = KeywordMatches(keyword=keyword, direct_matches=[], cosine_matches=[])
        self.keyword_matches[keyword].cosine_matches.extend(match_list)

    @property
    def total_direct_matches(self) -> int:
        return sum(len(km.direct_matches) for km in self.keyword_matches.values())

    @property
    def total_cosine_matches(self) -> int:
        return sum(len(km.cosine_matches) for km in self.keyword_matches.values())

    @property
    def total_keywords_with_matches(self) -> int:
        return sum(1 for km in self.keyword_matches.values() if km.total_matches > 0)

    @property
    def total_sentences_with_match(self) -> int:
        """Calculates the number of unique sentences that contain a match."""
        unique_sentences = set()
        for km in self.keyword_matches.values():
            for match in km.direct_matches + km.cosine_matches:
                if match.context:
                    unique_sentences.add(match.context)
        return len(unique_sentences)

    @property
    def percent_sentences_with_match(self) -> float:
        """Calculates the percentage of sentences that contain a match."""
        if not self.total_sentences_in_document or self.total_sentences_in_document == 0:
            return 0.0
        return (self.total_sentences_with_match / self.total_sentences_in_document) * 100

    def get_matches_by_keyword(self, keyword: str) -> KeywordMatches | None:
        return self.keyword_matches.get(keyword)

    def get_unique_matches(self) -> List[str]:
        """
        Get a list of unique words that were matched.
        """
        unique_matches = set()
        for km in self.keyword_matches.values():
            for match in km.direct_matches:
                unique_matches.add(match.matched_text.lower())
            for match in km.cosine_matches:
                unique_matches.add(match.matched_text.lower())
        
        return list(unique_matches)

    def __str__(self) -> str:
        lines = []
        lines.append("=" * 25 + " Exposure Analysis Results " + "=" * 25)
        if self.keyword_doc_name:
            lines.append(f"Keyword Document: {self.keyword_doc_name}")
        if self.earnings_call_name:
            lines.append(f"Earnings Call: {self.earnings_call_name}")
        if self.cosine_threshold is not None:
            lines.append(f"Cosine Similarity Threshold: {self.cosine_threshold}")
        if self.runtime_seconds is not None:
            lines.append(f"Runtime: {self.runtime_seconds:.2f} seconds")

        lines.append("\n" + "-" * 20 + " Summary " + "-" * 20)
        lines.append(f"Total keywords searched: {self.total_keywords_searched}")
        lines.append(f"Total keywords with matches: {self.total_keywords_with_matches}")
        lines.append(f"Total direct matches: {self.total_direct_matches}")
        lines.append(f"Total cosine matches: {self.total_cosine_matches}")
        if self.total_sentences_in_document:
             lines.append(f"Sentences with matches: {self.total_sentences_with_match} / {self.total_sentences_in_document} ({self.percent_sentences_with_match:.2f}%)")
        lines.append(f"Total unique matches: {len(self.get_unique_matches())}")
        lines.append(f"Unique matches: {self.get_unique_matches()}") 

        if not self.keyword_matches:
            lines.append("\nNo matches found.")
            return "\n".join(lines)

        lines.append("\n" + "=" * 20 + " Matches by Keyword " + "=" * 20)
        for keyword, km in self.keyword_matches.items():
            lines.append(f"\nKeyword: '{keyword}' ({km.total_matches} total matches)")
            
            if km.direct_matches:
                lines.append(f"  Direct Matches ({len(km.direct_matches)}):")
                for match in km.direct_matches:
                    match_info = f"    - Text: '{match.matched_text}', Context: '{match.context}'"
                    if match.position is not None:
                        match_info += f", Position: {match.position}"
                    lines.append(match_info)
            
            if km.cosine_matches:
                lines.append(f"  Cosine Similarity Matches ({len(km.cosine_matches)}):")
                for match in km.cosine_matches:
                    match_info = f"    - Text: '{match.matched_text}', Context: '{match.context}'"
                    if match.similarity_score is not None:
                        match_info += f", Score: {match.similarity_score:.4f}"
                    if match.position is not None:
                        match_info += f", Position: {match.position}"
                    lines.append(match_info)
        
        return "\n".join(lines)

    def export_to_dict(self, format: str = "word") -> Dict:
        """
        Convert to dictionary format for serialization.
        """
        metadata = {
            'keyword_doc_name': self.keyword_doc_name,
            'earnings_call_name': self.earnings_call_name,
            'cosine_threshold': self.cosine_threshold,
            'runtime_seconds': self.runtime_seconds,
            'total_keywords_searched': self.total_keywords_searched,
            'total_keywords_with_matches': self.total_keywords_with_matches,
            'total_direct_matches': self.total_direct_matches,
            'total_cosine_matches': self.total_cosine_matches,
            'total_sentences_in_document': self.total_sentences_in_document,
            'total_sentences_with_match': self.total_sentences_with_match,
            'percent_sentences_with_match': self.percent_sentences_with_match
        }

        if format == "word":
            return {
                'metadata': metadata,
                'matches': {
                    keyword: {
                        'direct_matches': [
                            {
                                'matched_text': m.matched_text,
                                'context': m.context,
                                'position': m.position
                            } for m in km.direct_matches
                        ],
                        'cosine_matches': [
                            {
                                'matched_text': m.matched_text,
                                'context': m.context,
                                'similarity_score': m.similarity_score,
                                'position': m.position
                            } for m in km.cosine_matches
                        ]
                    } for keyword, km in self.keyword_matches.items()
                }
            }
        
        elif format == "sentence":
            matches_by_sentence = {}
            for km in self.keyword_matches.values():
                # Process direct matches
                for match in km.direct_matches:
                    if not match.context: continue
                    sentence_entry = matches_by_sentence.setdefault(match.context, {'matches': []})
                    sentence_entry['matches'].append({
                        'keyword': match.keyword,
                        'matched_text': match.matched_text,
                        'type': 'direct',
                        'position': match.position,
                        'score': 1.0
                    })
                
                # Process cosine matches
                for match in km.cosine_matches:
                    if not match.context: continue
                    sentence_entry = matches_by_sentence.setdefault(match.context, {'matches': []})
                    sentence_entry['matches'].append({
                        'keyword': match.keyword,
                        'matched_text': match.matched_text,
                        'type': 'cosine',
                        'position': match.position,
                        'score': match.similarity_score
                    })
            
            return {
                'metadata': metadata,
                'matches_by_sentence': matches_by_sentence
            }

        else:
            raise ValueError("Invalid format specified. Choose 'word' or 'sentence'.")
            

    def export_to_json(self, output_dir: str = "results", export_format: str = "word"):
        """
        Export the result dictionary to a JSON file in the specified directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"exposure_results_{timestamp}.json"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.export_to_dict(format=export_format), f, indent=4)
            print(f"File saved to: ", file_path)