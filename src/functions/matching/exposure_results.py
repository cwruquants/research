import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MatchInstance:
    """Represents a single match occurrence"""
    matched_text: str
    context: str
    similarity_score: float | None = None
    position: int | None = None  # character position in document


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
    def __init__(self, keyword_doc, earnings_call, keyword_matches: Dict[str, KeywordMatches] | None = None, cosine_threshold: float | None = None):
        self.keyword_doc = keyword_doc
        self.earnings_call = earnings_call
        self.cosine_threshold = cosine_threshold

        if keyword_matches is None:
            self.keyword_matches: Dict[str, KeywordMatches] = {}
        else:
            self.keyword_matches = keyword_matches

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

    def get_matches_by_keyword(self, keyword: str) -> KeywordMatches | None:
        return self.keyword_matches.get(keyword)

    def get_unique_matches(self) -> List[str]:
        """
        Get a list of unique words that were matched.

        This method aggregates all matched text from both direct and cosine
        similarity matches, and returns a list of unique, case-insensitive words.
        If "risk" and "Risk" are matched, they are considered the same.

        Returns:
            List[str]: A list of unique matched words.
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
        if self.cosine_threshold is not None:
            lines.append(f"Cosine Similarity Threshold: {self.cosine_threshold}")

        lines.append("\n" + "-" * 20 + " Summary " + "-" * 20)
        lines.append(f"Total keywords searched: {len(self.keyword_matches)}")
        lines.append(f"Total keywords with matches: {self.total_keywords_with_matches}")
        lines.append(f"Total direct matches: {self.total_direct_matches}")
        lines.append(f"Total cosine matches: {self.total_cosine_matches}")
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

    def export_to_dict(self) -> Dict:
        """Convert to dictionary format when needed for serialization"""
        return {
            'metadata': {
                'cosine_threshold': self.cosine_threshold,
                'total_keywords_searched': len(self.keyword_matches),
                'total_keywords_with_matches': self.total_keywords_with_matches,
                'total_direct_matches': self.total_direct_matches,
                'total_cosine_matches': self.total_cosine_matches
            },
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
            

    def export(self, path: str = ""):
        """
            Calling export will export the result dictionary to a JSON file at the path of choice.
        """
        if not path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"exposure_results_{timestamp}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export_to_dict(), f, indent=4)

