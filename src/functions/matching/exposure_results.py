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

    def get_matches_by_keyword(self, keyword: str) -> KeywordMatches | None:
        return self.keyword_matches.get(keyword)

    def export_to_dict(self) -> Dict:
        """Convert to dictionary format when needed for serialization"""
        return {
            'metadata': {
                'cosine_threshold': self.cosine_threshold,
                'total_keywords': len(self.keyword_matches),
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
            Calling export will export the result dictionary to a path of choice.
        """
        pass

