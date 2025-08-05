import json
from collections import defaultdict
from typing import Dict, Any, Set
import os
from datetime import datetime

def _extract_sentence_map(d: Dict[str, Any]) -> Dict[str, Dict[str, Set[str]]]:
    """
    sentence → {"keywords": set(...), "matched_texts": set(...)}
    """
    out = {}
    for sent, payload in d.get("matches_by_sentence", {}).items():
        kws, mts = set(), set()
        for m in payload.get("matches", []):
            if "keyword" in m:
                kws.add(str(m["keyword"]))
            if "matched_text" in m:
                mts.add(str(m["matched_text"]))
        out[sent] = {"keywords": kws, "matched_texts": mts}
    return out

def _percent(numer: int, denom: int) -> float:
    return round(100.0 * numer / denom, 4) if denom else 0.0

def compare(compare1: str, compare2: str, *, export: bool = False, output_dir= "", normalize_sentence=None) -> Dict[str, Any]:
    """
    Compare two JSON files produced by the ExposureResults class.

    Returns a dictionary:
    {
      "metadata": {...},
      "by_sentence": {
         "<sentence>": {
             "file1": {"keywords": [...], "matched_texts": [...]},
             "file2": {"keywords": [...], "matched_texts": [...]}
         },
         ...
      }
    }
    """
    try:
        with open(compare1, "r", encoding="utf-8") as f:
            file1 = json.load(f)
        with open(compare2, "r", encoding="utf-8") as f:
            file2 = json.load(f)
    except Exception as e:
        raise ValueError(f"Could not load one of the files: {e}")

    # structural checks 
    if file1.keys() != file2.keys():
        raise ValueError(
            "The two JSONs do not share the same top-level structure "
            "(e.g., one may be sentence-level, the other word-level)."
        )
    md1, md2 = file1.get("metadata", {}), file2.get("metadata", {})
    if md1.get("keyword_doc_name") == md2.get("keyword_doc_name"):
        raise ValueError("Both files were generated with the same keyword document; comparison is meaningless.")
    if md1.get("earnings_call_name") != md2.get("earnings_call_name"):
        raise ValueError("The earnings-call names differ; both files must refer to the same call.")

    # build
    def _norm(s: str) -> str:
        if normalize_sentence:
            return normalize_sentence(s)
        return s

    m1 = {_norm(k): v for k, v in _extract_sentence_map(file1).items()}
    m2 = {_norm(k): v for k, v in _extract_sentence_map(file2).items()}

    sents1, sents2 = set(m1.keys()), set(m2.keys())
    common, union = sents1 & sents2, sents1 | sents2

    by_sentence = {}
    for sent in sorted(common):
        by_sentence[sent] = {
            "file1": {
                "keywords": sorted(m1[sent]["keywords"]),
                "matched_texts": sorted(m1[sent]["matched_texts"]),
            },
            "file2": {
                "keywords": sorted(m2[sent]["keywords"]),
                "matched_texts": sorted(m2[sent]["matched_texts"]),
            },
        }

    meta = {
        "file1_name": compare1,
        "file2_name": compare2,
        "sentences_with_matches_file1": len(sents1),
        "sentences_with_matches_file2": len(sents2),
        "union_sentences_with_matches": len(union),
        "common_sentences_with_matches": len(common),
        "pct_common_over_union": _percent(len(common), len(union)),
        "pct_file1_sentences_overlapped": _percent(len(common), len(sents1)),
        "pct_file2_sentences_overlapped": _percent(len(common), len(sents2)),
    }

    result = {"metadata": meta, "by_sentence": by_sentence}

    if export:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"analysis_results_{timestamp}.json"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
            print(f"File saved to: ", file_path)

    return result