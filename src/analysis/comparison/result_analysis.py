import json
from collections import defaultdict
from typing import Dict, Any, Set, List, Callable, Optional
import os
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

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

def find_shared_sentences(
    results_dir: str,
    keyword_names: List[str],
    *,
    export: bool = False,
    output_dir: str = "",
    normalize_sentence: Optional[Callable[[str], str]] = None
) -> Dict[str, Any]:
    """
    Find sentences shared across multiple keyword result sets for each earnings call found in the directory.

    Args:
        results_dir: Directory containing exposure result JSON files (searched recursively).
        keyword_names: List of keyword document names to look for.
        export: Whether to export the results to a JSON file.
        output_dir: Directory to save the exported JSON file.
        normalize_sentence: Optional function to normalize sentences for comparison.

    Returns:
        Dictionary mapping earnings call identifiers to their shared sentence analysis.
    """
    results_dir_path = Path(results_dir)
    if not results_dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    # Structure: call_name -> { keyword_name -> data }
    grouped_results = defaultdict(dict)
    
    # First, collect all JSON files to process
    all_json_files = []
    for root, _, files in os.walk(results_dir_path):
        for file in files:
            if file.endswith(".json"):
                all_json_files.append(os.path.join(root, file))
    
    # Walk and collect files with progress bar
    for file_path in tqdm(all_json_files, desc="Loading exposure results", unit="file", dynamic_ncols=True):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
            
        # Check if valid exposure result
        meta = data.get("metadata", {})
        kw_name = meta.get("keyword_doc_name")
        call_name = meta.get("earnings_call_name")
        
        if not kw_name or not call_name:
            continue
            
        if kw_name in keyword_names:
            grouped_results[call_name][kw_name] = data

    # Process intersections
    analysis_results = {}
    
    def _norm(s: str) -> str:
        return normalize_sentence(s) if normalize_sentence else s

    # Filter to only calls that have all required keyword sets
    valid_calls = {
        call_name: kw_map 
        for call_name, kw_map in grouped_results.items()
        if all(k in kw_map for k in keyword_names)
    }
    
    for call_name, kw_map in tqdm(valid_calls.items(), desc="Finding shared sentences", unit="call", dynamic_ncols=True):
        # Extract sentences for each keyword set
        # Map: keyword_name -> { normalized_sent -> { original_sent, keywords, matches } }
        call_data = {}
        
        for kw_name in keyword_names:
            data = kw_map[kw_name]
            sent_map = {}
            
            # Handle 'matches_by_sentence' format
            if "matches_by_sentence" in data:
                for sent, details in data["matches_by_sentence"].items():
                    norm_sent = _norm(sent)
                    # Extract keywords/matches
                    kws = set()
                    mts = set()
                    for m in details.get("matches", []):
                        if "keyword" in m: kws.add(str(m["keyword"]))
                        if "matched_text" in m: mts.add(str(m["matched_text"]))
                    
                    if norm_sent not in sent_map:
                        sent_map[norm_sent] = {
                            "original": sent,
                            "keywords": kws,
                            "matched_texts": mts
                        }
                    else:
                        sent_map[norm_sent]["keywords"].update(kws)
                        sent_map[norm_sent]["matched_texts"].update(mts)

            # Handle 'matches' (word-level) format
            elif "matches" in data:
                for kw, details in data["matches"].items():
                    # Check direct and cosine matches
                    all_matches = details.get("direct_matches", []) + details.get("cosine_matches", [])
                    for m in all_matches:
                        sent = m.get("context")
                        if not sent:
                            continue
                        
                        norm_sent = _norm(sent)
                        txt = m.get("matched_text", "")
                        
                        if norm_sent not in sent_map:
                            sent_map[norm_sent] = {
                                "original": sent,
                                "keywords": {kw},
                                "matched_texts": {txt} if txt else set()
                            }
                        else:
                            sent_map[norm_sent]["keywords"].add(kw)
                            if txt:
                                sent_map[norm_sent]["matched_texts"].add(txt)
            
            call_data[kw_name] = sent_map

        # Find intersection of normalized sentences
        sets_of_sents = [set(d.keys()) for d in call_data.values()]
        if not sets_of_sents:
            continue
            
        shared_normalized = set.intersection(*sets_of_sents)
        
        if not shared_normalized:
            analysis_results[call_name] = {
                "shared_sentences_count": 0,
                "sentences": []
            }
            continue

        # Build result for this call
        shared_details = []
        for norm_sent in sorted(shared_normalized):
            # Pick the original sentence from the first keyword set (arbitrary but consistent)
            first_kw = keyword_names[0]
            original_text = call_data[first_kw][norm_sent]["original"]
            
            entry = {
                "text": original_text,
                "hits": {}
            }
            
            for kw_name in keyword_names:
                info = call_data[kw_name][norm_sent]
                entry["hits"][kw_name] = {
                    "keywords": sorted(list(info["keywords"])),
                    "matched_texts": sorted(list(info["matched_texts"]))
                }
            
            shared_details.append(entry)

        analysis_results[call_name] = {
            "shared_sentences_count": len(shared_details),
            "sentences": shared_details
        }

    if export:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_dir:
            output_dir = results_dir
        os.makedirs(output_dir, exist_ok=True)
        filename = f"shared_sentences_{timestamp}.json"
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=4)
            print(f"Shared sentences analysis saved to: {out_path}")

    return analysis_results
