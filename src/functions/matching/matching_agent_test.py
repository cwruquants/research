import os
import sys
from src.functions.matching.matching_agent import MatchingAgent
from src.abstract_classes.attribute import DocumentAttr
from src.functions.decompose_transcript import extract_presentation_section, extract_qa_section, clean_spoken_content
from src.functions.matching.exposure_results import ExposureResults, MatchInstance

# --- Helpers ---
def load_sample_document(xml_path):
    try:
        presentation_text = extract_presentation_section(xml_path)
        qa_text = extract_qa_section(xml_path)
        full_text = presentation_text + "\n\n" + qa_text
        cleaned_text = clean_spoken_content(full_text)
        return DocumentAttr(document=cleaned_text)
    except Exception as e:
        print(f"Error loading document: {e}")
        return DocumentAttr(document="")

def load_text_document(path):
    try:
        with open(path, encoding="utf-8") as f:
            return DocumentAttr(document=f.read())
    except Exception as e:
        print(f"Error loading text document: {e}")
        return DocumentAttr(document="")

# --- Test Data Paths ---
KEYWORDS_FILE = "src/functions/matching/test_keywords.csv"
SAMPLE_XML = "data/earnings_calls/ex1.xml"
LARGE_DOC = "src/functions/matching/large_document.txt"
OVERLAP_DOC = "src/functions/matching/overlap_keywords_document.txt"

# --- Test Functions ---
def test_agent_init():
    print("Test: Agent Initialization...")
    try:
        agent = MatchingAgent(keywords_file=KEYWORDS_FILE, document=load_sample_document(SAMPLE_XML))
        assert agent.keywords_list, "Keywords should be loaded."
        assert agent.document.text, "Document should be loaded."
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

def test_keyword_loading():
    print("Test: Keyword Loading...")
    try:
        agent = MatchingAgent()
        agent.load_keywords(KEYWORDS_FILE)
        assert len(agent.keywords_list) > 0
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

def test_direct_match():
    print("Test: Direct Match...")
    try:
        agent = MatchingAgent(keywords_file=KEYWORDS_FILE, document=load_sample_document(SAMPLE_XML))
        results = agent.direct_match()
        print(f"  Total direct matches: {results.total_direct_matches}")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

def test_cos_similarity():
    print("Test: Cosine Similarity (word mode)...")
    try:
        agent = MatchingAgent(keywords_file=KEYWORDS_FILE, document=load_sample_document(SAMPLE_XML))
        results = agent.cos_similarity(match_type="word", threshold=0.7)
        print(f"  Total cosine matches: {results.total_cosine_matches}")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

def test_export_results():
    print("Test: Export Results...")
    try:
        agent = MatchingAgent(keywords_file=KEYWORDS_FILE, document=load_sample_document(SAMPLE_XML))
        results = agent.direct_match()
        d = results.export_to_dict()
        s = str(results)
        print(f"  Export dict keys: {list(d.keys())}")
        print(f"  String output preview: {s[:100]}...")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

def test_large_document():
    print("Test: Large Document Direct Match...")
    try:
        if not os.path.exists(LARGE_DOC):
            print("  SKIPPED: large_document.txt not found.")
            return
        agent = MatchingAgent(keywords_file=KEYWORDS_FILE, document=load_text_document(LARGE_DOC))
        results = agent.direct_match()
        print(f"  Total direct matches: {results.total_direct_matches}")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

def test_overlap_keywords():
    print("Test: Overlap Keywords Document...")
    try:
        if not os.path.exists(OVERLAP_DOC):
            print("  SKIPPED: overlap_keywords_document.txt not found.")
            return
        agent = MatchingAgent(keywords_file=KEYWORDS_FILE, document=load_text_document(OVERLAP_DOC))
        results = agent.direct_match()
        print(f"  Total direct matches: {results.total_direct_matches}")
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

def main():
    print("==== MatchingAgent Scripted Tests ====")
    test_agent_init()
    test_keyword_loading()
    test_direct_match()
    test_cos_similarity()
    test_export_results()
    test_large_document()
    test_overlap_keywords()
    print("==== All tests complete ====")

if __name__ == "__main__":
    main()
