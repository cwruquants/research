"""
Onboarding demo: find keywords in one earnings-call transcript.

    uv run research demo                       # synthetic transcript, no licensed data needed
    uv run research demo path/to/123_T.xml     # any transcript XML you have access to

Cosine matching runs only when the [ml] extra is installed (uv sync --extra ml).
"""
import importlib.util
import sys

from research.config import DATA_DIR, PROJECT_ROOT
from research.matching.direct import MatchingAgent

def run_onboarding_demo():
    """
    This function demonstrates how to use the MatchingAgent to find keywords in a document.
    """
    print("=" * 60)
    print("Starting Onboarding Demo")
    print("=" * 60)

    # ==========================================
    # STEP 1: Define Paths
    # ==========================================
    # We need to tell the code where to find the input files.
    # 1. Keywords file: A CSV containing words we want to search for.
    # 2. Document file: The text file (XML in this case) we want to search in.
    
    # pathlib paths work on both Windows and Mac/Linux.
    # The default transcript is synthetic (tests/fixtures), so the demo runs without licensed data.
    keywords_path = DATA_DIR / "test_sets" / "politics.csv"
    document_path = sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / "tests" / "fixtures" / "transcripts" / "spaced_T.xml"
    keywords_path, document_path = str(keywords_path), str(document_path)

    print(f"Using keywords from: {keywords_path}")
    print(f"Scanning document:   {document_path}")
    print("-" * 60)

    # ==========================================
    # STEP 2: Initialize the Matching Agent
    # ==========================================
    # The MatchingAgent is the main class responsible for finding matches.
    # We initialize it with the path to our keywords.
    # cos_threshold determines how strict the 'fuzzy' matching is (0.7 is a common default).
    
    print("Initializing MatchingAgent...")
    agent = MatchingAgent(
        keywords_path=keywords_path,
        cos_threshold=0.7 
    )
    print("Agent initialized successfully.")
    print("-" * 60)

    # ==========================================
    # STEP 3: Run Direct Matching
    # ==========================================
    # Direct matching looks for the EXACT word in the document.
    # It is fast and precise but might miss variations (like 'running' vs 'run').
    
    print("Running DIRECT matching (exact matches only)...")
    direct_results = agent.single_processing(
        document_path=document_path,
        matching_function="direct",  # Specify 'direct' for exact matching
        print_results=False,         # We will print manually below
        save_json=False              # We won't save to disk in this demo
    )

    print(f"Direct matching finished!")
    print(f"Total exact matches found: {direct_results.total_direct_matches}")
    
    # Let's inspect a few matches if any exist
    if direct_results.total_direct_matches > 0:
        print("\n--- Examples of Direct Matches ---")
        count = 0
        for keyword, matches in direct_results.keyword_matches.items():
            if matches.direct_matches:
                for match in matches.direct_matches:
                    print(f"Keyword: '{keyword}' found in text: '{match.matched_text}'")
                    # Context shows the surrounding text
                    print(f"Context: ...{match.context[:100]}...") 
                    count += 1
                    if count >= 3: break # Only show first 3
            if count >= 3: break
    else:
        print("No direct matches found for these keywords.")
    
    print("-" * 60)

    # ==========================================
    # STEP 4: Run Cosine Similarity Matching
    # ==========================================
    # Cosine matching uses AI (embeddings) to find words that have similar MEANINGS.
    # It can find 'automobile' if you search for 'car'.
    # This is slower but much more powerful.
    
    if importlib.util.find_spec("sentence_transformers") is None:
        print("Skipping cosine matching: install the [ml] extra with `uv sync --extra ml`.")
        print("=" * 60)
        return

    print("Running COSINE matching (semantic/meaning matches)...")
    print("This might take a moment to load the model...")
    
    cosine_results = agent.single_processing(
        document_path=document_path,
        matching_function="cosine",   # Switch to 'cosine'
        match_type="word",            # We are matching single words
        exclude_duplicates=True,      # Avoid counting the same match multiple times
        save_json=False
    )

    print(f"Cosine matching finished!")
    print(f"Total semantic matches found: {cosine_results.total_cosine_matches}")

    # Inspect cosine matches
    if cosine_results.total_cosine_matches > 0:
        print("\n--- Examples of Cosine Matches ---")
        count = 0
        for keyword, matches in cosine_results.keyword_matches.items():
            if matches.cosine_matches:
                for match in matches.cosine_matches:
                    print(f"Keyword: '{keyword}' matched with: '{match.matched_text}'")
                    print(f"Similarity Score: {match.similarity_score:.4f} (1.0 is exact)")
                    print(f"Context: ...{match.context[:100]}...")
                    print("")
                    count += 1
                    if count >= 3: break
            if count >= 3: break
    else:
        print("No cosine matches found.")

    print("=" * 60)
    print("Demo Completed Successfully!")
    print("You can now try changing the keywords file or the document path to explore more.")
    print("=" * 60)

if __name__ == "__main__":
    run_onboarding_demo()
