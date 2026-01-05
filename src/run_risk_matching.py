import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis.analyst_module import Analyst
from src.analysis.batch_runner import BatchRunner

def run_risk_matching_on_existing_batch():
    """
    Run keyword matching using the Risk dataset on an PRE-EXISTING batch of analyzed results.
    This is useful when you have already run the main analysis (sentiment, etc.) 
    and just want to add/update keyword exposure data without re-processing everything.
    """

    # 1. Path to the existing results batch you want to process
    BATCH_DIR = r"H:\My Drive\QUANTS\RESEARCH\RESULTS\2004"

    # 2. (Optional) Path to search for original XML transcripts if they moved
    # e.g., r"K:\My Drive\EarningsCalls" or None
    TRANSCRIPT_ROOT = r"H:\My Drive\QUANTS\RESEARCH\EARNINGS_CALLS\2004"

    # 3. Path to the risk keywords
    KEYWORDS_PATH = os.path.join(project_root, "data", "word_sets", "risk.csv")
    
    # 4. Matching Method ("cosine" or "direct")
    MATCHING_METHOD = "direct"

    # 5. Enable Concurrent I/O (Upload while matching)
    CONCURRENT_IO = True

    # 6. Number of threads for parallel matching
    # NUM_THREADS = 8
    # =========================================================================

    print("=" * 60)
    print("      Risk Keyword Matching (Existing Batch)")
    print("=" * 60)
    
    if not os.path.exists(KEYWORDS_PATH):
        print(f"Error: Risk keywords file not found at {KEYWORDS_PATH}")
        return

    batch_dir = Path(BATCH_DIR)
    if not batch_dir.exists():
        print(f"Error: Batch directory not found: {batch_dir}")
        return

    transcript_roots = [TRANSCRIPT_ROOT] if TRANSCRIPT_ROOT else None

    # Initialize Analyst
    print("Initializing Analyst...")
    analyst = Analyst()
    runner = BatchRunner(analyst)

    # Run Processing
    print(f"\nStarting matching process...")
    print(f"Keywords: {os.path.basename(KEYWORDS_PATH)}")
    print(f"Target:   {batch_dir}")
    if transcript_roots:
        print(f"Search in: {transcript_roots}")
    print(f"Concurrent I/O: {CONCURRENT_IO}")
    # print(f"Threads:        {NUM_THREADS}")
    print("-" * 60)

    try:
        result = runner.process_existing_batch(
            batch_dir=batch_dir,
            keyword_path=KEYWORDS_PATH,
            matching_method=MATCHING_METHOD,
            skip_on_error=True,
            transcript_roots=transcript_roots,
            search_recursive=True,
            concurrent_io=CONCURRENT_IO,
            # num_threads=NUM_THREADS
        )

        print("\n" + "=" * 60)
        print("Matching Complete!")
        print(f"Processed: {result['num_files_processed']} files")
        print(f"Summary CSV: {result['exposure_summary_csv']}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_risk_matching_on_existing_batch()
