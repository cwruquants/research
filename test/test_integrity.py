from pathlib import Path
from src.analysis.analyst_module import Analyst
from src.document.abstract_classes.setup_module import SentimentSetup


def main():
    # Set up paths
    earnings_calls_dir = Path("data/earnings_calls/2016")
    results_dir = Path("results/batch_20251030_031511-20251030T201729Z-1-001")
    
    # Initialize analyst with a SentimentSetup so sentiment can run
    setup = SentimentSetup(
    sheet_name_positive='ML_positive_unigram',
    sheet_name_negative='ML_negative_unigram',
    ml_wordlist_path="data/word_sets/Garcia_MLWords.xlsx",
    hf_model='cardiffnlp/twitter-roberta-base-sentiment-latest',
    device=0,
)
    analyst = Analyst(setups=setup)
    
    # 1. Run check_integrity
    print("=" * 60)
    print("STEP 1: CHECKING INTEGRITY")
    print("=" * 60)
    integrity_before = analyst.check_integrity(
        earnings_calls_dir=str(earnings_calls_dir),
        results_dir=str(results_dir)
    )
    print(f"Present: {integrity_before['present']}")
    print(f"Missing: {integrity_before['missing']}")
    print(f"Total transcripts: {integrity_before.get('total_transcripts', 'n/a')}")
    print(f"Total metadata files: {integrity_before['total_metadata_files']}")
    print()
    
    # 2. Run repair_directory with sentiment analysis
    print("=" * 60)
    print("STEP 2: RUNNING REPAIR WITH SENTIMENT ANALYSIS")
    print("=" * 60)
    repair_result = analyst.repair_directory(
        earnings_calls_dir=str(earnings_calls_dir),
        results_dir=str(results_dir),
        integrity_snapshot=integrity_before,
        run_sentiment=True,
        matching_method=None,
    )
    print(f"Files processed: {repair_result['processed']}")
    print(f"Errors encountered: {len(repair_result['errors'])}")
    print()
    
    # 3. Run check_integrity again
    print("=" * 60)
    print("STEP 3: CHECKING INTEGRITY AGAIN")
    print("=" * 60)
    integrity_after = analyst.check_integrity(
        earnings_calls_dir=str(earnings_calls_dir),
        results_dir=str(results_dir)
    )
    print(f"Present: {integrity_after['present']}")
    print(f"Missing: {integrity_after['missing']}")
    print(f"Total transcripts: {integrity_after.get('total_transcripts', 'n/a')}")
    print(f"Total metadata files: {integrity_after['total_metadata_files']}")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Missing before: {integrity_before['missing']}")
    print(f"Missing after: {integrity_after['missing']}")
    print(f"Fixed: {integrity_before['missing'] - integrity_after['missing']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
