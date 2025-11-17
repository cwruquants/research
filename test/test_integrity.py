from pathlib import Path
from datetime import datetime
from src.analysis.analyst_module import Analyst
from src.document.abstract_classes.setup_module import SentimentSetup


def main():
    # Set up paths
    earnings_calls_dir = Path("data/earnings_calls/2020")
    results_dir = Path("results/2020")
    
    # Set up save path in test directory
    test_results_dir = Path("test") / "integrity_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = test_results_dir / f"integrity_snapshot_{timestamp}.json"
    
    # Initialize analyst with a SentimentSetup so sentiment can run
    setup = SentimentSetup(
    sheet_name_positive='ML_positive_unigram',
    sheet_name_negative='ML_negative_unigram',
    ml_wordlist_path="data/word_sets/Garcia_MLWords.xlsx",
    hf_model='cardiffnlp/twitter-roberta-base-sentiment-latest',
    device=0, 
)
    analyst = Analyst(setups=setup)
    
    # Run check_integrity
    print("=" * 60)
    print("CHECKING INTEGRITY")
    print("=" * 60)
    integrity_result = analyst.check_integrity(
        earnings_calls_dir=str(earnings_calls_dir),
        results_dir=str(results_dir),
        save_results=True,
        save_path=str(save_path),
    )
    print(f"Present: {integrity_result['present']}")
    print(f"Missing: {integrity_result['missing']}")
    print(f"Total transcripts: {integrity_result.get('total_transcripts', 'n/a')}")
    print(f"Total metadata files: {integrity_result['total_metadata_files']}")
    if integrity_result.get('saved_to'):
        print(f"Results saved to: {integrity_result['saved_to']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
