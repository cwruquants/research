import sys
from pathlib import Path
import importlib

# Add project root to path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis.analyst_module import Analyst
from src.document.abstract_classes.setup_module import SentimentSetup
import toml

import src.analysis.analyst_module
from src.analysis.analyst_module import Analyst

print("=" * 80)
print("TEST 2: Direct matching on existing batch")
print("=" * 80)

# Specify the existing batch directory (adjust path as needed)
# The batch directory should contain subdirectories, each representing an earnings call
existing_batch_dir = str(project_root / "results" / "batch_20251030_031511-20251030T201729Z-1-001" / "batch_20251030_031511")

# Or use a variable if you ran Test 1:
# existing_batch_dir = batch_dir

# Create analyst (no setup needed for matching-only)
setup = SentimentSetup(
    sheet_name_positive='ML_positive_unigram',
    sheet_name_negative='ML_negative_unigram',
    ml_wordlist_path=str(project_root / "data" / "word_sets" / "Garcia_MLWords.xlsx"),
    hf_model='cardiffnlp/twitter-roberta-base-sentiment-latest',
    device=-1
)

analyst = Analyst(setups=[setup])

# Specify keyword file
keyword_path = str(project_root / "data" / "paper_word_sets" / "risk.csv")

# Run direct keyword matching only
result = analyst.process_existing_directory(
    batch_dir=existing_batch_dir,
    keyword_path=keyword_path,
    matching_method="direct",  # Use "direct" for exact matching only
    transcript_roots=[project_root / "data" / "earnings_calls" / "2016"],
)

print("\nMatching completed!")
print(f"Match ID: {result['match_id']}")
print(f"Files processed: {result['num_files_processed']}")
print(f"Exposure summary CSV: {result['exposure_summary_csv']}")
