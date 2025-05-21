# How to Run the Earnings Call Analysis Script

This guide will help you set up the environment and run the earnings call analysis script locally.

## Prerequisites

- Anaconda or Miniconda installed on your system
- Basic knowledge of terminal/command line

## 1. Setting Up the Conda Environment

1. Open your terminal and navigate to the project directory:
```bash
cd /path/to/your/project
```

2. Create a new conda environment with Python 3.10:
```bash
conda create -n research python=3.10
```

3. Activate the environment:
```bash
conda activate research
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

## 2. Running the Script

1. Make sure you're in the project directory and the conda environment is activated:
```bash
conda activate research
cd /path/to/your/project
```

2. The script can be run in two ways:

### Option 1: Process a single earnings call
```python
from script import process_earnings_call

# Example usage
input_xml = "path/to/your/earnings_call.xml"
exposure_csv = "src/data/paper_word_sets/political_words.csv"
output_json = "output.json"

result = process_earnings_call(input_xml, exposure_csv, n=10)
with open(output_json, 'w') as f:
    json.dump(result, f, indent=4)
```

### Option 2: Process multiple earnings calls (build dataset)
```python
from script import build_dataset

# Example usage
folder_path = "path/to/your/earnings_calls_folder"  # Folder containing XML files
exposure_csv = "src/data/paper_word_sets/political_words.csv"
output_csv = "earnings_calls_dataset.csv"
n = 10  # Window size for exposure calculation
max_files = 25  # Optional: limit number of files to process

build_dataset(folder_path, exposure_csv, n, output_csv, max_files=max_files)
```

## 3. Understanding the Parameters

- `folder_path`: Directory containing your earnings call XML files
- `exposure_csv`: Path to the CSV file containing exposure words
- `n`: Window size for exposure calculation (number of words before and after each exposure word)
- `output_csv`: Name of the output CSV file where results will be saved
- `max_files`: Optional parameter to limit the number of files processed

## 4. Output Files

The script generates:
1. A CSV file (`earnings_calls_dataset.csv`) containing all analysis results
2. Log files with processing information

## 5. Troubleshooting

If you encounter any issues:

1. Make sure all required packages are installed:
```bash
pip install -r requirements.txt
```

2. Check that your XML files are in the correct format

3. Verify that the exposure words CSV file exists and is properly formatted

4. Ensure you have sufficient disk space for the output files

## 6. Notes

- The script uses the RoBERTa model for sentiment analysis, which requires significant memory
- Processing time depends on the number of files and their size
- The `max_files` parameter is useful for testing with a subset of files before processing the entire dataset
