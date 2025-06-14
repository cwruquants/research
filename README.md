# Research Vertical Repository

## Models
1. Simple counting
2. Simple count + tf*idf
3. Simple counting with "risk" or something else without tf*idf
4. (3) with tf*df weighting
5. Sentiment +- 10 word parameter


## Folder Structure
- citations: past work, reference papers, resources that we're basing our work off of
- progress: where Ethan logs assigned work, work finished, etc.
- results: where our trial jsons are stored so we can see progression
- src: where all of our code is stored
- vault: where our massive data file will be stored as we build out our features

# Setup

# Earnings Call Analysis Script

This script provides a comprehensive toolkit for analyzing earnings call transcripts. It includes various functions for text extraction, sentiment analysis, readability assessment, and exposure analysis.

## Main Components

### Text Processing
- `extract_text(analyze_path)`: Extracts text from earnings call XML files
- `decompose_earnings_call(xml_path)`: Breaks down earnings calls into three sections:
  - Presentation section
  - Q&A executive responses
  - Q&A analyst questions

### Analysis Functions
- `analyze_presentation(text, exposure_csv, n)`: Analyzes the presentation section
- `analyze_qa_section(qa_list, exposure_csv, n)`: Analyzes Q&A sections
- `analyze_earnings_call(xml_path, exposure_csv, n)`: Performs full analysis of an earnings call

### Dataset Building
- `build_dataset_modular(folder_path, exposure_csv, n, output_csv)`: Processes multiple earnings calls and builds a comprehensive dataset
- `process_earnings_call(xml_path, exposure_csv, n)`: Processes a single earnings call for dataset building

### Analysis Metrics
The script calculates various metrics including:

1. **Exposure Analysis**
   - Exposure word counts
   - Risk word percentages

2. **Sentiment Analysis**
   - Twitter sentiment (positive/neutral/negative)
   - Loughran-McDonald sentiment metrics
   - Net sentiment
   - Polarity
   - Subjectivity

3. **Readability Metrics**
   - Coleman-Liau index
   - Dale-Chall score
   - Automated readability index
   - Flesch reading ease
   - Flesch-Kincaid grade level
   - Gunning fog index
   - SMOG index
   - Overall readability score

4. **Text Attributes**
   - Word count
   - Sentence count
   - Number-to-words ratio
   - Proportion of plural pronouns
   - Analyst count
   - Question count

## Usage Example

```python
# Process a single earnings call
input_xml = "path/to/earnings_call.xml"
exposure_csv = "path/to/exposure_words.csv"
result = process_earnings_call(input_xml, exposure_csv, n=10)

# Build dataset from multiple earnings calls
folder_path = "path/to/earnings_calls_folder"
output_csv = "earnings_calls_dataset.csv"
build_dataset_modular(folder_path, exposure_csv, n=10, output_csv=output_csv)
```

## Dependencies
- transformers
- numpy
- xml.etree.ElementTree
- logging
- pathlib
- json
- csv

## Output Format
The script can output results in both JSON and CSV formats, containing:
- Company information (name, ticker, date, city)
- Exposure analysis results
- Sentiment analysis results
- Readability metrics
- Text attributes

## Notes
- The script includes modular components that can be used independently
- Analysis can be customized by enabling/disabling specific metrics
- Results are saved in a structured format for further analysis


