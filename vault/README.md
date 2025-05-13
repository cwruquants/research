## What is the vault?

Our team is building our own dataset. The vault is going to be where we house our data of all of the earnings transcripts.   

As we build out our features, we are going to aggregate all of the features into a giant csv file, where we will be able to perform various machine learning algorithms.   

## Features in the Dataset

### Basic Information
- `file_name`: Name of the XML file containing the earnings call transcript
- `company_name`: Full name of the company
- `ticker`: Company's stock ticker symbol
- `date`: Date of the earnings call
- `city`: City where the earnings call was held

### Text Analysis Metrics
For each section (full document, presentation, analyst Q&A, executive Q&A):
- `word_count`: Total number of words in the section
- `sentence_count`: Total number of sentences in the section
- `number_to_words_ratio`: Ratio of numeric tokens to total word tokens
- `proportion_plural_pronouns`: Proportion of plural pronouns among all personal pronouns
- `analyst_count`: Number of unique analysts identified in the section
- `question_count`: Count of questions (based on question marks)

### Exposure Analysis
For each section (full document, presentation, analyst Q&A, executive Q&A):
- `exposure_count`: Number of exposure words found in the section
- `risk_percentage`: Percentage of risk words found in the exposure contexts

### Twitter Sentiment Analysis
For each section (full document, presentation, analyst Q&A, executive Q&A):
- `positive_count`: Number of positive sentiment instances
- `positive_percentage`: Percentage of positive sentiment instances
- `neutral_count`: Number of neutral sentiment instances
- `neutral_percentage`: Percentage of neutral sentiment instances
- `negative_count`: Number of negative sentiment instances
- `negative_percentage`: Percentage of negative sentiment instances

### Loughran-McDonald (LM) Sentiment Analysis
For each section (full document, presentation, analyst Q&A, executive Q&A):
- `lm_positive`: Count of positive words from LM dictionary
- `lm_negative`: Count of negative words from LM dictionary
- `lm_net_sentiment`: Net sentiment score (positive - negative)
- `lm_polarity`: Sentiment polarity score
- `lm_subjectivity`: Measure of text subjectivity

### Readability Metrics
For each section (full document, presentation, analyst Q&A, executive Q&A):
- `coleman_liau`: Coleman-Liau readability index
- `dale_chall`: Dale-Chall readability score
- `automated_readability`: Automated Readability Index (ARI)
- `flesch_ease`: Flesch Reading Ease score
- `flesch_kincaid`: Flesch-Kincaid Grade Level
- `gunning_fog`: Gunning Fog Index
- `smog_index`: SMOG readability formula
- `overall`: Overall readability score

### Section Breakdown
The dataset analyzes four distinct sections of each earnings call:
1. Full Document: Complete transcript
2. Presentation: Company's prepared remarks
3. Analyst Q&A: Questions from analysts
4. Executive Q&A: Responses from company executives

Each section is analyzed independently for all metrics, allowing for comparison between different parts of the earnings call.

### Data Format
- All data is stored in CSV format
- Each row represents one earnings call
- Columns are consistently ordered regardless of which analyses are enabled
- Missing or disabled analyses are represented as NULL values
- All numerical values are stored as native Python types (not NumPy types)
