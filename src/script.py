from functions import extract_text, extract_exposure, extract_exposure2, csv_to_list, sentiment_score, extract_company_info, calculate_risk_word_percentage
from readability import coleman_liau, dale_chall, automated_readability, flesch_ease, flesch_kincaid, gunning_fog, smog_index, overall
from sentiment import LM_Positive, LM_Negative, LM_net_sentiment, LM_Polarity, LM_Subjectivity
import json
import os
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
from transformers import pipeline
import numpy as np

def model3v1(analyze_path, exposure_csv, risk_path, n):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Extracting Text...")
    text = extract_text(analyze_path)

    logging.info("Loading Exposure Word List...")
    exposure_word_list = csv_to_list(exposure_csv)
    # print(exposure_word_list)

    logging.info("Calculating Risk-Word Percentage...")
    risk_list = calculate_risk_word_percentage(exposure, "src/data/risk.csv")
    print("Risk Percentage: ", risk_list[1])

    return 0

def model5v1(analyze_path, exposure_csv, n):
    """
        Model 5 Pipeline:
        - text extraction from earnings call
        - exposure csv to exposure word list
        - exposure search with +- parameter
        - sentiment analysis on found exposure

        Returns:
            dict with exposure strings, sentiment score, pos/neg
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Extracting Text...")
    text = extract_text(analyze_path)

    logging.info("Loading Exposure Word List...")
    exposure_word_list = csv_to_list(exposure_csv)
    # print(exposure_word_list)

    logging.info("Calculating Exposure...")
    exposure = extract_exposure(text, exposure_word_list, window=n)
    # print(exposure)

    logging.info("Calculating Risk-Word Percentage...")
    risk_list = calculate_risk_word_percentage(exposure, "src/data/risk.csv")
    print("Risk Percentage: ", risk_list[1])

    logging.info("Finding Sentiment...")
    final = sentiment_score(exposure)

    return final

def model5v2(analyze_path, exposure_csv, n):
    """
        Model 5 Pipeline:
        - text extraction from earnings call
        - exposure csv to exposure word list
        - exposure search with KEYBERT and +- parameter
        - sentiment analysis on found exposure

        Returns:
            dict with exposure strings, sentiment score, pos/neg
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Extracting Text...")
    text = extract_text(analyze_path)

    logging.info("Loading Exposure Word List...")
    exposure_word_list = csv_to_list(exposure_csv)
    print(exposure_word_list)

    logging.info("Calculating Exposure...")
    exposure = extract_exposure2(text, exposure_word_list, buffer=n)
    print(exposure)

    logging.info("Calculating Risk-Word Percentage...")
    risk = calculate_risk_word_percentage(exposure, "src/data/risk.csv")
    print("Risk percentage: ", risk)

    logging.info("Finding Sentiment...")
    final = sentiment_score(exposure)

    return final

def model5_f(folder_path, exposure_csv, buffer, output_file):
    """
    Processes a folder of earnings call transcripts, applies Model 5, and saves results in a JSON file.

    Inputs:
    - folder_path: str, path to the folder containing XML earnings call transcripts.
    - exposure_csv: str, comma-separated exposure words.
    - buffer: int, number of words before and after the exposure word.
    - output_file: str, JSON file to save the results.

    Output:
    - A JSON file containing:
        - File name
        - Total count of exposure instances
        - Positive count & percentage
        - Neutral count & percentage
        - Negative count & percentage
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    results = {}

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    count = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xml"):
            file_path = os.path.join(folder_path, file_name)
            logging.info(f"Processing {file_name}...")

            try:
                # run Model 5 pipeline
                exposure_dict = model5v2(file_path, exposure_csv, buffer)

                # get company info
                company_info = extract_company_info(file_path) # List[company name, ticker, earnings call date, city]
                
                # occurence count
                total_count = len(exposure_dict)
                positive_count = sum(1 for v in exposure_dict.values() if v["label"] == "positive")
                neutral_count = sum(1 for v in exposure_dict.values() if v["label"] == "neutral")
                negative_count = sum(1 for v in exposure_dict.values() if v["label"] == "negative")

                # percentage calc
                positive_percentage = round((positive_count / total_count) * 100, 2) if total_count > 0 else 0
                neutral_percentage = round((neutral_count / total_count) * 100, 2) if total_count > 0 else 0
                negative_percentage = round((negative_count / total_count) * 100, 2) if total_count > 0 else 0

                # save
                results[file_name] = {
                    "Company": company_info[0],
                    "Ticker": company_info[1],
                    "Earnings Call Date": company_info[2],
                    "City": company_info[3],
                    "total_count": total_count,
                    "positive_count": positive_count,
                    "positive_percentage": positive_percentage,
                    "neutral_count": neutral_count,
                    "neutral_percentage": neutral_percentage,
                    "negative_count": negative_count,
                    "negative_percentage": negative_percentage,
                }

                logging.info(f"Finished processing {file_name}")

            except Exception as e:
                logging.error(f"Error processing {file_name}: {e}")

        # TRIAL STUFF
        count += 1
        if count > 10:
            break
        # REMOVE THIS LATER

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Results saved to {output_file}")

def decompose_earnings_call(xml_path, output_json=None):
    """
    Decompose an earnings call transcript into three sections:
    1. Presentation section
    2. Q&A executive responses (with speaker tracking)
    3. Q&A analyst questions (with speaker tracking)
    
    Args:
        xml_path (str): Path to the earnings call XML file
        output_json (str, optional): Path to save the JSON output. If None, results are returned but not saved.
        
    Returns:
        dict: Dictionary containing the three sections with their respective text
    """
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get the transcript text
    body = root.find(".//Body").text
    
    # Initialize sections
    presentation = []
    qa_executive = []
    qa_analyst = []
    
    # Split the text into lines
    lines = body.split('\n')
    
    # Flags to track current section and speaker
    in_presentation = True
    in_qa = False
    current_speaker = None
    current_text = []
    
    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section transitions
        if "Questions and Answers" in line:
            in_presentation = False
            in_qa = True
            continue
            
        if in_presentation:
            # Skip headers and footers
            if "Corporate Participants" in line or "Conference Call Participants" in line:
                continue
            if "================================================================================" in line:
                continue
            presentation.append(line)
            
        elif in_qa:
            # Skip separator lines and operator messages
            if "--------------------------------------------------------------------------------" in line:
                continue
            if "Operator" in line:
                continue
            if "Your line is now open" in line:
                continue
            if "(Operator Instructions)" in line:
                continue
                
            # Check for speaker changes
            if any(title in line for title in ["Analyst", "Research", "Capital Markets", "Chairman", "CEO", "CFO", "President", "EVP", "VP"]):
                # If we have a current speaker, save their text
                if current_speaker:
                    if any(title in current_speaker for title in ["Analyst", "Research", "Capital Markets"]):
                        qa_analyst.append({
                            "speaker": current_speaker,
                            "text": "\n".join(current_text)
                        })
                    elif any(title in current_speaker for title in ["Chairman", "CEO", "CFO", "President", "EVP", "VP"]):
                        qa_executive.append({
                            "speaker": current_speaker,
                            "text": "\n".join(current_text)
                        })
                
                # Start new speaker
                current_speaker = line
                current_text = []
            else:
                # Add line to current speaker's text
                if current_speaker:
                    current_text.append(line)
    
    # Handle the last speaker's text
    if current_speaker and current_text:
        if any(title in current_speaker for title in ["Analyst", "Research", "Capital Markets"]):
            qa_analyst.append({
                "speaker": current_speaker,
                "text": "\n".join(current_text)
            })
        elif any(title in current_speaker for title in ["Chairman", "CEO", "CFO", "President", "EVP", "VP"]):
            qa_executive.append({
                "speaker": current_speaker,
                "text": "\n".join(current_text)
            })
    
    result = {
        "presentation": "\n".join(presentation),
        "qa_executive": qa_executive,
        "qa_analyst": qa_analyst
    }
    
    # Save to JSON if output path is provided
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=4)
        logging.info(f"Results saved to {output_json}")
    
    return result

def analyze_presentation(text, exposure_csv, n, sentiment_analyzer=None):
    """
    Analyze the presentation section of an earnings call using our models.
    
    Args:
        text (str): The presentation section text
        exposure_csv (str): Path to the exposure words CSV file
        n (int): Window size for exposure calculation
        sentiment_analyzer: Optional pre-initialized sentiment analyzer
        
    Returns:
        dict: Results containing exposure, risk percentage, and sentiment analysis
    """
    logging.info("Analyzing presentation section...")
    
    # Load exposure words
    exposure_word_list = csv_to_list(exposure_csv)
    
    # Calculate exposure
    exposure = extract_exposure(text, exposure_word_list, window=n)
    
    # Calculate risk percentage
    risk = calculate_risk_word_percentage(exposure, "src/data/paper_word_sets/risk.csv")
    
    # Calculate sentiment
    sentiment = sentiment_score(exposure, sentiment_analyzer)
    
    return {
        "exposure": exposure,
        "risk_percentage": risk[1],
        "sentiment": sentiment
    }

def analyze_qa_section(qa_list, exposure_csv, n, sentiment_analyzer=None):
    """
    Analyze a Q&A section (either analyst questions or executive responses) using our models.
    
    Args:
        qa_list (list): List of dictionaries containing speaker and text
        exposure_csv (str): Path to the exposure words CSV file
        n (int): Window size for exposure calculation
        sentiment_analyzer: Optional pre-initialized sentiment analyzer
        
    Returns:
        list: List of analysis results for each Q&A entry
    """
    logging.info("Analyzing Q&A section...")
    
    # Load exposure words
    exposure_word_list = csv_to_list(exposure_csv)
    
    results = []
    for entry in qa_list:
        # Calculate exposure for this entry
        exposure = extract_exposure(entry["text"], exposure_word_list, window=n)
        
        # Calculate risk percentage
        risk = calculate_risk_word_percentage(exposure, "src/data/paper_word_sets/risk.csv")
        
        # Calculate sentiment
        sentiment = sentiment_score(exposure, sentiment_analyzer)
        
        results.append({
            "speaker": entry["speaker"],
            # "text": entry["text"],
            "exposure": exposure,
            "risk_percentage": risk[1],
            "sentiment": sentiment
        })
    
    return results

def analyze_earnings_call(xml_path, exposure_csv, n, output_json=None):
    """
    Full analysis of an earnings call transcript, including decomposition and model analysis.
    
    Args:
        xml_path (str): Path to the earnings call XML file
        exposure_csv (str): Path to the exposure words CSV file
        n (int): Window size for exposure calculation
        output_json (str, optional): Path to save the JSON output
        
    Returns:
        dict: Complete analysis results
    """
    # Initialize sentiment analyzer once
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    
    # First decompose the earnings call
    decomposed = decompose_earnings_call(xml_path)
    
    # Analyze each section
    presentation_analysis = analyze_presentation(decomposed["presentation"], exposure_csv, n, sentiment_analyzer)
    analyst_analysis = analyze_qa_section(decomposed["qa_analyst"], exposure_csv, n, sentiment_analyzer)
    executive_analysis = analyze_qa_section(decomposed["qa_executive"], exposure_csv, n, sentiment_analyzer)
    
    # Combine results
    results = {
        "presentation": presentation_analysis,
        "analyst_qa": analyst_analysis,
        "executive_qa": executive_analysis
    }
    
    # Save to JSON if output path is provided
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Analysis results saved to {output_json}")
    
    return results

def process_earnings_call(xml_path, exposure_csv, n):
    """
    Process an earnings call transcript and extract all features for dataset building.
    
    Args:
        xml_path (str): Path to the earnings call XML file
        exposure_csv (str): Path to the exposure words CSV file
        n (int): Window size for exposure calculation
        
    Returns:
        dict: Dictionary containing all features for the earnings call
    """
    logging.info(f"Processing earnings call: {xml_path}")
    
    # Initialize sentiment analyzer once
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    
    # Get company info
    company_info = extract_company_info(xml_path)
    logging.info(f"Company info: {company_info}")
    
    # Decompose the earnings call
    decomposed = decompose_earnings_call(xml_path)
    logging.info("Earnings call decomposed into sections")
    
    # Load exposure words
    exposure_word_list = csv_to_list(exposure_csv)
    logging.info(f"Loaded {len(exposure_word_list)} exposure words")
    
    # Process entire document
    full_text = decomposed["presentation"] + "\n" + "\n".join([q["text"] for q in decomposed["qa_analyst"]]) + "\n" + "\n".join([q["text"] for q in decomposed["qa_executive"]])
    
    # Calculate exposure for entire document
    full_exposure = extract_exposure(full_text, exposure_word_list, window=n)
    logging.info(f"Full document exposure count: {len(full_exposure)}")
    full_risk = calculate_risk_word_percentage(full_exposure, "src/data/paper_word_sets/risk.csv")
    logging.info(f"Full document risk percentage: {full_risk[1]}")
    full_sentiment = sentiment_score(full_exposure, sentiment_analyzer)
    
    # Calculate readability metrics for full document
    full_readability = {
        "coleman_liau": coleman_liau(full_text),
        "dale_chall": dale_chall(full_text),
        "automated_readability": automated_readability(full_text),
        "flesch_ease": flesch_ease(full_text),
        "flesch_kincaid": flesch_kincaid(full_text),
        "gunning_fog": gunning_fog(full_text),
        "smog_index": smog_index(full_text),
        "overall": overall(full_text)
    }
    
    # Calculate sentiment metrics for full document
    full_sentiment_metrics = {
        "LM_Positive": LM_Positive(full_text),
        "LM_Negative": LM_Negative(full_text),
        "LM_net_sentiment": LM_net_sentiment(full_text),
        "LM_Polarity": LM_Polarity(full_text),
        "LM_Subjectivity": LM_Subjectivity(full_text)
    }
    
    # Process presentation section
    pres_exposure = extract_exposure(decomposed["presentation"], exposure_word_list, window=n)
    logging.info(f"Presentation section exposure count: {len(pres_exposure)}")
    pres_risk = calculate_risk_word_percentage(pres_exposure, "src/data/paper_word_sets/risk.csv")
    pres_sentiment = sentiment_score(pres_exposure, sentiment_analyzer)
    
    # Calculate readability metrics for presentation
    pres_readability = {
        "coleman_liau": coleman_liau(decomposed["presentation"]),
        "dale_chall": dale_chall(decomposed["presentation"]),
        "automated_readability": automated_readability(decomposed["presentation"]),
        "flesch_ease": flesch_ease(decomposed["presentation"]),
        "flesch_kincaid": flesch_kincaid(decomposed["presentation"]),
        "gunning_fog": gunning_fog(decomposed["presentation"]),
        "smog_index": smog_index(decomposed["presentation"]),
        "overall": overall(decomposed["presentation"])
    }
    
    # Calculate sentiment metrics for presentation
    pres_sentiment_metrics = {
        "LM_Positive": LM_Positive(decomposed["presentation"]),
        "LM_Negative": LM_Negative(decomposed["presentation"]),
        "LM_net_sentiment": LM_net_sentiment(decomposed["presentation"]),
        "LM_Polarity": LM_Polarity(decomposed["presentation"]),
        "LM_Subjectivity": LM_Subjectivity(decomposed["presentation"])
    }
    
    # Process Q&A sections
    analyst_text = "\n".join([q["text"] for q in decomposed["qa_analyst"]])
    analyst_exposure = extract_exposure(analyst_text, exposure_word_list, window=n)
    logging.info(f"Analyst Q&A exposure count: {len(analyst_exposure)}")
    analyst_risk = calculate_risk_word_percentage(analyst_exposure, "src/data/paper_word_sets/risk.csv")
    analyst_sentiment = sentiment_score(analyst_exposure, sentiment_analyzer)
    
    # Calculate readability metrics for analyst Q&A
    analyst_readability = {
        "coleman_liau": coleman_liau(analyst_text),
        "dale_chall": dale_chall(analyst_text),
        "automated_readability": automated_readability(analyst_text),
        "flesch_ease": flesch_ease(analyst_text),
        "flesch_kincaid": flesch_kincaid(analyst_text),
        "gunning_fog": gunning_fog(analyst_text),
        "smog_index": smog_index(analyst_text),
        "overall": overall(analyst_text)
    }
    
    # Calculate sentiment metrics for analyst Q&A
    analyst_sentiment_metrics = {
        "LM_Positive": LM_Positive(analyst_text),
        "LM_Negative": LM_Negative(analyst_text),
        "LM_net_sentiment": LM_net_sentiment(analyst_text),
        "LM_Polarity": LM_Polarity(analyst_text),
        "LM_Subjectivity": LM_Subjectivity(analyst_text)
    }
    
    executive_text = "\n".join([q["text"] for q in decomposed["qa_executive"]])
    executive_exposure = extract_exposure(executive_text, exposure_word_list, window=n)
    logging.info(f"Executive Q&A exposure count: {len(executive_exposure)}")
    executive_risk = calculate_risk_word_percentage(executive_exposure, "src/data/paper_word_sets/risk.csv")
    executive_sentiment = sentiment_score(executive_exposure, sentiment_analyzer)
    
    # Calculate readability metrics for executive Q&A
    executive_readability = {
        "coleman_liau": coleman_liau(executive_text),
        "dale_chall": dale_chall(executive_text),
        "automated_readability": automated_readability(executive_text),
        "flesch_ease": flesch_ease(executive_text),
        "flesch_kincaid": flesch_kincaid(executive_text),
        "gunning_fog": gunning_fog(executive_text),
        "smog_index": smog_index(executive_text),
        "overall": overall(executive_text)
    }
    
    # Calculate sentiment metrics for executive Q&A
    executive_sentiment_metrics = {
        "LM_Positive": LM_Positive(executive_text),
        "LM_Negative": LM_Negative(executive_text),
        "LM_net_sentiment": LM_net_sentiment(executive_text),
        "LM_Polarity": LM_Polarity(executive_text),
        "LM_Subjectivity": LM_Subjectivity(executive_text)
    }
    
    return {
        "company_info": {
            "name": company_info[0],
            "ticker": company_info[1],
            "date": company_info[2],
            "city": company_info[3]
        },
        "full_document": {
            "exposure_count": len(full_exposure),
            "risk_percentage": full_risk[1],
            "sentiment": full_sentiment,
            "readability": full_readability,
            "sentiment_metrics": full_sentiment_metrics
        },
        "presentation": {
            "exposure_count": len(pres_exposure),
            "risk_percentage": pres_risk[1],
            "sentiment": pres_sentiment,
            "readability": pres_readability,
            "sentiment_metrics": pres_sentiment_metrics
        },
        "analyst_qa": {
            "exposure_count": len(analyst_exposure),
            "risk_percentage": analyst_risk[1],
            "sentiment": analyst_sentiment,
            "readability": analyst_readability,
            "sentiment_metrics": analyst_sentiment_metrics
        },
        "executive_qa": {
            "exposure_count": len(executive_exposure),
            "risk_percentage": executive_risk[1],
            "sentiment": executive_sentiment,
            "readability": executive_readability,
            "sentiment_metrics": executive_sentiment_metrics
        }
    }

def build_dataset(folder_path, exposure_csv, n, output_csv, max_files=None):
    """
    Process earnings calls in a folder and build a comprehensive dataset.
    
    Args:
        folder_path (str): Path to folder containing earnings call XML files
        exposure_csv (str): Path to the exposure words CSV file
        n (int): Window size for exposure calculation
        output_csv (str): Path to save the output CSV file
        max_files (int, optional): Maximum number of files to process. If None, processes all files.
    """
    import os
    import csv
    from pathlib import Path
    
    # Get all XML files in the folder
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    
    # Sort files for consistent processing order
    xml_files.sort()
    
    # Limit number of files if max_files is specified
    if max_files is not None:
        xml_files = xml_files[:max_files]
        logging.info(f"Processing {len(xml_files)} files (limited by max_files={max_files})")
    else:
        logging.info(f"Processing all {len(xml_files)} files")
    
    # Define CSV headers
    headers = [
        'file_name',
        'company_name',
        'ticker',
        'date',
        'city',
        # Full document metrics
        'full_doc_exposure_count',
        'full_doc_risk_percentage',
        'full_doc_positive_count',
        'full_doc_positive_percentage',
        'full_doc_neutral_count',
        'full_doc_neutral_percentage',
        'full_doc_negative_count',
        'full_doc_negative_percentage',
        'full_doc_coleman_liau',
        'full_doc_dale_chall',
        'full_doc_automated_readability',
        'full_doc_flesch_ease',
        'full_doc_flesch_kincaid',
        'full_doc_gunning_fog',
        'full_doc_smog_index',
        'full_doc_overall',
        'full_doc_lm_positive',
        'full_doc_lm_negative',
        'full_doc_lm_net_sentiment',
        'full_doc_lm_polarity',
        'full_doc_lm_subjectivity',
        # Presentation metrics
        'presentation_exposure_count',
        'presentation_risk_percentage',
        'presentation_positive_count',
        'presentation_positive_percentage',
        'presentation_neutral_count',
        'presentation_neutral_percentage',
        'presentation_negative_count',
        'presentation_negative_percentage',
        'presentation_coleman_liau',
        'presentation_dale_chall',
        'presentation_automated_readability',
        'presentation_flesch_ease',
        'presentation_flesch_kincaid',
        'presentation_gunning_fog',
        'presentation_smog_index',
        'presentation_overall',
        'presentation_lm_positive',
        'presentation_lm_negative',
        'presentation_lm_net_sentiment',
        'presentation_lm_polarity',
        'presentation_lm_subjectivity',
        # Analyst Q&A metrics
        'analyst_qa_exposure_count',
        'analyst_qa_risk_percentage',
        'analyst_qa_positive_count',
        'analyst_qa_positive_percentage',
        'analyst_qa_neutral_count',
        'analyst_qa_neutral_percentage',
        'analyst_qa_negative_count',
        'analyst_qa_negative_percentage',
        'analyst_qa_coleman_liau',
        'analyst_qa_dale_chall',
        'analyst_qa_automated_readability',
        'analyst_qa_flesch_ease',
        'analyst_qa_flesch_kincaid',
        'analyst_qa_gunning_fog',
        'analyst_qa_smog_index',
        'analyst_qa_overall',
        'analyst_qa_lm_positive',
        'analyst_qa_lm_negative',
        'analyst_qa_lm_net_sentiment',
        'analyst_qa_lm_polarity',
        'analyst_qa_lm_subjectivity',
        # Executive Q&A metrics
        'executive_qa_exposure_count',
        'executive_qa_risk_percentage',
        'executive_qa_positive_count',
        'executive_qa_positive_percentage',
        'executive_qa_neutral_count',
        'executive_qa_neutral_percentage',
        'executive_qa_negative_count',
        'executive_qa_negative_percentage',
        'executive_qa_coleman_liau',
        'executive_qa_dale_chall',
        'executive_qa_automated_readability',
        'executive_qa_flesch_ease',
        'executive_qa_flesch_kincaid',
        'executive_qa_gunning_fog',
        'executive_qa_smog_index',
        'executive_qa_overall',
        'executive_qa_lm_positive',
        'executive_qa_lm_negative',
        'executive_qa_lm_net_sentiment',
        'executive_qa_lm_polarity',
        'executive_qa_lm_subjectivity'
    ]
    
    # Process each file and collect results
    all_results = []
    for xml_file in xml_files:
        try:
            logging.info(f"Processing {xml_file}...")
            result = process_earnings_call(os.path.join(folder_path, xml_file), exposure_csv, n)
            
            # Convert to flat structure
            row = {
                'file_name': xml_file,
                'company_name': result['company_info']['name'],
                'ticker': result['company_info']['ticker'],
                'date': result['company_info']['date'],
                'city': result['company_info']['city']
            }
            
            # Helper function to process each section
            def process_section(section_name, section_data):
                # Exposure and risk
                row[f'{section_name}_exposure_count'] = section_data['exposure_count']
                row[f'{section_name}_risk_percentage'] = section_data['risk_percentage']
                
                # Sentiment counts and percentages
                sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
                for sentiment in section_data['sentiment'].values():
                    sentiment_counts[sentiment['label']] += 1
                
                total = sum(sentiment_counts.values()) if sum(sentiment_counts.values()) > 0 else 1
                row[f'{section_name}_positive_count'] = sentiment_counts['positive']
                row[f'{section_name}_positive_percentage'] = (sentiment_counts['positive'] / total) * 100
                row[f'{section_name}_neutral_count'] = sentiment_counts['neutral']
                row[f'{section_name}_neutral_percentage'] = (sentiment_counts['neutral'] / total) * 100
                row[f'{section_name}_negative_count'] = sentiment_counts['negative']
                row[f'{section_name}_negative_percentage'] = (sentiment_counts['negative'] / total) * 100
                
                # Readability metrics
                for metric, value in section_data['readability'].items():
                    row[f'{section_name}_{metric}'] = value
                
                # Sentiment metrics
                for metric, value in section_data['sentiment_metrics'].items():
                    row[f'{section_name}_{metric.lower()}'] = value
            
            # Process each section
            process_section('full_doc', result['full_document'])
            process_section('presentation', result['presentation'])
            process_section('analyst_qa', result['analyst_qa'])
            process_section('executive_qa', result['executive_qa'])
            
            all_results.append(row)
            logging.info(f"Completed processing {xml_file}")
            
        except Exception as e:
            logging.error(f"Error processing {xml_file}: {str(e)}")
    
    # Write results to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_results)
    
    logging.info(f"Dataset built and saved to {output_csv}")
    logging.info(f"Processed {len(all_results)} files successfully")

def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

if __name__ == "__main__":
    # Example usage
    input_xml = "/Users/efang/Desktop/coding/research/src/data/earnings_calls/ex1.xml"
    exposure_csv = "src/data/paper_word_sets/political_words.csv"
    
    # Process a single earnings call
    result = process_earnings_call(input_xml, exposure_csv, n=10)
    
    # Convert numpy types to Python native types
    result = convert_numpy_types(result)
    
    # Save to JSON file
    with open("output.json", 'w') as f:
        json.dump(result, f, indent=4)
    
    # Build dataset for all earnings calls
    folder_path = "/Users/efang/Documents/Transcript/2016"
    output_csv = "earnings_calls_dataset.csv"
    build_dataset(folder_path, exposure_csv, n=10, output_csv=output_csv, max_files=25)

