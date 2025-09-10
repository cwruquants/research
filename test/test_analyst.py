#!/usr/bin/env python3
"""
Test script for the Analyst module.
Port of test_analyst.ipynb to a regular Python file.
"""

import os
from pathlib import Path

from src.analysis.analyst_module import Analyst
from src.document.abstract_classes.setup_module import Setup


def main():
    """Main test function."""
    
    # Get the project root directory (assuming this script is in test/)
    project_root = Path(__file__).parent.parent
    
    # Setup configuration
    setup = Setup(
        sheet_name_positive='ML_negative_unigram',
        sheet_name_negative='ML_positive_unigram',
        file_path=str(project_root / "data" / "word_sets" / "Garcia_MLWords.xlsx"),
        hf_model='cardiffnlp/twitter-roberta-base-sentiment-latest',
        device=-1
    )
    
    # Test the new fit_single_document functionality
    # This will create a directory structure and save results as TOML
    
    # Create an analyst with keyword path
    analyst = Analyst(keyword_path=str(project_root / "data" / "paper_word_sets" / "political_words.csv"))
    
    # Test with a sample earnings call
    earnings_call_path = str(project_root / "data" / "earnings_calls" / "ex1.xml")
    
    # Run the analysis - this will create a new directory in results/
    result = analyst.fit_single_document(
        earnings_call_path=earnings_call_path,
        similarity="cosine",
    )
    
    print("Analysis completed!")
    print(f"Output directory: {result['output_directory']}")
    print(f"TOML file: {result['toml_path']}")
    print(f"Exposure results: {result['exposure_results_path']}")
    
    # Show some summary statistics
    qa_fit = result['qa_fit']
    pres_fit = result['pres_fit']
    exposure_results = result['exposure_results']
    
    print(f"\nSentiment Analysis Summary:")
    print(f"Q&A Section - Sentiment: {qa_fit.sentiment:.4f}, ML: {qa_fit.ML}, LM: {qa_fit.LM}, HIV4: {qa_fit.HIV4}")
    print(f"Presentation Section - Sentiment: {pres_fit.sentiment:.4f}, ML: {pres_fit.ML}, LM: {pres_fit.LM}, HIV4: {pres_fit.HIV4}")
    
    print(f"\nMatching Analysis Summary:")
    print(f"Total keywords searched: {exposure_results.total_keywords_searched}")
    print(f"Total matches found: {exposure_results.total_direct_matches + exposure_results.total_cosine_matches}")
    print(f"Keywords with matches: {exposure_results.total_keywords_with_matches}")
    
    # Create another analyst with setup
    a = Analyst(
        setup=setup,
        keyword_path=str(project_root / "test" / "test_keywords.csv"),
    )
    
    # Run analysis with the setup analyst
    a.fit_single_document(
        earnings_call_path=str(project_root / "data" / "earnings_calls" / "ex1.xml"),
        setup_dict=None,
        similarity="cosine"
    )


if __name__ == "__main__":
    main()
