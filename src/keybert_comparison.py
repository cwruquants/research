import pandas as pd
from functions import extract_text, extract_exposure, extract_exposure2
import matplotlib.pyplot as plt
from collections import Counter
import os
from pathlib import Path

def process_single_transcript(file_path, seed_words, buffer=10):
    """
    Process a single transcript and return the extraction results.
    
    Args:
        file_path (str): Path to the earnings transcript XML file
        seed_words (list): List of seed words to search for
        buffer (int): Number of words to extract around each match
        
    Returns:
        tuple: (direct_matches, keybert_matches)
    """
    try:
        text = extract_text(file_path)
        direct_matches = extract_exposure(text, seed_words, window=buffer)
        keybert_matches = extract_exposure2(text, seed_words, buffer=buffer)
        return direct_matches, keybert_matches
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {}, {}

def compare_multiple_transcripts(folder_path, seed_words, num_files=10, buffer=10):
    """
    Compare extraction methods across multiple transcripts.
    
    Args:
        folder_path (str): Path to folder containing transcript XML files
        seed_words (list): List of seed words to search for
        num_files (int): Number of files to process
        buffer (int): Number of words to extract around each match
    """
    # Get list of XML files
    xml_files = list(Path(folder_path).glob("*.xml"))[:num_files]
    
    if not xml_files:
        print(f"No XML files found in {folder_path}")
        return
    
    # Initialize counters for aggregation
    total_direct_matches = Counter()
    total_keybert_matches = Counter()
    all_direct_contexts = {}
    all_keybert_contexts = {}
    
    # Process each file
    for file_path in xml_files:
        print(f"\nProcessing {file_path.name}...")
        direct_matches, keybert_matches = process_single_transcript(str(file_path), seed_words, buffer)
        
        # Aggregate direct matches
        for word, contexts in direct_matches.items():
            total_direct_matches[word] += len(contexts)
            if word not in all_direct_contexts:
                all_direct_contexts[word] = []
            all_direct_contexts[word].extend(contexts)
        
        # Aggregate KeyBERT matches
        for word, contexts in keybert_matches.items():
            total_keybert_matches[word] += len(contexts)
            if word not in all_keybert_contexts:
                all_keybert_contexts[word] = []
            all_keybert_contexts[word].extend(contexts)
    
    # Print overall statistics
    print("\n=== Overall Extraction Method Comparison ===")
    print(f"Number of files processed: {len(xml_files)}")
    print(f"Total direct matches: {sum(total_direct_matches.values())}")
    print(f"Total KeyBERT matches: {sum(total_keybert_matches.values())}")
    
    # Print detailed results
    print("\nDirect Matches Summary:")
    for word, count in total_direct_matches.most_common():
        print(f"\nWord: {word}")
        print(f"Total occurrences: {count}")
        print("All contexts:")
        for i, context in enumerate(all_direct_contexts[word], 1):
            print(f"{i}. {context}")
    
    print("\nKeyBERT Matches Summary:")
    for word, count in total_keybert_matches.most_common():
        print(f"\nWord: {word}")
        print(f"Total occurrences: {count}")
        print("All contexts:")
        for i, context in enumerate(all_keybert_contexts[word], 1):
            print(f"{i}. {context}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Number of matches per word
    plt.subplot(2, 1, 1)
    words = sorted(set(list(total_direct_matches.keys()) + list(total_keybert_matches.keys())))
    direct_values = [total_direct_matches.get(word, 0) for word in words]
    keybert_values = [total_keybert_matches.get(word, 0) for word in words]
    
    x = range(len(words))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], direct_values, width, label='Direct Matching')
    plt.bar([i + width/2 for i in x], keybert_values, width, label='KeyBERT')
    plt.xlabel('Words')
    plt.ylabel('Number of Matches')
    plt.title('Matches per Word: Direct vs KeyBERT')
    plt.xticks(x, words, rotation=45, ha='right')
    plt.legend()
    
    # Plot 2: Total matches comparison
    plt.subplot(2, 1, 2)
    total_direct = sum(total_direct_matches.values())
    total_keybert = sum(total_keybert_matches.values())
    
    plt.bar(['Direct Matching', 'KeyBERT'], [total_direct, total_keybert])
    plt.ylabel('Total Number of Matches')
    plt.title('Total Matches: Direct vs KeyBERT')
    
    plt.tight_layout()
    plt.savefig('extraction_comparison_multiple.png')
    plt.close()

if __name__ == "__main__":
    folder_path = "src/data/earnings_calls/2016"
    seed_words = ["risk", "uncertainty", "challenge", "volatility", "concern"]
    
    compare_multiple_transcripts(folder_path, seed_words, num_files=10) 