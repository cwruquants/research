import pandas as pd
from functions import extract_text, extract_exposure, extract_exposure2, extract_exposure3
import matplotlib.pyplot as plt
from collections import Counter
import os
from pathlib import Path

def process_single_transcript(file_path, seed_words, buffer, threshold, window):
    """
    Process a single transcript and return the extraction results.
    
    Args:
        file_path (str): Path to the earnings transcript XML file
        seed_words (list): List of seed words to search for
        buffer (int): Number of words to extract around each match for direct and keybert
        threshold (float): Similarity threshold for extract_exposure3
        window (int): Window size for extract_exposure3
        
    Returns:
        tuple: (direct_matches, keybert_matches, exposure3_matches)
    """
    try:
        text = extract_text(file_path)
        direct_matches = extract_exposure(text, seed_words, window=buffer)
        keybert_matches = extract_exposure2(text, seed_words, buffer=buffer)
        exposure3_matches = extract_exposure3(text, seed_words, threshold=threshold, window=window)
        return direct_matches, keybert_matches, exposure3_matches
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {}, {}, {}

def compare_multiple_transcripts(folder_path, seed_words, num_files=10, buffer=2, threshold=0.8, window=1):
    """
    Compare extraction methods across multiple transcripts.
    
    Args:
        folder_path (str): Path to folder containing transcript XML files
        seed_words (list): List of seed words to search for
        num_files (int): Number of files to process
        buffer (int): Number of words to extract around each match for direct and keybert
        threshold (float): Similarity threshold for extract_exposure3
        window (int): Window size for extract_exposure3
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
    total_exposure3_matches = Counter()
    all_exposure3_contexts = {}
    
    # Process each file
    for file_path in xml_files:
        print(f"\nProcessing {file_path.name}...")
        direct_matches, keybert_matches, exposure3_matches = process_single_transcript(
            str(file_path), seed_words, buffer=buffer, threshold=threshold, window=window
        )
        
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
        
        # Aggregate exposure3 matches
        for matched_keyword, contexts_list in exposure3_matches.items():
            total_exposure3_matches[matched_keyword] += len(contexts_list)
            if matched_keyword not in all_exposure3_contexts:
                all_exposure3_contexts[matched_keyword] = []
            all_exposure3_contexts[matched_keyword].extend(contexts_list)
    
    # Print overall statistics
    print("\n=== Overall Extraction Method Comparison ===")
    print(f"Number of files processed: {len(xml_files)}")
    print(f"Total direct matches: {sum(total_direct_matches.values())}")
    print(f"Total KeyBERT matches: {sum(total_keybert_matches.values())}")
    print(f"Total extract_exposure3 matches: {sum(total_exposure3_matches.values())}")
    
    # Print detailed results
    print("\nDirect Matches Summary:")
    for word, count in total_direct_matches.most_common():
        print(f"\nWord: {word}")
        print(f"Total occurrences: {count}")
        # print("All contexts:")
        # for i, context in enumerate(all_direct_contexts[word], 1):
        #     print(f"{i}. {context}")
    
    print("\nKeyBERT Matches Summary:")
    for word, count in total_keybert_matches.most_common():
        print(f"\nWord: {word}")
        print(f"Total occurrences: {count}")
        print("All contexts:")
        for i, context in enumerate(all_keybert_contexts[word], 1):
            print(f"{i}. {context}")
    
    print("\nextract_exposure3 Matches Summary:")
    for word, count in total_exposure3_matches.most_common():
        print(f"\nWord: {word}")
        print(f"Total occurrences: {count}")
        print("All contexts:")
        for i, context in enumerate(all_exposure3_contexts[word], 1):
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
    folder_path = "src/data/earnings_test/"
    seed_words = ["risk", "uncertainty", "challenge", "volatility", "concern"]
    
    compare_multiple_transcripts(folder_path, seed_words, num_files=10, buffer=2, threshold=0.7, window=5)