import pandas as pd
from pathlib import Path

def compare_risk_files():
    project_root = Path(__file__).parent.parent
    risk_csv_path = project_root / "data" / "word_sets" / "Risk.csv"
    risk_paper_csv_path = project_root / "data" / "word_sets" / "risk_paper.csv"

    # Read Risk.csv
    # It seems to have no header based on the prompt showing "risk", "risks" as first lines
    # But let's verify if the first line is used as header by pandas default
    try:
        # Assuming no header first, we'll check the first row
        df_risk = pd.read_csv(risk_csv_path, header=None)
        # If the first item is 'risk', it might be data. 
        risk_words = set(df_risk[0].astype(str).str.lower().str.strip())
    except Exception as e:
        print(f"Error reading Risk.csv: {e}")
        return

    # Read risk_paper.csv
    # The prompt showed "word" as the first line, likely a header.
    try:
        df_paper = pd.read_csv(risk_paper_csv_path)
        if 'word' in df_paper.columns:
            paper_words = set(df_paper['word'].astype(str).str.lower().str.strip())
        else:
            # Fallback if 'word' is not the header but the first line was 'word'
            # Check if the column name itself is 'word'
            if df_paper.columns[0].lower() == 'word':
                 paper_words = set(df_paper.iloc[:, 0].astype(str).str.lower().str.strip())
            else:
                # If it was treated as header but shouldn't have been, or vice versa
                # Let's just read all values
                 paper_words = set(df_paper.iloc[:, 0].astype(str).str.lower().str.strip())
                 # add the header itself if it looks like a word and not "word"
                 header_val = df_paper.columns[0]
                 if str(header_val).lower() != 'word':
                     paper_words.add(str(header_val).lower().strip())
    except Exception as e:
        print(f"Error reading risk_paper.csv: {e}")
        return

    # Comparison
    common = risk_words.intersection(paper_words)
    unique_risk = risk_words - paper_words
    unique_paper = paper_words - risk_words

    print(f"Total unique words in Risk.csv: {len(risk_words)}")
    print(f"Total unique words in risk_paper.csv: {len(paper_words)}")
    print(f"Common words: {len(common)}")
    print(f"Unique to Risk.csv: {len(unique_risk)}")
    print(f"Unique to risk_paper.csv: {len(unique_paper)}")

    print("\n--- Words unique to Risk.csv (Sample 10) ---")
    print(list(unique_risk)[:10])

    print("\n--- Words unique to risk_paper.csv (Sample 10) ---")
    print(list(unique_paper)[:10])
    
    if len(unique_risk) == 0 and len(unique_paper) == 0:
        print("\nThe sets are identical.")
    elif len(unique_risk) == 0:
        print("\nRisk.csv is a subset of risk_paper.csv")
    elif len(unique_paper) == 0:
        print("\nrisk_paper.csv is a subset of Risk.csv")

if __name__ == "__main__":
    compare_risk_files()
