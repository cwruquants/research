import os
import shutil
import pandas as pd
from pathlib import Path

def convert_and_organize_word_sets():
    # Define paths
    project_root = Path(__file__).parent.parent
    word_sets_dir = project_root / "data" / "word_sets"
    paper_word_sets_dir = project_root / "data" / "paper_word_sets"
    
    print(f"Project root: {project_root}")
    print(f"Word sets dir: {word_sets_dir}")
    print(f"Paper word sets dir: {paper_word_sets_dir}")
    
    if not word_sets_dir.exists():
        print(f"Creating {word_sets_dir}...")
        word_sets_dir.mkdir(parents=True, exist_ok=True)
        
    # 1. Convert .xlsx in data/word_sets to .csv
    print("\n--- Converting .xlsx to .csv in word_sets ---")
    xlsx_files = list(word_sets_dir.glob("*.xlsx"))
    if not xlsx_files:
        print("No .xlsx files found in word_sets.")
    
    for xlsx_path in xlsx_files:
        try:
            # Read excel
            df = pd.read_excel(xlsx_path)
            
            # Construct csv path
            csv_name = xlsx_path.stem + ".csv"
            csv_path = word_sets_dir / csv_name
            
            # Save as csv
            df.to_csv(csv_path, index=False)
            print(f"Converted: {xlsx_path.name} -> {csv_name}")
            
        except Exception as e:
            print(f"Error converting {xlsx_path.name}: {e}")

    # 2. Move files from data/paper_word_sets to data/word_sets
    print("\n--- Moving files from paper_word_sets to word_sets ---")
    if paper_word_sets_dir.exists():
        paper_files = list(paper_word_sets_dir.glob("*"))
        if not paper_files:
            print("No files found in paper_word_sets.")
        
        for file_path in paper_files:
            if file_path.is_file():
                dest_path = word_sets_dir / file_path.name
                
                # Handle collision
                if dest_path.exists():
                    print(f"Warning: {file_path.name} already exists in word_sets. Skipping move to avoid overwrite.")
                    # Optional: append suffix?
                    # new_name = f"{file_path.stem}_paper{file_path.suffix}"
                    # dest_path = word_sets_dir / new_name
                    # shutil.move(str(file_path), str(dest_path))
                    # print(f"Moved (renamed): {file_path.name} -> {new_name}")
                else:
                    try:
                        shutil.move(str(file_path), str(dest_path))
                        print(f"Moved: {file_path.name}")
                    except Exception as e:
                        print(f"Error moving {file_path.name}: {e}")
            else:
                 print(f"Skipping directory/other: {file_path.name}")
        
        # Check if empty and remove
        if not any(paper_word_sets_dir.iterdir()):
             print("paper_word_sets is empty. Removing directory.")
             paper_word_sets_dir.rmdir()
        else:
             print("paper_word_sets is not empty. Left as is.")
    else:
        print("data/paper_word_sets does not exist.")

    print("\nDone.")

if __name__ == "__main__":
    convert_and_organize_word_sets()
