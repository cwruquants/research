"""Convert the word-list spreadsheets in data/word_sets/original_xlsx/ to CSV.

This is how the CSVs in data/word_sets/ were made: one CSV per sheet, no header, one
keyword per row. A workbook with several sheets (Garcia_MLWords.xlsx) gives one CSV per
sheet, named <workbook>_<sheet>.csv. Existing files are never overwritten.

Two CSVs were edited after conversion, so they differ from this output: Risk.csv
(duplicate words removed) and LM_Negative.csv (question and ill removed, 2026-11-01).
Activeness.xlsx became activeness.csv.

    uv run python scripts/split_garcia_word_lists.py --out /tmp/word_sets_check
"""

import argparse
from pathlib import Path

import pandas as pd

from research.config import WORD_SETS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--xlsx-dir", type=Path, default=WORD_SETS_DIR / "original_xlsx")
    parser.add_argument("--out", type=Path, required=True, help="Folder for the CSVs")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for xlsx_path in sorted(args.xlsx_dir.glob("*.xlsx")):
        # The lists have no header row, and each sheet of a workbook is its own list
        sheets = pd.read_excel(xlsx_path, sheet_name=None, header=None)
        for sheet_name, df in sheets.items():
            csv_name = f"{xlsx_path.stem}.csv" if len(sheets) == 1 else f"{xlsx_path.stem}_{sheet_name}.csv"
            csv_path = args.out / csv_name
            if csv_path.exists():
                print(f"Skipped {csv_name}: already exists")
                continue
            df.to_csv(csv_path, index=False, header=False)
            print(f"{xlsx_path.name} [{sheet_name}] -> {csv_name}")


if __name__ == "__main__":
    main()
