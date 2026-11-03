# data/

| Folder | Tracked? | What |
|---|---|---|
| `word_sets/` | yes | Word lists, one keyword or phrase per cell |
| `word_sets/original_xlsx/` | yes | The spreadsheets some lists were converted from |
| `test_sets/politics.csv` | yes | Five political words for the onboarding demo |
| `private/` | **no** | Licensed data and private work; see [docs/data.md](../docs/data.md) |
| `earnings_calls/` | **no** | Local transcript samples, if you have them |

## Word lists

| File | Words | Source | Used by |
|---|---|---|---|
| `Risk.csv` | 139 | Risk and uncertainty synonyms (Hassan, Hollander, van Lent and Tahoun 2019 approach) | keyword set `risk` in the DB |
| `risk_paper.csv` | 139 | The same list as printed in the paper, with a header row | reference only |
| `Risk_v2.csv`, `Risk_v2_exclusions.csv` | 161, 11 | `Risk.csv` reviewed against labeled earnings-call sentences (2026-11), plus terms of art to exclude | not yet a keyword set |
| `activeness.csv` | 334 | Risk-management action words (manage, hedge, offset, ...) | keyword set `activeness` in the DB |
| `LM_*.csv` | varies | Loughran–McDonald sentiment categories, converted from `original_xlsx/` | reference; `lm_score` comes from pysentiment2's copy |
| `Garcia_MLWords_ML_{positive,negative}_{unigram,bigram}.csv` | varies | Garcia, Hu and Rohrer machine-learned finance words, one CSV per sheet of `original_xlsx/Garcia_MLWords.xlsx` (`scripts/split_garcia_word_lists.py`) | `ml_score` uses the unigram sheets |
| `political_bigrams.csv` | ~190k | Hassan et al. (2019) political bigrams with weights | experiments |
| `political_words.csv`, `political_words_extended.csv` | 58, 302 | Political vocabulary assembled by the group in 2025 | experiments |
| `climate_regulatory.csv` | 100 | Climate regulatory bigrams (Sautner, van Lent, Vilkov and Zhang 2023 approach) | experiments |

Bigram lists hold multi-word phrases. Direct matching finds phrases only as consecutive
words, so lists built from stripped text (stopwords removed) need `normalize_phrases`.
