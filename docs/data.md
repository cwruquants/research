# Data sources

Licensed data never goes into git. It lives under `data/private/`, which `.gitignore`
excludes, and every step takes its input paths as flags. Ask the maintainer for access;
each source has its own license terms, and some come through Prof. Bae.

| Source | What | Used by | Where it goes |
|---|---|---|---|
| Earnings-call transcripts (Thomson StreetEvents / LSEG) | one XML per call, 2002–2024, zipped by year; the database holds 2002–2021, and the 2022–2024 files are not loaded yet | `ingest`, `backfill` | the group's shared drive; `backfill --source` reads the zips, `ingest --input-dir` needs an unzipped year folder |
| LSEG QuoteFile, CUSIPFile | RIC → InstrumentPI → CUSIP bridge | `link route-a`, `link coverage` | `data/private/lseg/` |
| CRSP stocknames (WRDS) | names, tickers and NCUSIPs with date ranges | `link route-a`, `link route-b` | `data/private/crsp/stocknames.csv` |
| CRSP daily files (WRDS) | `dsf`, `dsi`, delistings | `returns fetch`, `returns compute` | `data/private/wrds/crsp/` |
| I/B/E/S (WRDS) | actuals, consensus, detail, CRSP link | `returns surprise` | `data/private/wrds/ibes/` |

`results/event_returns/crsp` and `results/event_returns/ibes` may exist as links to the
`data/private/wrds/` folders, for scripts written before the move.

## What may be shared

- **Fine to commit:** code, SQL, word lists in `data/word_sets/`, synthetic fixtures, and
  aggregate results (counts, coefficients, charts with no firm-level rows).
- **Never commit:** transcript text or sentences quoted from calls, any file of firm-level
  rows built from CRSP, LSEG or I/B/E/S (including `results/`), and database exports.

## Word lists

See [data/README.md](../data/README.md) for where each list in `data/word_sets/` comes from.
