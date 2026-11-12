# CWRU Quants research

Text measures from earnings-call transcripts, linked to stock returns. The pipeline parses
about 507,000 call transcripts (2002–2021), counts how often each call uses words from
research word lists (risk, Loughran–McDonald, Garcia, political, climate), links each call
to its CRSP security, and computes event-window returns around the call. Everything lands
in a Postgres database, `research`, that the group queries for analysis.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and git. No database or licensed data is needed
to run the tests or the demo.

```bash
git clone https://github.com/cwruquants/research.git
cd research
uv sync --group dev            # Python 3.12 and every dependency
uv run pytest                  # fast tests on synthetic transcripts
uv run research demo           # keyword matching on one transcript
uv run research --help         # every pipeline step
```

Add `--extra ml` to `uv sync` for sentiment scores and cosine matching (torch, transformers).

Next steps:

- **Analysing the data:** get read access to the database ([docs/setup.md](docs/setup.md)),
  then read [docs/pipeline.md](docs/pipeline.md) for what each table holds and
  [docs/known_issues.md](docs/known_issues.md) before trusting a number.
- **Changing the pipeline:** read [CONTEXT.md](CONTEXT.md) for the vocabulary, then the
  stage you are changing in [docs/pipeline.md](docs/pipeline.md).

## Repo map

| Path | What lives there |
|---|---|
| `src/research/` | The installable package. One folder per part of the pipeline: |
| `  transcripts/` | Parse the vendor XML (`parse.py`), split text into words and sentences, find files in folders and zips |
| `  matching/` | Keyword matching: direct and phrase matching (`direct.py`), cosine similarity (`cosine.py`) |
| `  features/` | Readability and sentiment scores |
| `  db/` | The only code that writes to the database, plus `research export` |
| `  pipelines/` | `ingest` (XML → documents rows) and `backfill` (documents → keyword exposures) |
| `  linking/` | Earnings call → CRSP PERMNO (route A by ticker, route B by event title) |
| `  returns/` | WRDS pulls, event-window returns, pre-call market cap, earnings surprise |
| `  config.py`, `cli.py` | Repo paths and settings; the `research` command |
| `config/keyword_sets.toml` | Which word lists run as keyword sets in the DB, with their matching options |
| `data/word_sets/` | The word lists ([data/README.md](data/README.md) says where each comes from) |
| `data/private/` | Licensed data (git-ignored). See [docs/data.md](docs/data.md) |
| `db/` | Schema, roles, migrations, saved queries, table loads and data fixes ([db/README.md](db/README.md)) |
| `tests/` | `pytest` suites; `tests/fixtures/` holds synthetic transcripts |
| `examples/`, `scripts/` | The onboarding demo; one-off tools kept as a record |
| `docs/` | Pipeline, setup, data sources, known issues, 2025 meeting notes |

## Working in this repo

- Branch from `main`, open a pull request, and squash-merge it. CI runs the tests and lint.
- Commit messages are plain imperative sentences ("Add pre-call market cap").
- Never commit licensed data (transcripts, CRSP, LSEG, I/B/E/S), firm-level outputs,
  hostnames, IP addresses or credentials. The repo is public.
- A change that alters numbers stored in the database needs its own pull request, with
  the SQL in `db/` and a note in `docs/known_issues.md` or the pull request.
