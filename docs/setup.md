# Setup

## Python environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repo root:

```bash
uv sync --group dev              # Python 3.12, the package, pytest and ruff
uv sync --group dev --extra ml   # also torch, transformers, sentence-transformers
uv run pytest
```

`uv` downloads Python 3.12 if you don't have it, and the same command works on macOS,
Windows and Linux. Run everything through `uv run` (`uv run research ...`,
`uv run python ...`) so it uses the project environment.

## Settings

Commands read these environment variables. `.env.example` lists them; uv does not load a
`.env` file by itself, so export them in your shell or pass `uv run --env-file .env`.

| Variable | Needed by | Example |
|---|---|---|
| `DATABASE_URL` | `ingest`, `backfill` (write role); optional for `export` | `postgresql://research_app@<db-host>:5432/research` |
| `TEST_DATABASE_URL` | `pytest -m db` only; a throwaway database, never `research` | `postgresql://postgres@localhost:5432/scratch_test` |

Keep passwords out of URLs. Put them in the PostgreSQL password file instead:
`~/.pgpass` on macOS and Linux (`chmod 600`), `%APPDATA%\postgresql\pgpass.conf` on
Windows, one line per server:

```
<db-host>:5432:research:research_ro:<password>
```

## Database access

The database runs on a group machine reachable over the group's private network. Ask the
maintainer for network access and a role:

- `research_ro`: read only, for analysis and exports. This is what most people need.
- `research_app`: writes rows, for running `ingest` and `backfill`.

Then pull data with a saved or ad-hoc query:

```bash
export DATABASE_URL=postgresql://research_ro@<db-host>:5432/research
uv run research export --sql-file db/queries/events.sql --out results/events.csv.gz
uv run research export --sql "select * from public.keyword_sets" --out results/sets.csv
```

`research export` forces a read-only session and never overwrites a file.

## WRDS

`research returns fetch`, `mcap` and `surprise` pull from WRDS with your own WRDS account.
Create the `~/.pgpass` entry once, for example by running `wrds.Connection()` in Python, or
by adding the line `wrds-pgdata.wharton.upenn.edu:9737:wrds:<wrds username>:<password>`.
WRDS may send a Duo push when a connection opens.
