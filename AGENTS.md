# Notes for coding agents

Read `README.md` first, then `docs/pipeline.md` (which code writes which table) and
`CONTEXT.md` (vocabulary). This file only adds the rules that the code cannot show.

## Commands

- Set up: `uv sync --group dev` (add `--extra ml` for sentiment or cosine matching).
- Test: `uv run pytest`. Lint: `uv run ruff check .`. Every step: `uv run research --help`.
- Pull data: `uv run research export --sql-file db/queries/<name>.sql --out <new file>`.
  It refuses to overwrite files and always runs read-only.

## Hard rules

- The repo is public. Never commit hostnames, IP addresses, usernames, emails, Supabase
  project refs, passwords or tokens. Use placeholders such as `<db-host>`.
- Never commit licensed data or text from it: transcripts, CRSP, LSEG and I/B/E/S rows,
  firm-level outputs, or sentences quoted from calls. It lives in `data/private/`
  (git-ignored). Test fixtures are synthetic (`tests/fixtures/make_transcripts.py`).
- The database is read-only unless the user approves a write. Read with the `research_ro`
  role. Every approved change to rows or schema also gets a SQL file in `db/migrations/`
  (schema) or `db/fixes/` (one-off data changes), so the repo can rebuild the database.
- Refactors must not change stored numbers. Compare old and new outputs on real data
  before merging anything that touches parsing, matching, linking or returns.
- No AI attribution lines (Co-Authored-By, "Generated with ...") in commits or PRs.

## Gotchas

- Transcripts in the tight layout lost their Q&A before the parser fix of 2026-10-31.
  Stored exposures and scores still come from the old text, so exclude calls dated
  2017-04-01 to 2019-12-31 from text analyses (see `docs/known_issues.md`).
- About two-thirds of stored `risk` keyword matches are the words "question(s)".
- `documents.num_words` uses the whole XML body; keyword matching uses only the
  Presentation and Q&A sections, so ratios that mix them have mismatched denominators.
