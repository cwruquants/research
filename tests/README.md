# tests/

```bash
uv run pytest                 # everything that needs no database
uv run pytest -m db           # integration tests; needs TEST_DATABASE_URL (see db/README.md)
```

- `unit/<area>/` mirrors `src/research/<area>/`.
- `fixtures/transcripts/` holds synthetic transcripts in the vendor layout, written by
  `fixtures/make_transcripts.py`. Never add real transcripts or text quoted from them.
- A change to parsing, matching, linking or returns should also be checked against real
  data before merging: run the old and new code on the same private sample and compare
  the outputs byte for byte.
