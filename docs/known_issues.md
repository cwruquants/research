# Known issues in the stored data

Read this before using a number from the database. Each item says what is wrong, which
rows it touches, and what to do until it is fixed.

## Tight-layout transcripts lost their Q&A (stored text measures)

Before the parser fix of 2026-10-31 (cwruquants/research#56), `load_document` cleaned the
Presentation and Q&A together, and in transcripts with no blank lines around one-line
turns its speaker-line skip drifted one line off. It then kept speaker names and dropped
the answers. Hit calls lost about 45% of their words.

- **Rows:** about half the earnings calls dated 2017-04 to 2019-12, 3–14% per quarter in
  2020–2021, and scattered older calls.
- **Affected:** `exposure_runs`, `keyword_matches` (all made before the fix).
- **Until reprocessed:** exclude calls dated 2017-04-01 to 2019-12-31 from text analyses
  and say so. The current code parses these files correctly; the stored rows don't.

## Word counts and exposures use different text

`documents.num_words`, `num_sentences` and readability come from the whole XML body.
Exposure counts come from the Presentation and Q&A only. A ratio such as matches per
word mixes the two. Use `exposure_runs.total_sentences_in_document` as the denominator
for exposures.

## "question" dominates the risk keyword set

About 67% of stored `risk` matches are the words "question" or "questions" (analyst Q&A
boilerplate). Count risk exposure without them. `Risk_v2.csv` drops them, and the
Loughran–McDonald negative list no longer has "question" or "ill" (2026-11-01), but the
stored `risk` runs still use `Risk.csv`.

## Sentiment sums grow with call length

`lm_score`, `hiv4_score` and `ml_score` are sums over sentences, not rates. Divide by
`num_sentences` or `num_words` before comparing calls of different lengths.

## Links and returns

- Calls whose company row has no U.S. listing (about 138,500) or no CRSP match have no
  `document_crsp_links` row, so no returns.
- `mcap_pre` is missing for 1,737 of 352,115 linked calls (no CRSP price in the 30 days
  before day 0).
- For drift windows use `bhar_`, not `car_mm_`: subtracting the market-model alpha flips
  the sign of long-run drift.
- The parquet cache in `research returns surprise` is reused whenever its files exist,
  whatever the call set or date range. Delete the cache folder after changing either.
