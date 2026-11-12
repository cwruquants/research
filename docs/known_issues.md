# Known issues in the stored data

Read this before using a number from the database. Each item says what is wrong, which
rows it touches, and what to do until it is fixed.

The counts and the checks behind most items are in
[data_audit_2026-11-11.md](data_audit_2026-11-11.md), a read-only audit of every table.
Nothing it found has been fixed yet.

## Tight-layout transcripts lost their Q&A (stored text measures)

Before the parser fix of 2026-10-31 (cwruquants/research#56), `load_document` cleaned the
Presentation and Q&A together, and in transcripts with no blank lines around one-line
turns its speaker-line skip drifted one line off. It then kept speaker names and dropped
the answers. Hit calls lost about 45% of their words.

That fix covered `load_document` only. `analyze_document`, which fills `documents`, kept
cleaning the whole body in one pass until 2026-11-11. Both now take their text from one
function, `spoken_text`, so the code is right. The stored rows are not: nothing has been
recomputed.

- **Rows:** about half the earnings calls dated 2017-04 to 2019-12, 3–14% per quarter in
  2020–2021, and older calls. A per-call rule (below) flags 4–11% of earnings calls in
  each year from 2002 to 2014; that share is not yet confirmed against the raw files.
- **Affected:** `exposure_runs` and `keyword_matches` (all made before the fix), and in
  `documents` the columns `num_words`, `num_sentences`, `sentiment` and every
  readability index. In the hit quarters the median earnings call has about 45% fewer
  words, sentiment about 25% higher and Flesch reading ease about 8 points lower than
  in the quarters before.
- **Until reprocessed:** exclude calls dated 2017-04-01 to 2019-12-31 from analyses of
  exposures and of the `documents` text measures, and say so.
- **To flag single calls instead:** an earnings call whose `risk` run has three or fewer
  "question(s)" matches and whose `num_words / num_sentences` is above 22. The rule
  flags about 1% of 2015–2016 calls and 40–47% of 2017–2018 calls.

## Word counts and exposures use different text

The stored `documents.num_words`, `num_sentences` and readability come from the whole
XML body (header and participant lists included). Exposure counts come from the
Presentation and Q&A only. A ratio such as matches per word mixes the two. Use
`exposure_runs.total_sentences_in_document` as the denominator for exposures.

On a sample of runs the gap is small: the last match sits within about 1% of
`num_words` in the median run, because both paths start from nearly the same text and
share the Q&A damage above. The advice on the denominator stands.

Since 2026-11-11 `research ingest` measures the same spoken text that matching reads,
so this goes away once `documents` is recomputed. On 703 sample transcripts from
2002–2024, the new code's word count is within 5% of the old code's for 61%, and more
than 20% higher for 17% (57% of the 2018 calls).

## Exposure runs come from four batches

`research backfill` on 2026-10-17 made only 261,425 of the 1,014,924 stored runs (all of
2021 and parts of other years). The rest date from 2025-11 (482 runs), 2026-01 (145,712)
and 2026-02 (607,305), made by older code and word lists.

- 482 `risk` runs for 2016 calls are cosine runs labelled `direct`. Their counts include
  cosine matches (`match_type = 'cosine'`, threshold 0.7).
- 6,551 `risk` runs used an older list that matched "word" and never "volatile".
- `keyword_matches.context` is cut at 2,000 characters in the January and February runs
  (about 164,000 rows). Counts are not affected.
- 19 runs for 2018 calls have their matches stored twice (837 extra rows).

Until every document is rerun, filter on `exposure_runs.created_at` when the batch
matters, and drop `match_type = 'cosine'` rows from direct counts.

## "question" dominates the risk keyword set

About 67% of stored `risk` matches are the words "question" or "questions" (analyst Q&A
boilerplate). Count risk exposure without them. `Risk_v2.csv` drops them, and the
Loughran–McDonald negative list no longer has "question" or "ill" (2026-11-01), but the
stored `risk` runs still use `Risk.csv`.

## Other noise in keyword matches

These figures are scaled up from a 1% sample of `keyword_matches`.

- **Safe-harbor boilerplate.** About 19% of `risk` matches other than "question(s)" sit
  in forward-looking-statement sentences: 62% of "uncertainties", 48% of "risks". The
  stored runs did not use the safe-harbor option.
- **Speaker labels.** Lines naming a speaker's firm and job title stay in the matched
  text, in every year. About 406,000 matches have the keyword inside one ("Management",
  "Limited", "Securities").
- **Activeness words that mean something else.** "adjusted" (72% "adjusted EBITDA" and
  the like), "balance" (60% "balance sheet"), "prepared" (55% "prepared remarks") and
  "limited" (49% a company name) make up about 5% of all `activeness` matches. The list
  has not been reviewed.
- **Risk words with the same problem.** "venture" is 78% "joint venture", "chance" 54%
  idiom, "bet" 37% "you bet".

Until the rerun, drop these keywords from counts, and for `risk` also drop matches whose
`context` is safe-harbor text. The rerun's keyword sets (`risk_2026_11`,
`activeness_2026_11` in `config/keyword_sets.toml`) use `Risk_v2.csv`, exclusion lists for
terms of art and the safe-harbor filter; the current parser also drops almost all speaker
labels (11 of 704 sample calls keep a few).

## Sentiment sums grow with call length

`lm_score` and `ml_score` are sums over the call, not rates, so they grow with its length.
To compare calls, use `lm_per_1000_words` and `ml_per_1000_words` (added 2026-11-11), the
same scores per 1,000 words, which correlate with length at about −0.07. Postgres computes
them on read from the sums and `num_words`, so they always agree with the stored columns.

They inherit the word-count problem above: for calls that lost their Q&A, both the sum
and `num_words` come from the surviving text only.

## Stored LM and ML scores use the old dictionary method

Until the rerun, `lm_score` and `ml_score` come from code that has since been replaced:

- `lm_score` used the pysentiment2 package. It stems every word before looking it up, so
  "quite" counted as negative "quit" and "press release" as negative "press"; it dropped
  "good" and "strong" because they are on its list of surnames; and its own copy of the LM
  dictionary still lists "question", which alone was 22% of negative hits in a sample of
  calls. The new code matches the repo's LM lists exactly. On 703 sample calls it halves
  the negative words per call (median 68 to 35) and changes the sign of 147 calls' score.
- `ml_score` scored +1 only when two neighbouring words were both on the Garcia unigram
  list, so a single listed word counted for nothing. The paper counts each listed word.
  On the same calls the median rises from 1.4 to 20.4 per 1,000 words.

`hiv4_score` (Harvard IV-4) is no longer computed, and its two columns are dropped at the
rerun by `db/migrations/20261111_2000_drop_documents_hiv4.sql`; until then they hold the
old values for existing calls and NULL for new ones. It stemmed words too: nearly half
its hits in calls were not the dictionary word itself ("will" scored positive through
"willing", "margin" negative through "marginal"), and the Harvard lists treat words such as
"cost" and "capital" as negative. Analyses that used it need another tone measure.

## Sentiment of short documents depends on the load batch

Calls from 2015, 2017, 2019 and 2021 were loaded later than the other years, and their
short documents have `sentiment` of exactly 0 far more often: 21.6% of 2021 documents
with 30–100 sentences, against at most 1.1% in the years loaded earlier. The dictionary
scores do not differ, so the sentiment model ran differently. The cause is not known,
and the database does not record the model version.

Stored rows also have 0 for every score of a transcript with no spoken text, so an
empty document and a neutral one look the same. Since 2026-11-11 `ingest` writes NULL
for those; the stored rows change at the rerun.

Do not compare the `sentiment` of short documents across years.

## Stub and junk transcripts

7,348 documents are not usable text; 2,951 of them have a returns row. Most are
untranslated foreign-language calls from 2006–2014 (a language tag in the title, a
median of 183 words, sentiment 0, exposures stored as real zeros), notices that are not
calls, and 861 parses that collapsed to 5–12 sentences.

Filter out documents with fewer than 500 words, more than 60 words per sentence, a
Flesch-Kincaid grade above 30, or a Flesch reading ease below 0.

## Linsear Write and `text_standard` measure the header

`linsear_write_formula` reads only the first 100 words, which in the stored rows are the
vendor header and participant list. 31% of rows score above 30, and `text_standard`
partly inherits it. Do not use either column as stored. After a recompute the first 100
words will be the start of the Presentation, which is still not the whole call.

## Legacy columns in `documents`

`rc_active_ratio`, `rc_passive_ratio`, `re_active_ratio`, `re_passive_ratio`,
`rx_active_ratio`, `rx_passive_ratio` and `risk_total_sentences` came over from an older
database. No code writes or reads them, and they are empty for 2015, 2017, 2019 and
2021. Do not use them.

## A company row is not a firm

In mid-2020 the vendor changed from bare tickers (`F`) to RICs with an exchange suffix
(`F.N`), which made a second `companies` row for the same firm.

- **Rows:** 2,723 firms split across 5,497 company rows, holding 226,205 documents
  (44.6%). Another 203 firms are split by a change of exchange suffix.
- **Effect:** anything keyed on `company_id` (firm fixed effects, lags, changes within a
  firm) breaks in June 2020.
- **Until a firm key exists:** for linked calls use the PERMNO from
  `document_crsp_links`, which joins the two rows for 2,695 of 2,736 U.S. pairs. A bare
  ticker alone is not a key: 247 tickers carry more than one company name over time.

About 840 company rows are not listed firms at all (governments and public bodies,
vendor pseudo-tickers, a vendor test row, rows with no ticker), holding about 13,000
documents.

## "Earnings call" is a title guess

The event type in each XML is parsed but not stored. `is_earnings_call` in
`document_event_returns` is `title ILIKE '%earnings%'`.

- It matches 347,816 documents, 229,912 of them linked. About 29% of all documents are
  conferences, investor days, shareholder meetings and other events.
- It misses about 5,700 results calls whose title lacks the word "earnings" (about 1,000
  linked, mostly 2002–2012).
- It includes about 835 linked guidance, pre-announcement and merger calls.

## Calls end in 2021

Calls in `documents` run from 2002-01-14 to 2021-12-31. The raw files go on to 2024, and
since 2026-11-11 all of them are in `transcript_files`. 109,908 calls there have no
`documents` row:

- about 108,000 whose files sit only in the 2022–2024 folders. Those folders hold
  122,815 files, but many are later revisions of older calls, which is also why the
  2021 folder has more files (41,975) than 2021 has documents (39,130);
- 1,978 older published transcripts that were never ingested: 709 in the 2018 folder,
  mostly dated December 2018 and January 2019, 586 in 2011 and 250 in 2020. This is
  why December 2018 has about half the calls of the Decembers either side.

None of these calls has text measures, exposures, links or returns. January to March
2002 is thin in the vendor files themselves.

## Links and returns

- Calls whose company row has no U.S. listing (about 138,500) or no CRSP match have no
  `document_crsp_links` row, so no returns.
- **Wrong links.** The 311,296 ticker-and-date links sampled clean. The errors sit in
  small groups: 1,068 title links made on a name prefix (about half wrong), 899 links
  noted `title_unnamed` (about 15% wrong), and 426 ticker or CUSIP links whose title
  names a different listed firm (about a quarter wrong). Another 671 links have a call
  date outside the PERMNO's CRSP listing dates. Until they are unlinked, drop these
  groups or check them by hand.
- **`title_score` cannot filter them.** It is fixed at 1.0 on all 14,427 route A title
  links and NULL on all 1,399 route B links.
- **Several calls on one event.** 3,020 (PERMNO, day 0) pairs have more than one row
  (6,179 rows), and 316 of them hold two or more earnings calls. Keep one row per event
  before an event study.
- `mcap_pre` is missing for 1,737 of 352,115 linked calls (no CRSP price in the 30 days
  before day 0).
- Use `bhar_`, not `car_mm_`. For drift windows, subtracting the market-model alpha
  flips the sign of long-run drift. For short windows, one jump day in the estimation
  window can make alpha several percent a day: it is above 1% a day in 981 rows and
  above 0.5% in 9,450.
- **Partial windows.** A window gets a return if it has any one trading day. 5,277 rows
  have a short [+2, +121] window, and 21 report `ret_p0_p1` with no day-0 return. Filter
  on the `n_<window>` columns.
- 100 events have a delisting in the window with no CRSP delisting return, 41 of them
  for performance, so that loss is missing from the return.
- The parquet cache in `research returns surprise` is reused whenever its files exist,
  whatever the call set or date range. Delete the cache folder after changing either.
