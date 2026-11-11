# Data audit of the research database, 2026-11-11

This is a read-only audit of every table in the `research` database, made to plan a data
cleanup. Nothing in the database was changed. Five audits ran in parallel, one per area
(companies and document metadata, document text measures, exposures and keyword matches,
CRSP links, event returns), and the headline counts below were then re-run by hand
against the live tables.

How to read it:

- **Verified** means example rows or the code were checked and the problem is real.
  **Suspected** means the pattern is clear but the cause or the exact rows are not
  confirmed.
- Counts are exact unless marked "about" with a sample size. Counts inside
  `keyword_matches` are scaled up from a 1% sample (506,541 rows) unless stated.
- Row counts on the audit date: 18,947 companies, 507,462 documents, 1,014,924 exposure
  runs, 50,655,891 keyword matches, 352,115 links and 352,115 returns rows.
- Example ids, company names and the scripts behind the link and returns checks are in
  `data/private/audits/data_audit_2026-11-11_examples.md`, which is not in git because
  it names firms from the licensed data.
- Problems already in [known_issues.md](known_issues.md) are not repeated, except where
  the audit found them larger than documented.

## Summary

Ranked by how much each could distort a result, not by row count.

| # | Problem | Size | Status | Section |
|---|---|---|---|---|
| 1 | The tight-layout parser bug also damaged `documents` word counts, readability and sentiment, and `research ingest` still has the bug | about a third of earnings calls from 2017-04 to 2019-03, fewer through 2021 | Verified | Text measures |
| 2 | `company_id` is not a firm id: a vendor ticker change in mid-2020 split most firms into two company rows | 2,723 firms, 226,205 documents (44.6%) | Verified | Companies |
| 3 | The event type in the XML is never stored, so "earnings call" is a title guess | 29% of documents are not earnings calls; about 5,700 results calls lack the word "earnings" | Verified | Companies |
| 4 | Stored keyword matches include speaker labels, safe-harbor boilerplate and untested activeness words | about 0.4M label matches; 19% of risk matches other than "question(s)"; 5% of activeness matches from four phrases | Verified on a 1% sample | Exposures |
| 5 | Wrong CRSP links cluster in three small link groups, and `title_score` cannot find them | roughly 750 wrong links, plus 671 links to firms not listed on the call date | Verified on samples | Links |
| 6 | Several transcripts share one (PERMNO, day 0) event | 3,020 events, 6,179 rows; 316 events with two or more earnings calls | Verified | Returns |
| 7 | Stub and junk transcripts carry meaningless text measures and zero exposures | 7,348 documents, 2,951 of them with returns | Verified | Text measures |
| 8 | Exposure runs were made in four batches by different code and word lists | only 261,425 of 1,014,924 runs are from the October backfill | Verified | Exposures |
| 9 | Sentiment differs by load batch for short documents | the 2015, 2017, 2019 and 2021 loads | Suspected (cause unknown) | Text measures |
| 10 | `linsear_write_formula` and `text_standard` measure the transcript header | 31% of rows out of range | Verified | Text measures |
| 11 | Market-model abnormal returns are thrown off by extreme alphas, and partial windows are reported without a minimum length | 981 rows badly, 9,450 mildly; up to 5,277 short windows | Verified | Returns |
| 12 | Non-corporate and junk company rows | about 840 company rows, about 13,000 documents | Verified | Companies |

The returns arithmetic itself is sound: 300 random rows recomputed from the CRSP files
matched to 14 decimal places.

## Text measures in `documents`

### The tight-layout bug is in `documents` too, and in the current ingest code

`known_issues.md` says the lost-Q&A bug touched only `exposure_runs` and
`keyword_matches`. It also damaged `documents.num_words`, `num_sentences`, every
readability column and `sentiment`.

The cause is the same line skip. `load_document` was fixed on 2026-10-31 to clean the
Presentation and Q&A separately. `analyze_document`
(`src/research/pipelines/ingest.py`) still calls
`clean_spoken_content(extract_full_body_text(path))` on the whole body, so its fixed
three-line skip after each rule still drifts in tight-layout files. Re-running
`research ingest` today would write the same wrong values.

Medians for documents titled "earnings call", by quarter:

| Quarter | Words | Words per sentence | Flesch reading ease | Sentiment | Under 3,000 words |
|---|---|---|---|---|---|
| 2016Q1 to 2017Q1 | 6,460–7,130 | 18.3–18.7 | 56.7–57.1 | 0.039–0.041 | 6–8% |
| 2017Q2 to 2018Q4 | 3,560–4,060 | 22.3–23.1 | 48.1–49.1 | 0.046–0.053 | 25–35% |
| 2019Q1 | 4,142 | 22.4 | 49.2 | 0.052 | 24% |
| 2019Q2 to 2019Q4 | 4,870–5,630 | 18.9–19.6 | 54.6–55.7 | 0.041–0.042 | 15–21% |
| 2020 | 6,190–6,630 | 18.4–18.8 | 55.4–56.4 | 0.028–0.037 | 8–11% |

So in the hit period stored sentiment is about 25% too high, Flesch about 8 points too
low and word counts about 45% too low. The bias is in the level of the measure, not just
in a denominator.

This also changes the "different text" item in `known_issues.md`. In practice
`documents.num_words` tracks the matched text closely (the last match position is within
about 1% of it in the median run), because both paths start from nearly the same text
and share the same damage. (Suspected: checked on a sample of runs, not on raw files.)

Cleanup:

1. Make `analyze_document` clean each section on its own, as `load_document` now does.
2. Recompute every text measure in `documents`, for all years, in one run. An earlier
   scan of the raw files found about 17% of all calls change by more than 20%,
   including 18–26% a year in 2002–2008.
3. Until then, extend the 2017-04-01 to 2019-12-31 exclusion in `known_issues.md` to the
   `documents` columns.

### Stub and junk transcripts

7,348 documents meet at least one of the rules below; 3,856 have "earnings" in the title
and 2,951 have a returns row.

| Rule | Documents |
|---|---|
| Fewer than 500 words | 6,074 |
| More than 60 words per sentence | 1,402 |
| Flesch-Kincaid grade above 30 or Flesch reading ease below 0 | 3,779 |
| Foreign-language tag in the title and fewer than 2,000 words | 3,483 |

Three kinds of document are behind them:

- **Untranslated foreign-language calls, 2006–2014.** Titles tagged "(Japanese)",
  "(Chinese)" and so on. 3,395 of the 3,989 tagged calls have fewer than 500 words
  (median 183) and sentiment of exactly 0. Their exposure runs are stored as real zeros:
  of 3,440 Japanese-tagged calls, 3,425 have three sentences or fewer and 3,437 have no
  risk match. From 2015 the vendor supplies full text.
- **Items that are not calls**, such as regulatory notices of a few dozen words.
- **Collapsed parses.** 861 documents have a normal word count but 5 to 12 sentences,
  while their exposure run counted at least 30 sentences and more than twice as many
  (median 12 against 104). 475 are earnings calls and 666 have returns. This is the
  signature the October recompute fixed for 1,254 old-layout transcripts; none of the
  861 were in that set.

The October recompute also left `exposure_runs` untouched: for 141 of its 1,254
documents the exposure run still has under 80% of the recomputed sentence count.

Cleanup: after the recompute above, add a flag column (for example
`documents.text_quality`) set by these rules, and filter on it. Short real calls exist,
so 1,000 to 2,000 words (20,846 documents) should stay.

### Sentiment depends on the load batch for short documents

The rows loaded on 2026-02-12, 02-13, 03-02 and 03-03 (the 2015, 2017, 2019 and 2021
calls) give short documents a sentiment of exactly 0 far more often than the rows
loaded on 2026-02-08 to 02-11 (every other year).

| Check | Earlier loads | Later loads |
|---|---|---|
| Documents of 30–100 sentences with sentiment exactly 0, by year | 0.0–1.1% | 0.9% (2015), 1.5% (2017), 2.8% (2019), 21.6% (2021) |
| Shareholder meetings of 300–2,000 words, mean sentiment | 0.033–0.040 | 0.004–0.020 |
| Same meetings, sentiment exactly 0 | 14 of 1,564 | 617 of 1,625 |

The dictionary scores (`hiv4_score` per sentence) are the same across the two groups, so
the documents are alike and the sentiment model ran differently. The model version,
truncation setting or paragraph split may have differed; the database does not record
them. The later loads are also the rows with no legacy columns (see below), which points
to a different code version.

A second problem makes this harder to see: `analyze_document` writes `0.0` when the
model returns nothing, so a failed score and a neutral document look the same.

Cleanup: recompute sentiment for all rows in the same run as the parser fix, with the
model and package versions pinned and written to the database, and store NULL on
failure. Until then, do not compare the sentiment of short documents across years.

### Linsear Write and `text_standard` measure the header

`textstat.linsear_write_formula` reads only the first 100 words, which here are the
vendor header and participant list. 159,506 documents (31%) score above 30, the
correlation with Flesch-Kincaid grade on clean documents is 0.009 (Gunning Fog and ARI
correlate with it at 0.987), and the yearly median moves with the vendor's header
layout. `text_standard` partly inherits it: 4,278 rows are rated grade 20 or higher.

Cleanup: drop both columns, or recompute them on the spoken text with
`strict_upper=False`. Do not use them as stored.

### Legacy columns

`rc_active_ratio`, `rc_passive_ratio`, `re_active_ratio`, `re_passive_ratio`,
`rx_active_ratio`, `rx_passive_ratio` and `risk_total_sentences` came over from Supabase.
No code in any branch writes or reads them. They are filled on 371,279 rows (73%) and
empty for 2015, 2017, 2019 and 2021. `risk_total_sentences` is a copy of the old risk
run's `total_sentences_with_match`. Drop them in a migration, or document them as
frozen.

## Exposures and keyword matches

Structure is clean: every document has exactly one run per keyword set, there are no
orphans, the stored counts equal the stored rows apart from the cases below, and the two
runs of a document always agree on its sentence count.

### Runs come from four batches, not one

`pipeline.md` says the exposures were last filled by `research backfill` on 2026-10-17.
Only a quarter were.

| Computed | Runs | Calls covered |
|---|---|---|
| 2025-11 | 482 (risk only) | 482 calls from 2016 |
| 2026-01 | 145,712 | 2002–2006 |
| 2026-02 | 607,305 | 2005, 2007–2020 |
| 2026-10 | 261,425 | all of 2021, and parts of every other year |

For calls dated 2017-04 to 2019-12, about 31,460 documents still have February runs and
about 54,580 have October runs, so the known Q&A problem spans both batches. The batches differ in
ways that matter:

- **482 risk runs are cosine runs labelled `direct`.** They carry `cosine_threshold =
  0.7` and 15,620 `match_type = 'cosine'` rows, and their sentence and keyword counts
  include the cosine hits. 58% of those hits are "if" matched to "iffy".
- **A stale risk list.** In 6,551 runs (the first 2002 batch and the 482 above) the
  keyword "word" was matched and "volatile" never was: 1,433 "word" rows and no
  "volatile" rows, while every other batch matches "volatile" and never "word". "word"
  is in no version of `Risk.csv` in git.
- **Contexts cut at 2,000 characters** in the January and February runs: about 164,000
  rows. October runs keep the full sentence. Counts are unaffected, but anything that
  reads `context` is.
- **19 runs had their matches inserted twice**, 837 extra rows in all, the second copy
  with positions shifted by 3 so the unique constraint did not catch it. All are for
  2018 calls, from one February batch.

Cleanup: rerunning `research backfill` for every document after the parser fix replaces
all four batches and removes each of these. Record the code and word-list version on
`exposure_runs` when it does.

### What the matches contain

All figures here are from the 1% sample.

- **Speaker labels.** Lines such as "Name, Firm - Title [3]" stay in the matched text,
  so words in firm names and job titles count: "Management", "Limited", "Strategy",
  "Securities". About 406,000 matches (0.8% of all, 1.0% of activeness) have the keyword
  inside a label. This is in every year, not only the tight-layout period: a third of
  sampled 2015 earnings calls have a label in some context.
- **Safe-harbor boilerplate.** 19.3% of risk matches other than "question(s)" sit in a
  sentence that the code's own safe-harbor pattern would skip, about 676,000 rows: 62%
  of "uncertainties", 48% of "risks", 6% of "risk". The stored runs never used the
  safe-harbor option. So a risk count cleaned only of "question(s)" is still about a
  fifth legal boilerplate.
- **Activeness words that are mostly something else.** Nobody has reviewed this list.

  | Keyword | Share that is noise | What it is |
  |---|---|---|
  | adjusted | 72% | "adjusted EBITDA", "adjusted EPS" |
  | balance | 60% | "balance sheet" |
  | prepared | 55% | "prepared remarks" |
  | limited | 49% | "Limited" in a company name |
  | management | 24% | "management team", "risk management" |
  | report, reported | 18–21% | "as reported", "annual report" |
  | discuss, review | 12–17% | "we will discuss our results" |

  The first four alone are 5.1% of all activeness matches.
- **Risk words with the same problem:** "venture" is 78% "joint venture", "chance" is
  54% idiom ("had a chance to"), "bet" is 37% "you bet".
- HTML entities such as `&amp;` remain in 2.8% of contexts. They never match a keyword.

Cleanup: add an exclusions file for activeness and extend the one for risk, turn on the
safe-harbor rule, and strip speaker labels in the parser, all before the rerun.

### A per-document flag for lost Q&A

The date-range exclusion in `known_issues.md` is coarse. A rule that flags single calls:
an earnings call whose risk run has three or fewer "question(s)" matches and whose
`num_words / num_sentences` is above 22. It flags 0.8% of 2015 calls and 1.5% of 2016
calls, which bounds its false-positive rate.

Flag rate on a 1.5% sample of earnings calls (5,332 calls): 40% in 2017, 47% in 2018,
20% in 2019, 6% in 2020 and 2% in 2021. It is also 4–11% in each year from 2002 to 2014
(11% in 2004 and 2008), which is more than the "scattered older calls" in
`known_issues.md`. Some of those may be calls that never had a Q&A. (Suspected: only
re-parsing the files with the fixed parser settles it.)

## Companies and document metadata

### `company_id` is not a firm id

Until May 2020 about half of all calls sit on company rows with a bare ticker (`F`).
From July 2020 the vendor files the same firms under a RIC with an exchange suffix
(`F.N`), which made a second company row. Bare-ticker documents per month fell from
2,694 in May 2020 to 716, 169, 109 and then 1 in September.

| Group of rows that are one firm | Groups | Company rows | Documents |
|---|---|---|---|
| Bare ticker and suffixed RIC, same name | 2,723 | 5,497 | 226,205 |
| Same name, different exchange suffix (`.VX` to `.S`) | 203 | 406 | 6,877 |
| Delisted (`^`) and active RIC | 72 | not counted | not counted |
| Same ticker, name differs only in case or punctuation | 43 | 86 | not counted |

Any panel keyed on `company_id` (firm fixed effects, lags, changes within a firm) breaks
in June 2020. The count of companies with calls shows it: 7,229 in 2019, 11,473 in 2020
and 8,381 in 2021. The CRSP link mostly repairs this for linked calls: of 2,736 U.S.
pairs, 2,695 share a PERMNO.

Bare tickers are also reused by different firms over time (247 tickers carry more than
one name), so the ticker alone cannot be the key either.

Cleanup: add a firm key (a `firm_id` column or a grouping table) and keep the vendor's
rows as they are, since the RIC is real vendor data. Use PERMCO as the firm for linked
calls. Say in `CONTEXT.md` that a company row is not a firm.

### The event type is dropped at ingest

The parser reads `eventTypeId` and `eventTypeName` from each XML and `analyze_document`
passes them on, but `documents` has no column for them. The only way to tell an earnings
call from a conference is the title.

By title, about 71% of documents are earnings or results calls, 23% conferences,
investor days and presentations, 2% shareholder meetings, 1% M&A calls, 1% government or
policy events, and 3% other.

`is_earnings_call` in `document_event_returns` is `title ILIKE '%earnings%'`. That rule:

- matches 347,816 documents (68.5%), of which 229,912 are linked. The "about 230,000"
  in `CONTEXT.md` is the linked count, not the corpus count;
- misses about 5,700 results calls with titles such as "Financial Results Conference
  Call" (about 1,000 of them linked, mostly 2002–2012);
- includes about 835 linked guidance, pre-announcement and merger calls.

Cleanup: add `event_type_id` and `event_type` columns in a migration, backfill them from
the XMLs, and define earnings calls from them.

### Non-corporate and junk company rows

| Kind | Company rows | Documents | Note |
|---|---|---|---|
| Sovereigns and public bodies (ticker like `US***`, `**ECB`) | about 37 | about 6,300 | one U.S. government row holds 6,065 hearings and briefings; 192 documents are linked to CRSP, all wrongly |
| Vendor pseudo-tickers (`**NAME`, `1ESP_..._*` and similar) | about 310 | about 3,500 | private issuers and vendor codes, not RICs |
| Vendor test and catch-all row | 1 | 112 | titles are bare tickers, "Test Event", "UPLOAD FOR ..." |
| Rows the vendor marks "(Duplicate)" | 8 | about 100 | |
| No ticker | 462 | 3,237 | 17 have a same-name row with a ticker that the October cleanup rule skipped |
| No documents | 20 | 0 | shells created 2026-02-11 to 02-13 |

Cleanup: add a `ticker_kind` column (RIC, bare, pseudo, sovereign) and keep pseudo and
sovereign rows out of linking; merge the 43 case-and-punctuation groups the way the
October cleanup merged rows; delete the 20 empty rows; decide whether the test row's
calls duplicate real ones before deleting it.

### Duplicate transcripts

- 13 files named `<id>_T (1).xml` are browser-download copies of `<id>_T.xml`, with the
  same company, date, title and word count.
- 51 groups share company, timestamp and title. 26 have identical word counts; 8 differ
  widely and are a preliminary and a final version of one call.
- 208 pairs share company and day with word and sentence counts within 1%. Some are a
  renamed second copy, some are real morning and evening calls.

Cleanup: delete the 13 copies (the delete cascades to their exposure runs, matches,
links and returns); review the other roughly 100 candidates by hand and keep the longest
version.

### Smaller metadata problems

- **`city`** was never cleaned: 13,307 values are whitespace, 11,624 are a literal
  backslash-n, 251 are `-`, 99 are ZIP-like numbers, 22 hold HTML entities, 42% are in
  capitals, and New York is spelled a dozen ways. The current `clean_text` would null
  the blanks; the stored rows predate it.
- **`event_title`**: 3,311 have leading or trailing whitespace, 4,880 double spaces, 210
  HTML entities, 10 mojibake, and 6 are exactly 255 characters long, so possibly cut.
  About 21 name a fiscal year two or more years from the call date, mostly typos.
- **Date range.** Calls run from 2002-01-14 to 2021-12-31. `README.md` and `data.md`
  said 2002–2024 (corrected after the audit). The raw files do run to 2024: a later
  check of the source zip found 122,815 transcripts for 2022–2024 that were never
  loaded, and 41,975 files in the 2021 folder against 39,130 documents dated 2021.
  January to March 2002 is thin (52, 61 and 213 calls), and December 2018 is
  low (414 against 870 and 1,122 in the Decembers either side), which may be a partial
  gap.

## CRSP links

Of the 352,115 links, the 311,296 made by ticker and date are clean: 40 of 40 sampled at
full score were right. The table equals the linker's output file row for row, every link
has a returns row with the same PERMNO, and the stored NCUSIP, ticker and name match the
CRSP name record in effect on the call date.

### Wrong links

| Link group | Links | Wrong in a sample | Cause |
|---|---|---|---|
| Route A title match on a name *prefix* | 1,068 | 14 of 30 (47%); 418 confirmed by group review | the CRSP name, with words like HOLDINGS and GROUP stripped, only has to start the title, so a one-word name such as NATIONAL matches any title beginning with it |
| Ticker or CUSIP link whose title cleanly names a different listed firm | 426 flagged | about 25% of flagged; 93 confirmed | old calls relabeled under the later owner, and reused tickers |
| Links noted `title_unnamed` | 899 | 4 or 5 of 30 (about 15%) | a target's or presenter's call linked to the acquirer or the broker hosting it |
| Call date outside the PERMNO's CRSP listing dates | 671 | all | 596 are up to a year after delisting, 75 up to 30 days before listing; the firm is right but there is no stock, and all but 5 have no day-0 return |
| Subsidiary calls linked to the parent's stock | about 65 | judgement call | captive finance arms and joint ventures |

The worst single case is one PERMNO that receives 236 calls from 15 company rows, of
which 7 are right. 181 are U.S. government hearings.

Two code causes are worth fixing before any re-link:

- The title matcher treats month names and words such as "Corporate", "Annual" and
  "Strategic" as the end of the company name, so titles of firms named after a month or
  starting with "Corporate" read as naming no company.
- The other-name pass accepts any name the PERMNO ever had, including one adopted years
  after the call. 319 of 601 such links matched a later name.

`title_score` cannot be used to filter these. It is hard-coded to 1.0 on all 14,427
route A title links and NULL on all 1,399 route B links, which are the groups with the
errors. On ticker links a low score mostly means the title names no company.

Cleanup:

1. Unlink the prefix-match links whose stripped CRSP name is one word (756), review the
   other 312, and unlink the 671 out-of-span links or flag them.
2. Review the 426 flagged ticker links and the 899 `title_unnamed` links by hand.
3. Store the real name score and the match status (matched, prefix, other name, near
   date) in the table.
4. Fix the two matcher rules, then compare old and new links before reloading.

### Missed links

- **Name check failed (2,325 documents, 1,671 earnings calls).** A U.S. RIC found a
  PERMNO but the title did not pass the name check. 13 of 30 sampled should have been
  linked: abbreviated company names, or relabeled targets the title matcher missed.
- **No ticker (1,312 documents, 930 earnings calls)** include U.S. listed firms that a
  company-name match would find.
- **No CRSP match (13,155).** Mostly right: 5,834 are the U.S. government row, the rest
  private debt issuers and OTC names. About 90 are strong missed links.
- Route B's call list (1,457 calls on 73 company rows) still has no query that rebuilds
  it. Its sampled error rate is 0 to 2%.

## Event returns

### Several rows for one event

348,956 distinct (PERMNO, day 0) pairs carry 352,115 rows. 3,020 events have more than
one row (6,179 rows, at most 9).

| Kind | Events |
|---|---|
| Two or more earnings calls | 316 (633 rows, 0.28% of earnings-call rows) |
| An earnings call plus another event | 749 |
| No earnings call (several conferences in a day) | 1,955 |

The double earnings calls are split sessions and separate Q&A transcripts, a "Q4" and a
"Full Year" copy of one call, the same call filed under two company rows, exact
duplicates (115 same-day groups have an identical title), and wrong links (66 of the 316
mix two company rows).

Cleanup: add a `primary_event` flag or a view with one row per (PERMNO, day 0, earnings
call), preferring the full call over a Q&A-only part. Without it an event study counts
these days twice.

### Market-model returns and partial windows

- **Extreme alphas.** Alpha is the mean daily residual over [-250, -30], so one jump day
  in that window can make it several percent a day. |alpha| is above 1% a day in 981
  rows, which moves `car_mm_m1_p1` by more than 3 points, and above 0.5% in 9,450 rows.
  `known_issues.md` warns about `car_mm_` only for drift windows; short windows are hit
  too. 20 rows have |beta| above 5.
- **Partial windows.** A window with any one trading day gets a return. 5,277 rows have a
  short [+2, +121] window (1,538 with under 60 days), 536 a short run-up window, and 21
  rows report `ret_p0_p1` with no day-0 return, so the "reaction" is day +1 alone.
- **Missing delisting returns.** 100 events have a delisting in the window with no CRSP
  delisting return; 41 are performance delistings, where the loss is simply absent.
- **All-null rows.** 1,422 rows have every return null: 529 calls held after the stock
  left CRSP and 893 inside stretches where CRSP has no return.
- **`mcap_pre`.** 159 consecutive pairs for one PERMNO jump more than 10-fold, 40 within
  120 days. At least one is a one-off share-count error in CRSP. There is no
  thousands-for-millions unit error, and no value is zero or negative.

Cleanup: use `bhar_` everywhere, or refit with winsorized estimation returns; filter on
`n_<window>` (full length for reaction windows, at least 80% for run-up and drift);
decide on a fill for missing performance-delisting returns; winsorize log size.

## Schema and operations

- **Loose columns.** `documents.company_id`, `exposure_runs.document_id`,
  `exposure_runs.keyword_set_id` and `keyword_matches.exposure_run_id` allow NULL though
  none is NULL today. `documents.source_file` and `event_date` allow NULL too.
  `exposure_runs.run_timestamp` is text. `company_cleanup_20261017` has no primary key.
- **No stored provenance.** No table records the code version, model version or word
  list version behind a row. The sentiment batch effect and the stale risk list could
  not be traced for this reason.
- **Cascading deletes.** Every foreign key is `ON DELETE CASCADE` from `companies` down
  to `keyword_matches`. Deleting a company row deletes its documents, runs, matches,
  links and returns. That is what the cleanup deletes above want, but it also means a
  mistaken company delete is expensive.
- **Statistics counters are all zero** although the tables have column statistics. The
  server last started on 2026-11-11, and PostgreSQL discards these counters after an
  unclean shutdown, so the PC probably lost power or was forced off. (Suspected.) Run
  `ANALYZE` after the next bulk change.
- The live constraints and indexes match `db/tables.sql` and `db/constraints.sql`.

## Checks that came back clean

- No orphan or NULL `company_id`; no NULL, pre-2002 or future `event_date`; `year`
  agrees with the call date in both UTC and New York time, so the October year fix is
  complete; `source_file` is unique and well formed apart from the 13 copies.
- The October company cleanup held: no merged row has come back and no updated row has
  drifted.
- No negative counts, no word count below the sentence count, sentiment always within
  [-0.27, 0.65], and the readability indices other than Linsear move together as their
  formulas imply.
- Exposure runs: no duplicates or orphans; `pct_sentences_with_match` is always
  100 × matched sentences / sentences; `total_keywords_searched` equals the list length
  (139 risk, 334 activeness); every stored keyword is in its list except "word"; direct
  matches always equal their keyword; no empty contexts.
- Links and returns agree row for row. `call_day`, `call_timing` and `day0` recompute
  exactly from `event_date`, including across daylight-saving changes. Day 0 is never on
  a weekend and never before the call. The absolute return peaks on day 0 for every year
  and every call hour, so no period is misaligned.
- No return is at or below -100%. The largest returns checked are real events. No drift
  window is cut off by the end of the CRSP file.
- No share-class flipping within a company row.

## Not checked

- Raw XML files were not opened, apart from one call used to confirm the parser bug. So
  the audit cannot say which pre-2017 flagged calls truly lost their Q&A, whether
  December 2018 is missing files, or whether the test row's calls duplicate real ones.
- Stored distinct-keyword and sentence counts were compared with the match rows on a
  sample of about 3,000 runs and one id range, not on all 1,014,924 runs.
- Link error rates come from 30-row samples and group reviews by title and name, not
  from reading transcripts.
- CRSP prices and shares were not available locally, so the `mcap_pre` jumps were judged
  from returns.

## Suggested order of work

1. Fix `analyze_document`, strip speaker labels, and settle the exclusion lists and the
   safe-harbor rule. Then recompute all `documents` text measures and rerun
   `research backfill` for every document, in one pass with versions recorded. This
   clears rows 1, 4, 8 and 9 of the summary, most of 7 and 10, and the known Q&A issue.
2. Add the event-type columns and a firm key. Both are additive migrations.
3. Clean the links: unlink the confirmed wrong groups, flag out-of-span links, store the
   real score, then fix the matcher and compare.
4. Add the quality and primary-event flags, so analyses filter on columns and not on
   rules copied between scripts.
5. Do the small deletes and merges (copies, empty and duplicate company rows, legacy
   columns, `city`), each with a file in `db/fixes/` or `db/migrations/`.
6. Update `known_issues.md`, `pipeline.md`, `README.md`, `data.md` and `CONTEXT.md` with
   what changed.
