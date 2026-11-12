# Vocabulary

Terms as this repo and the `research` database use them.

## Transcripts and companies

- **Transcript / call / document.** One vendor XML file (`<id>_T.xml`) is one call, stored
  as one row of `documents`. Most are earnings calls (`documents.event_title` contains
  "Earnings", about 230,000). The rest are conferences, investor days and M&A calls.
- **Revision.** The vendor re-issues a call under the same file name with a later
  `lastUpdate` stamp, usually to change the header, the ticker or a few words. Each
  distinct file is a row of `transcript_files`; the latest published one is the call's
  transcript (`transcript_files_latest`).
- **Withdrawn call.** A call whose latest file is an empty stub with `action="delete"`.
- **Company row.** A row of `companies`, keyed by name and ticker as the vendor files the
  call. The vendor relabels old calls with the company's *current* owner, so a company row
  can hold calls about other firms (TPG Capital rows hold Midwest Express calls). Link each
  call by its own date and title, not by its company row.
- **Ticker / RIC.** `companies.ticker` is a Refinitiv RIC: root, optional share class,
  exchange suffix and a `^` delisting code (`AAPL.OQ`, `OCR^H15`).
- **Presentation / Q&A.** The usual two spoken sections of a transcript. Some calls head
  them differently ("Transcript", "q and a") or repeat them. Keyword matching, word
  counts, readability and sentiment all read every section except the participant lists,
  with speaker lines removed (`spoken_text`). The stored word counts and readability still
  come from the whole XML body.

## Word lists and matching

- **Word list / word set.** A CSV of keywords and phrases in `data/word_sets/`.
- **Keyword set.** A word list that has been run against the corpus and stored in the
  database (`keyword_sets`; listed in `config/keyword_sets.toml`). Today: `risk`
  (`Risk.csv`) and `activeness`.
- **Direct match.** An exact, case-insensitive hit of a keyword (one word) or phrase
  (consecutive words). Optional: **normalized phrases** also match across stopwords and
  plurals; **exclusions** are terms of art whose words never count ("risk management");
  the **safe-harbor** rule skips forward-looking-statement boilerplate and the two
  sentences either side.
- **Cosine match.** A document word whose MiniLM embedding is close to a keyword's
  (threshold 0.7). Not stored in the database.
- **Exposure run.** One (document, keyword set) result: `exposure_runs` holds the counts,
  `keyword_matches` one row per hit with the matched sentence as `context`.

## Linking

- **PERMNO / PERMCO / NCUSIP.** CRSP's security id, company id and the CUSIP in effect on a date.
- **Route A.** Link a call by its company row's RIC on the call date (ticker, then LSEG
  CUSIP, then the ticker's later owner), checked against the CRSP name the title names.
- **Route B.** Link a call by its event title alone, for company rows named after an
  acquirer or a conference host.
- **Call date.** The call's date in New York time (`call_date_ny`).

## Returns

- **Day 0.** The first trading session that can react to the call: the call's own date,
  or the next trading day if the call starts at or after the 4pm New York close.
- **Windows.** Trading-day offsets from day 0, named by their ends: `m20_m2` = [-20,-2]
  (run-up), `p0_p0`, `p0_p1`, `m1_p1`, `m1_p2`, `p0_p2` (reaction), `p2_p21` … `p2_p121` (drift).
- **ret / bhar / car_mm.** Buy-and-hold raw return; that minus the CRSP value-weighted
  index over the same days; market-model cumulative abnormal return with alpha and beta
  estimated on days [-250,-30]. `n_<window>` counts the days used. Prefer BHAR for drift
  windows.
- **mcap_pre.** Market cap in USD millions on the last trading day at least two calendar
  days before day 0.
- **SUE.** Standardized unexpected earnings: (actual EPS − median consensus) / price,
  from I/B/E/S. `sue_fresh` uses analysts' latest forecasts in the 90 days before the
  announcement.
