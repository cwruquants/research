-- Applied once on 2026-10-17: clean the 697 junk company rows (padded names, junk
-- tickers, duplicates, the empty "None" row). The audit table company_cleanup_20261017
-- records every change. Statements are as run, in order, from the DB write log, except
-- that the reviewed CRSP ticker matches, inline when it ran, are now read from a private
-- CSV. The CREATE TABLE fails once the audit table exists; every later step is guarded
-- on the reviewed row counts, so it changes nothing on a re-run.
-- Backups: data/private/db_fixes/company_cleanup_2026-10-17_*.csv
--
-- The \copy path is relative, so run it from the folder holding the CSV:
--   cd data/private/db_fixes
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f ../../../db/fixes/2026-10-17_company_cleanup.sql

-- [2026-10-17T12:21]
-- Change list for the 2026-10-17 junk company cleanup (audit record).
-- One row per junk company (excluding the "None" row): its cleaned name, the
-- ticker it should get, and the row its documents end up on.
-- The reviewed matches of a company's own name to CRSP stocknames at its call dates
-- are CRSP-derived, so they stay private in 2026-10-17_company_tickers.csv.
CREATE TEMP TABLE crsp (company_id integer, ticker text);
\copy crsp FROM '2026-10-17_company_tickers.csv' WITH (FORMAT csv, HEADER true)
CREATE TABLE public.company_cleanup_20261017 AS
WITH c AS (
	SELECT
		id, name, ticker,
		btrim(name) AS clean_name,
		-- same rule as research.transcripts.parse.clean_ticker()
		CASE
			WHEN ticker IS NULL THEN NULL
			WHEN regexp_replace(ticker, '\\[nrt]|\s', '', 'g') = '' THEN NULL
			ELSE btrim(ticker)
		END AS clean_ticker
	FROM public.companies
),
junk AS (
	SELECT * FROM c
	WHERE (ticker IS DISTINCT FROM clean_ticker OR name <> clean_name)
		AND clean_name NOT IN ('None', '')
),
sib AS (
	-- junk rows whose name exists elsewhere with exactly one real ticker
	SELECT j.id, min(c2.clean_ticker) AS ticker
	FROM junk j
	JOIN c c2 ON c2.clean_name = j.clean_name AND c2.clean_ticker IS NOT NULL AND c2.id <> j.id
	WHERE j.clean_ticker IS NULL
	GROUP BY j.id
	HAVING count(DISTINCT c2.clean_ticker) = 1
),
target AS (
	SELECT
		j.id, j.name AS old_name, j.ticker AS old_ticker, j.clean_name AS new_name,
		coalesce(j.clean_ticker, sib.ticker, crsp.ticker) AS new_ticker,
		CASE
			WHEN j.clean_ticker IS NOT NULL THEN 'kept'
			WHEN sib.ticker IS NOT NULL THEN 'same-name row'
			WHEN crsp.ticker IS NOT NULL THEN 'crsp'
		END AS ticker_source
	FROM junk j
	LEFT JOIN sib ON sib.id = j.id
	LEFT JOIN crsp ON crsp.company_id = j.id
),
keeper AS (
	SELECT
		t.*,
		coalesce(
			-- an existing clean row with the target name and ticker wins
			(SELECT min(c3.id) FROM c c3
				WHERE c3.name = t.new_name AND c3.ticker IS NOT DISTINCT FROM t.new_ticker
					AND c3.id NOT IN (SELECT id FROM junk)),
			min(t.id) OVER (PARTITION BY t.new_name, t.new_ticker)
		) AS keeper_id
	FROM target t
)
SELECT
	k.id AS company_id,
	k.old_name, k.old_ticker, k.new_name, k.new_ticker, k.ticker_source,
	k.keeper_id,
	CASE WHEN k.id = k.keeper_id THEN 'update' ELSE 'merge' END AS action,
	(SELECT count(*) FROM public.documents d WHERE d.company_id = k.id) AS documents
FROM keeper k
ORDER BY k.id;

-- [2026-10-17T12:21]
-- Step 1: move documents from duplicate junk rows onto their keeper row.
-- Changes nothing unless exactly the reviewed 372 documents are still on those rows.
UPDATE public.documents d
SET company_id = m.keeper_id
FROM public.company_cleanup_20261017 m
WHERE d.company_id = m.company_id
	AND m.action = 'merge'
	AND (
		SELECT count(*)
		FROM public.documents d2
		JOIN public.company_cleanup_20261017 m2 ON m2.company_id = d2.company_id AND m2.action = 'merge'
	) = 372;

-- [2026-10-17T12:22]
-- Step 2: delete the 107 duplicate junk rows, now holding no documents.
-- The NOT EXISTS keeps the ON DELETE CASCADE from ever touching a document,
-- and the count check makes it a no-op if anything drifted.
DELETE FROM public.companies co
USING public.company_cleanup_20261017 m
WHERE co.id = m.company_id
	AND m.action = 'merge'
	AND co.name = m.old_name
	AND co.ticker IS NOT DISTINCT FROM m.old_ticker
	AND NOT EXISTS (SELECT 1 FROM public.documents d WHERE d.company_id = co.id)
	AND (
		SELECT count(*)
		FROM public.companies co2
		JOIN public.company_cleanup_20261017 m2 ON m2.company_id = co2.id AND m2.action = 'merge'
		WHERE NOT EXISTS (SELECT 1 FROM public.documents d2 WHERE d2.company_id = co2.id)
	) = 107;

-- [2026-10-17T12:25]
-- Step 3: trim names and replace junk tickers on the 589 remaining rows
-- (CRSP or same-name ticker where found, NULL otherwise).
-- Only touches rows still holding their reviewed old values; no-op unless all 589 do.
UPDATE public.companies co
SET name = m.new_name,
	ticker = m.new_ticker
FROM public.company_cleanup_20261017 m
WHERE co.id = m.company_id
	AND m.action = 'update'
	AND co.name = m.old_name
	AND co.ticker IS NOT DISTINCT FROM m.old_ticker
	AND (
		SELECT count(*)
		FROM public.companies co2
		JOIN public.company_cleanup_20261017 m2 ON m2.company_id = co2.id AND m2.action = 'update'
		WHERE co2.name = m2.old_name AND co2.ticker IS NOT DISTINCT FROM m2.old_ticker
	) = 589;

-- [2026-10-17T12:26]
-- Step 4a: delete the 156 empty documents filed under the company named "None"
-- (0 words, no title, no date). Cascades to their 4 exposure runs and 207 keyword matches,
-- all backed up in data/private/db_fixes. No-op unless all 156 are still empty and still there.
DELETE FROM public.documents d
WHERE d.company_id = 17338
	AND coalesce(d.num_words, 0) = 0
	AND d.event_title IS NULL
	AND d.event_date IS NULL
	AND (SELECT name FROM public.companies WHERE id = 17338) = 'None'
	AND (
		SELECT count(*) FROM public.documents d2
		WHERE d2.company_id = 17338
			AND coalesce(d2.num_words, 0) = 0 AND d2.event_title IS NULL AND d2.event_date IS NULL
	) = 156
	AND (SELECT count(*) FROM public.documents d3 WHERE d3.company_id = 17338) = 156;

-- [2026-10-17T12:32]
-- Step 4b: delete the now-empty company row named "None".
DELETE FROM public.companies co
WHERE co.id = 17338
	AND co.name = 'None'
	AND co.ticker IS NULL
	AND NOT EXISTS (SELECT 1 FROM public.documents d WHERE d.company_id = co.id);
