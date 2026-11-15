-- Applied once on 2026-11-15 with research_agent: 507,462 rows (no row rerun yet). The check
-- below gave the same count and fingerprint 916f92d97f336e226a16f03d09d063e3 for both tables.
--
-- Snapshot of the documents text measures before the 2026-11 rerun.
--
-- `research ingest --from-db --measures-only` overwrites these columns in place, and the
-- HIV4, Linsear Write and text_standard columns are dropped after it
-- (db/migrations/20261111_2000_*, 20261111_2100_*). This table keeps the stored values, one
-- row per documents row, so old and new can be compared and analyses made with the old
-- numbers can be reproduced. lm_per_1000_words, hiv4_per_1000_words and ml_per_1000_words
-- are computed by Postgres from these columns and num_words, so they are not copied.
--
-- Run this before the rerun starts. The WHERE clause makes it copy nothing if any row
-- already names a measures run, i.e. if the rerun has begun; the check below then fails.

CREATE TABLE public.documents_measures_before_rerun_20261115 AS
SELECT
    d.id,
    d.source_file,
    d.year,
    d.num_sentences,
    d.num_words,
    d.sentiment,
    d.ml_score,
    d.lm_score,
    d.hiv4_score,
    d.flesch_reading_ease,
    d.flesch_kincaid_grade,
    d.smog_index,
    d.coleman_liau_index,
    d.automated_readability,
    d.dale_chall_score,
    d.difficult_words,
    d.linsear_write_formula,
    d.gunning_fog,
    d.text_standard
FROM public.documents d
WHERE NOT EXISTS (SELECT 1 FROM public.documents x WHERE x.measures_run_id IS NOT NULL);

ALTER TABLE public.documents_measures_before_rerun_20261115 ADD PRIMARY KEY (id);

COMMENT ON TABLE public.documents_measures_before_rerun_20261115 IS
    'documents text measures as stored before the 2026-11 rerun (db/fixes/2026-11-15_documents_measures_snapshot.sql). Read only.';

-- Nobody but the owner writes to it
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.documents_measures_before_rerun_20261115 FROM research_app;

-- Check (read only): same rows and same values as documents. Both sides must give the same
-- count and fingerprint before the rerun starts.
-- SELECT count(*), md5(string_agg(row(id, num_sentences, num_words, sentiment, ml_score, lm_score,
--         hiv4_score, flesch_reading_ease, flesch_kincaid_grade, smog_index, coleman_liau_index,
--         automated_readability, dale_chall_score, difficult_words, linsear_write_formula,
--         gunning_fog, text_standard)::text, '|' ORDER BY id))
-- FROM public.documents;   -- and the same FROM public.documents_measures_before_rerun_20261115
