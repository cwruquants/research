-- Adds length-adjusted versions of the three dictionary sentiment scores to documents.
-- lm_score, hiv4_score and ml_score are sums over the call, so they grow with its length;
-- these are the same scores per 1,000 words, the ones to compare across calls.
--
-- They are virtual generated columns: Postgres computes them from the stored columns on
-- read, so they hold no data of their own, need no backfill and cannot disagree with the
-- sums. NULL when the document has no words.
-- Created by the research agent for the data cleanup, 2026-11-11.

ALTER TABLE public.documents
    ADD COLUMN lm_per_1000_words double precision
        GENERATED ALWAYS AS (1000.0 * lm_score / NULLIF(num_words, 0)) VIRTUAL,
    ADD COLUMN hiv4_per_1000_words double precision
        GENERATED ALWAYS AS (1000.0 * hiv4_score / NULLIF(num_words, 0)) VIRTUAL,
    ADD COLUMN ml_per_1000_words double precision
        GENERATED ALWAYS AS (1000.0 * ml_score / NULLIF(num_words, 0)) VIRTUAL;

COMMENT ON COLUMN public.documents.lm_per_1000_words IS
    'lm_score per 1,000 words (computed on read). Use this, not lm_score, to compare calls.';
COMMENT ON COLUMN public.documents.hiv4_per_1000_words IS
    'hiv4_score per 1,000 words (computed on read). Use this, not hiv4_score, to compare calls.';
COMMENT ON COLUMN public.documents.ml_per_1000_words IS
    'ml_score per 1,000 words (computed on read). Use this, not ml_score, to compare calls.';
