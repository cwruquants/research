-- Adds six columns to documents that the rerun of text measures fills.
--
-- lm_positive, lm_negative, ml_positive, ml_negative: the two word counts behind each
-- dictionary score (lm_score = lm_positive - lm_negative). The net score alone cannot tell
-- a flat call from one with many positive and many negative words.
--
-- event_type_id, event_type: the vendor's event type, read from the Event tag of the
-- transcript file (1, 'Earnings Conference Call'). It was parsed but never stored, so
-- "earnings call" had to be guessed from the title.
--
-- All six are NULL until `research ingest` writes the row again; measures_run_id says
-- which run did.
-- Created by the research agent for the data cleanup, 2026-11-12.

ALTER TABLE public.documents
    ADD COLUMN lm_positive integer,
    ADD COLUMN lm_negative integer,
    ADD COLUMN ml_positive integer,
    ADD COLUMN ml_negative integer,
    ADD COLUMN event_type_id integer,
    ADD COLUMN event_type text;

COMMENT ON COLUMN public.documents.lm_positive IS
    'Words on the Loughran-McDonald positive list. lm_score = lm_positive - lm_negative.';
COMMENT ON COLUMN public.documents.lm_negative IS
    'Words on the Loughran-McDonald negative list. lm_score = lm_positive - lm_negative.';
COMMENT ON COLUMN public.documents.ml_positive IS
    'Words on the Garcia ML positive unigram list. ml_score = ml_positive - ml_negative.';
COMMENT ON COLUMN public.documents.ml_negative IS
    'Words on the Garcia ML negative unigram list. ml_score = ml_positive - ml_negative.';
COMMENT ON COLUMN public.documents.event_type_id IS
    'The vendor''s event type id, from the Event tag of the transcript file.';
COMMENT ON COLUMN public.documents.event_type IS
    'The vendor''s event type name, from the Event tag of the transcript file.';
