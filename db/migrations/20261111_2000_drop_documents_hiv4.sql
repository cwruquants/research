-- Drops the Harvard IV-4 dictionary score from documents: hiv4_score and its rate column
-- hiv4_per_1000_words. In earnings calls the score was not usable. pysentiment2 stems each
-- word before looking it up, so nearly half its hits were not the dictionary word itself
-- ("will" counted as positive through "willing", "margin" as negative through "marginal"),
-- and the Harvard lists are general English, treating words such as "cost", "capital" and
-- "service" as negative. Loughran-McDonald (lm_score) is the finance-specific dictionary.
-- Decided by the user, 2026-11-11. The values are in the nightly dumps from before this.
-- Created by the research agent for the data cleanup, 2026-11-11.

ALTER TABLE public.documents
    DROP COLUMN hiv4_per_1000_words,
    DROP COLUMN hiv4_score;
