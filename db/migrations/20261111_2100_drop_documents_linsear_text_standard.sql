-- Drops linsear_write_formula and text_standard from documents. textstat's Linsear Write
-- reads only a text's first 100 words: in the stored rows that was the vendor header and
-- participant list (31% of rows scored above 30), and on the spoken text it would be only
-- the Operator's opening. text_standard is a vote across indexes that includes it.
-- Decided by the user, 2026-11-11; applied at the rerun, with the HIV4 drop. The values
-- are in the nightly dumps from before this.
-- Created by the research agent for the data cleanup, 2026-11-11.

ALTER TABLE public.documents
    DROP COLUMN linsear_write_formula,
    DROP COLUMN text_standard;
