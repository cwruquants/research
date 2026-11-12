-- Adds pipeline_runs: one row per run of `research ingest` or `research backfill`, saying
-- which code, package versions, sentiment model and word lists made the rows it wrote.
-- documents.measures_run_id and sentiment_run_id, and exposure_runs.pipeline_run_id,
-- point at it. They are NULL for rows written before 2026-11-11, which is how the earlier
-- batches (four exposure batches, two sentiment batches) became impossible to trace.
--
-- measures_run_id covers the counts, readability and LM/ML tone; sentiment_run_id the
-- transformer score. They differ when ingest --measures-only runs without --sentiment.
-- Rows are append-only: a trigger refuses updates, deletes and truncates for every login.
-- Created by the research agent for the data cleanup, 2026-11-11.

CREATE TABLE public.pipeline_runs (
    id bigint GENERATED ALWAYS AS IDENTITY,
    step text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT now(),
    command text NOT NULL,
    git_commit text,
    git_dirty boolean,
    code_sha256 text NOT NULL,
    versions jsonb NOT NULL,
    settings jsonb NOT NULL,
    CONSTRAINT pipeline_runs_pkey PRIMARY KEY (id),
    CONSTRAINT pipeline_runs_step_check CHECK (step IN ('ingest', 'backfill'))
);

COMMENT ON TABLE public.pipeline_runs IS
    'One row per ingest or backfill run: the code, versions, model and word lists behind the rows it wrote.';
COMMENT ON COLUMN public.pipeline_runs.command IS 'The command line, e.g. research ingest --from-db --measures-only.';
COMMENT ON COLUMN public.pipeline_runs.git_commit IS 'HEAD of the checkout it ran from; NULL when that was not a git checkout.';
COMMENT ON COLUMN public.pipeline_runs.git_dirty IS 'True when the checkout had uncommitted changes under src/.';
COMMENT ON COLUMN public.pipeline_runs.code_sha256 IS
    'SHA-256 over the research package source files, so runs from the same code match even without git.';
COMMENT ON COLUMN public.pipeline_runs.versions IS 'Python, platform and package versions.';
COMMENT ON COLUMN public.pipeline_runs.settings IS
    'The run''s options: source, years, sentiment model and revision, word lists with their SHA-256.';

-- Append-only. The agent login owns the table and keeps UPDATE, because the foreign-key
-- checks on documents and exposure_runs lock the referenced row (SELECT ... FOR KEY SHARE),
-- which needs it; a trigger refuses every update, delete and truncate instead.
CREATE FUNCTION public.pipeline_runs_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'pipeline_runs is append-only: a run''s record never changes after it starts';
END
$$;
CREATE TRIGGER pipeline_runs_no_update_or_delete BEFORE UPDATE OR DELETE ON public.pipeline_runs
    FOR EACH ROW EXECUTE FUNCTION public.pipeline_runs_append_only();
CREATE TRIGGER pipeline_runs_no_truncate BEFORE TRUNCATE ON public.pipeline_runs
    FOR EACH STATEMENT EXECUTE FUNCTION public.pipeline_runs_append_only();
REVOKE UPDATE, DELETE, TRUNCATE ON public.pipeline_runs FROM research_app;

ALTER TABLE public.documents
    ADD COLUMN measures_run_id bigint,
    ADD COLUMN sentiment_run_id bigint,
    ADD CONSTRAINT documents_measures_run_id_fkey FOREIGN KEY (measures_run_id)
        REFERENCES public.pipeline_runs(id),
    ADD CONSTRAINT documents_sentiment_run_id_fkey FOREIGN KEY (sentiment_run_id)
        REFERENCES public.pipeline_runs(id);

ALTER TABLE public.exposure_runs
    ADD COLUMN pipeline_run_id bigint,
    ADD CONSTRAINT exposure_runs_pipeline_run_id_fkey FOREIGN KEY (pipeline_run_id)
        REFERENCES public.pipeline_runs(id);

COMMENT ON COLUMN public.documents.measures_run_id IS
    'pipeline_runs row that wrote num_words, num_sentences, readability, lm_score and ml_score.';
COMMENT ON COLUMN public.documents.sentiment_run_id IS 'pipeline_runs row that wrote sentiment.';
COMMENT ON COLUMN public.exposure_runs.pipeline_run_id IS 'pipeline_runs row of the backfill that wrote this run.';
