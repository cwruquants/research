-- Adds transcript_files: a working copy of every raw transcript XML the vendor sent,
-- one row per file revision, so recomputes and checks need no files on disk. The zips
-- stay the record of truth; `research transcripts load` fills the table from them and
-- `research transcripts verify` checks it against them.
--
-- The vendor re-issues a call under the same file name with a later lastUpdate stamp,
-- so one source_file can have several rows. transcript_files_latest picks the one to
-- measure. Rows are append-only: the pipeline logins may insert and read, not change.
-- Created by the research agent for the data cleanup, 2026-11-11.

CREATE TABLE public.transcript_files (
    id bigint GENERATED ALWAYS AS IDENTITY,
    source_file text NOT NULL,
    sha256 bytea NOT NULL,
    byte_length integer NOT NULL,
    xml text COMPRESSION lz4 NOT NULL,
    last_update timestamptz,
    story_action text,
    story_version text,
    story_type text,
    folder_year integer,
    archive text NOT NULL,
    member text NOT NULL,
    loaded_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT transcript_files_pkey PRIMARY KEY (id),
    CONSTRAINT transcript_files_source_file_sha256_key UNIQUE (source_file, sha256)
);

COMMENT ON TABLE public.transcript_files IS
    'Raw vendor transcript XMLs, one row per file revision. Working copy; the zips are the record of truth.';
COMMENT ON COLUMN public.transcript_files.source_file IS
    'File name, <vendor event id>_T.xml (or _B.xml for a brief). Joins to documents.source_file.';
COMMENT ON COLUMN public.transcript_files.sha256 IS 'SHA-256 of the original file bytes.';
COMMENT ON COLUMN public.transcript_files.xml IS 'The whole file, decoded as UTF-8.';
COMMENT ON COLUMN public.transcript_files.last_update IS
    'The Event tag''s lastUpdate stamp; orders revisions.';
COMMENT ON COLUMN public.transcript_files.story_action IS
    'EventStory action: publish, or delete for a stub that withdraws the call.';
COMMENT ON COLUMN public.transcript_files.story_version IS 'EventStory version: Final or Preliminary.';
COMMENT ON COLUMN public.transcript_files.story_type IS 'EventStory storyType: transcript or brief.';
COMMENT ON COLUMN public.transcript_files.folder_year IS 'Year folder the file sat in inside its archive.';
COMMENT ON COLUMN public.transcript_files.archive IS
    'File name of the zip (or folder) the row was loaded from.';
COMMENT ON COLUMN public.transcript_files.member IS 'Path of the file inside the archive.';

REVOKE UPDATE, DELETE, TRUNCATE ON public.transcript_files FROM research_agent, research_app;

-- One row per call: the latest published transcript. Ties on last_update go to the
-- later year folder, then to the hash, so the choice never depends on load order.
-- withdrawn_at is set when the vendor sent a delete stub after that revision.
CREATE VIEW public.transcript_files_latest AS
SELECT DISTINCT ON (f.source_file)
    f.id,
    f.source_file,
    f.sha256,
    f.byte_length,
    f.xml,
    f.last_update,
    f.story_version,
    f.folder_year,
    f.archive,
    f.member,
    f.loaded_at,
    (
        SELECT max(d.last_update)
        FROM public.transcript_files d
        WHERE d.source_file = f.source_file
            AND d.story_action = 'delete'
            AND d.last_update > f.last_update
    ) AS withdrawn_at
FROM public.transcript_files f
WHERE f.story_action = 'publish'
    AND f.story_type = 'transcript'
ORDER BY f.source_file, f.last_update DESC NULLS LAST, f.folder_year DESC NULLS LAST, f.sha256 DESC;

COMMENT ON VIEW public.transcript_files_latest IS
    'The latest published transcript revision of each call, with withdrawn_at when the vendor later pulled it.';
