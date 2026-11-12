-- Base tables and identity sequences for the local research database.
-- Keep this file free of indexes and foreign keys for efficient bulk loads.

-- One row per ingest or backfill run: the code, versions, model and word lists behind it.
CREATE TABLE public.pipeline_runs (
    id bigint GENERATED ALWAYS AS IDENTITY,
    step text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT now(),
    command text NOT NULL,
    git_commit text,
    git_dirty boolean,
    code_sha256 text NOT NULL,
    versions jsonb NOT NULL,
    settings jsonb NOT NULL
);

CREATE TABLE public.companies (
    id bigint GENERATED ALWAYS AS IDENTITY,
    name text NOT NULL,
    ticker text,
    created_at timestamptz DEFAULT now()
);

CREATE TABLE public.documents (
    id bigint GENERATED ALWAYS AS IDENTITY,
    company_id bigint,
    year integer NOT NULL,
    source_file text,
    event_title text,
    event_date timestamptz,
    city text,
    num_sentences integer,
    num_words integer,
    sentiment double precision,
    ml_score double precision,
    lm_score double precision,
    flesch_reading_ease double precision,
    flesch_kincaid_grade double precision,
    smog_index double precision,
    coleman_liau_index double precision,
    automated_readability double precision,
    dale_chall_score double precision,
    difficult_words integer,
    gunning_fog double precision,
    created_at timestamptz DEFAULT now(),
    rc_active_ratio double precision,
    rc_passive_ratio double precision,
    re_active_ratio double precision,
    re_passive_ratio double precision,
    rx_active_ratio double precision,
    rx_passive_ratio double precision,
    risk_total_sentences integer,
    -- The dictionary scores per 1,000 words, computed on read from the sums above
    lm_per_1000_words double precision
        GENERATED ALWAYS AS (1000.0 * lm_score / NULLIF(num_words, 0)) VIRTUAL,
    ml_per_1000_words double precision
        GENERATED ALWAYS AS (1000.0 * ml_score / NULLIF(num_words, 0)) VIRTUAL,
    measures_run_id bigint,   -- pipeline_runs row behind the counts, readability and LM/ML tone
    sentiment_run_id bigint   -- pipeline_runs row behind sentiment
);

CREATE TABLE public.keyword_sets (
    id bigint GENERATED ALWAYS AS IDENTITY,
    name text NOT NULL,
    source_file text,
    total_keywords integer,
    created_at timestamptz DEFAULT now()
);

CREATE TABLE public.exposure_runs (
    id bigint GENERATED ALWAYS AS IDENTITY,
    document_id bigint,
    keyword_set_id bigint,
    matching_method text DEFAULT 'direct',
    cosine_threshold double precision,
    runtime_seconds double precision,
    total_keywords_searched integer,
    total_keywords_with_matches integer,
    total_direct_matches integer,
    total_cosine_matches integer,
    total_sentences_in_document integer,
    total_sentences_with_match integer,
    pct_sentences_with_match double precision,
    run_timestamp text,
    created_at timestamptz DEFAULT now(),
    pipeline_run_id bigint    -- pipeline_runs row of the backfill that wrote it
);

CREATE TABLE public.keyword_matches (
    id bigint GENERATED ALWAYS AS IDENTITY,
    exposure_run_id bigint,
    keyword text NOT NULL,
    match_type text NOT NULL,
    matched_text text,
    context text,
    "position" integer,
    created_at timestamptz DEFAULT now(),
    CONSTRAINT keyword_matches_match_type_check CHECK (match_type IN ('direct', 'cosine'))
);

CREATE TABLE public.document_crsp_links (
    document_id bigint NOT NULL,
    permno integer NOT NULL,
    permco integer,
    ncusip8 text,
    crsp_ticker text,
    comnam text,
    route text NOT NULL,
    method text NOT NULL,
    note text,
    title_score double precision,
    linked_at timestamptz DEFAULT now(),
    CONSTRAINT document_crsp_links_route_check CHECK (route IN ('A', 'B'))
);

CREATE TABLE public.document_event_returns (
    document_id bigint NOT NULL,
    permno integer NOT NULL,
    call_day date NOT NULL,
    call_timing text NOT NULL,
    day0 date,
    is_earnings_call boolean NOT NULL,
    mm_est_days integer,
    mm_alpha double precision,
    mm_beta double precision,
    n_m20_m2 integer,
    ret_m20_m2 double precision,
    bhar_m20_m2 double precision,
    car_mm_m20_m2 double precision,
    n_p0_p0 integer,
    ret_p0_p0 double precision,
    bhar_p0_p0 double precision,
    car_mm_p0_p0 double precision,
    n_p0_p1 integer,
    ret_p0_p1 double precision,
    bhar_p0_p1 double precision,
    car_mm_p0_p1 double precision,
    n_m1_p1 integer,
    ret_m1_p1 double precision,
    bhar_m1_p1 double precision,
    car_mm_m1_p1 double precision,
    n_m1_p2 integer,
    ret_m1_p2 double precision,
    bhar_m1_p2 double precision,
    car_mm_m1_p2 double precision,
    n_p0_p2 integer,
    ret_p0_p2 double precision,
    bhar_p0_p2 double precision,
    car_mm_p0_p2 double precision,
    n_p2_p21 integer,
    ret_p2_p21 double precision,
    bhar_p2_p21 double precision,
    car_mm_p2_p21 double precision,
    n_p22_p42 integer,
    ret_p22_p42 double precision,
    bhar_p22_p42 double precision,
    car_mm_p22_p42 double precision,
    n_p43_p63 integer,
    ret_p43_p63 double precision,
    bhar_p43_p63 double precision,
    car_mm_p43_p63 double precision,
    n_p2_p61 integer,
    ret_p2_p61 double precision,
    bhar_p2_p61 double precision,
    car_mm_p2_p61 double precision,
    n_p2_p121 integer,
    ret_p2_p121 double precision,
    bhar_p2_p121 double precision,
    car_mm_p2_p121 double precision,
    computed_at timestamptz DEFAULT now(),
    mcap_pre double precision,  -- USD millions, last trading day >= 2 calendar days before day0
    CONSTRAINT document_event_returns_call_timing_check
        CHECK (call_timing IN ('pre_open', 'market_hours', 'after_close'))
);

-- Raw vendor transcript XMLs, one row per file revision (a working copy of the zips).
CREATE TABLE public.transcript_files (
    id bigint GENERATED ALWAYS AS IDENTITY,
    source_file text NOT NULL,
    sha256 bytea NOT NULL,  -- of the original file bytes
    byte_length integer NOT NULL,
    xml text COMPRESSION lz4 NOT NULL,
    last_update timestamptz,
    story_action text,
    story_version text,
    story_type text,
    folder_year integer,
    archive text NOT NULL,
    member text NOT NULL,
    loaded_at timestamptz NOT NULL DEFAULT now()
);
