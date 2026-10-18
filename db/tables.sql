-- Base tables and identity sequences for the local research database.
-- Keep this file free of indexes and foreign keys for efficient bulk loads.

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
    hiv4_score double precision,
    flesch_reading_ease double precision,
    flesch_kincaid_grade double precision,
    smog_index double precision,
    coleman_liau_index double precision,
    automated_readability double precision,
    dale_chall_score double precision,
    difficult_words integer,
    linsear_write_formula double precision,
    gunning_fog double precision,
    text_standard text,
    created_at timestamptz DEFAULT now(),
    rc_active_ratio double precision,
    rc_passive_ratio double precision,
    re_active_ratio double precision,
    re_passive_ratio double precision,
    rx_active_ratio double precision,
    rx_passive_ratio double precision,
    risk_total_sentences integer
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
    created_at timestamptz DEFAULT now()
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
    n_m1_p1 integer,
    ret_m1_p1 double precision,
    bhar_m1_p1 double precision,
    car_mm_m1_p1 double precision,
    n_p0_p1 integer,
    ret_p0_p1 double precision,
    bhar_p0_p1 double precision,
    car_mm_p0_p1 double precision,
    n_p2_p61 integer,
    ret_p2_p61 double precision,
    bhar_p2_p61 double precision,
    car_mm_p2_p61 double precision,
    computed_at timestamptz DEFAULT now(),
    CONSTRAINT document_event_returns_call_timing_check
        CHECK (call_timing IN ('pre_open', 'market_hours', 'after_close'))
);
