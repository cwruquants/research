-- Adds document_event_returns: event-window stock returns for each call in
-- document_crsp_links, from src/returns/event_returns.py. Day 0 is the first
-- trading session after the call (next day for calls at or after the 4pm New
-- York close). Windows are trading days from day 0: m1_p1 = [-1,+1],
-- p0_p1 = [0,+1], p2_p61 = [+2,+61]. ret_ is the raw buy-and-hold return,
-- bhar_ that minus the CRSP value-weighted index, car_mm_ the market-model
-- CAR with alpha/beta from days [-250,-30]. n_ counts the days with a return.
-- Created by the research agent for the event-returns work, 2026-10-18.

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
    CONSTRAINT document_event_returns_pkey PRIMARY KEY (document_id),
    CONSTRAINT document_event_returns_document_id_fkey FOREIGN KEY (document_id)
        REFERENCES public.documents(id) ON DELETE CASCADE,
    CONSTRAINT document_event_returns_call_timing_check
        CHECK (call_timing IN ('pre_open', 'market_hours', 'after_close'))
);

CREATE INDEX idx_document_event_returns_permno ON public.document_event_returns (permno);
