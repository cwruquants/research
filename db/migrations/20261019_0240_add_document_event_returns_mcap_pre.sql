-- Adds mcap_pre to document_event_returns: market cap in USD millions,
-- |prc| * shrout from CRSP dsf on the last trading day at least 2 calendar
-- days before day0 (within 30 days), null if there is none. Size control for
-- the return regressions, from src/returns/market_cap.py.
-- Created by the research agent at the user's request, 2026-10-19.

ALTER TABLE public.document_event_returns
    ADD COLUMN mcap_pre double precision;

COMMENT ON COLUMN public.document_event_returns.mcap_pre IS
    'Market cap in USD millions, |prc| * shrout from CRSP dsf on the last trading day at least 2 calendar days before day0 (within 30 days); null if none';
