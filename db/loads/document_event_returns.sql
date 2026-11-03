-- Load public.document_event_returns from the output of `research returns compute`
-- (event_returns.csv, decompressed), then fill mcap_pre from `research returns mcap`
-- (mcap_pre.csv). Reconstructed 2026-11-03; it reproduces the live table. The original
-- 2026-10-18 loads went through the DB tool's bulk CSV import, which logged no SQL.
--
-- The \copy paths are relative, so run it in psql from the folder holding both CSVs
-- (results/event_returns/ in the commands' usage examples; git-ignored):
--   cd results/event_returns && gzip -dk event_returns.csv.gz
--   psql "$DATABASE_URL" -f ../../db/loads/document_event_returns.sql

BEGIN;

\copy public.document_event_returns (document_id,permno,call_day,call_timing,day0,is_earnings_call,mm_est_days,mm_alpha,mm_beta,n_m20_m2,ret_m20_m2,bhar_m20_m2,car_mm_m20_m2,n_p0_p0,ret_p0_p0,bhar_p0_p0,car_mm_p0_p0,n_p0_p1,ret_p0_p1,bhar_p0_p1,car_mm_p0_p1,n_m1_p1,ret_m1_p1,bhar_m1_p1,car_mm_m1_p1,n_m1_p2,ret_m1_p2,bhar_m1_p2,car_mm_m1_p2,n_p0_p2,ret_p0_p2,bhar_p0_p2,car_mm_p0_p2,n_p2_p21,ret_p2_p21,bhar_p2_p21,car_mm_p2_p21,n_p22_p42,ret_p22_p42,bhar_p22_p42,car_mm_p22_p42,n_p43_p63,ret_p43_p63,bhar_p43_p63,car_mm_p43_p63,n_p2_p61,ret_p2_p61,bhar_p2_p61,car_mm_p2_p61,n_p2_p121,ret_p2_p121,bhar_p2_p121,car_mm_p2_p121) FROM 'event_returns.csv' WITH (FORMAT csv, HEADER)

CREATE TEMP TABLE stage_mcap_pre (document_id bigint PRIMARY KEY, mcap_pre double precision NOT NULL) ON COMMIT DROP;
\copy stage_mcap_pre FROM 'mcap_pre.csv' WITH (FORMAT csv, HEADER)
UPDATE public.document_event_returns r
SET mcap_pre = s.mcap_pre
FROM stage_mcap_pre s
WHERE s.document_id = r.document_id;

COMMIT;
