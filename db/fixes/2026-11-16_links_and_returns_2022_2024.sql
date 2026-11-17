-- Applied once on 2026-11-16 with research_agent, through the DB tool's bulk CSV import
-- (mode insert), which runs the same COPY as the \copy lines below: 60,918 rows into each
-- table. document_crsp_links went from 352,115 to 413,033 rows, document_event_returns
-- likewise; 60,799 of the new returns rows have mcap_pre.
--
-- Links and event returns for the 107,994 calls that `research ingest --from-db --years
-- 2022 2023 2024` added (pipeline_runs id 3). Only these calls were exported and linked, so
-- no stored link or return changed. How the two load files were made:
--
--   1. db/queries/link_documents.sql with
--        WHERE d.measures_run_id = 3
--          AND NOT EXISTS (SELECT 1 FROM public.document_crsp_links l WHERE l.document_id = d.id)
--      exported with PGTZ=America/New_York (107,994 rows), then
--      `research link route-a` with the same stocknames.csv (CRSP end 2024-12-31) and LSEG
--      files as the 2026-10-17 run and an empty --route-b file (header only): the new calls
--      have no hand-picked route B list. 60,918 matched (56.4%), 45,879 foreign_listing.
--      document_crsp_links_load.csv = the matched rows in the columns below, permco filled
--      from stocknames where route A left it blank (the rule in db/loads/document_crsp_links.sql;
--      every row is route A, so method and note are stored as written).
--   2. db/queries/events.sql with
--        WHERE NOT EXISTS (SELECT 1 FROM public.document_event_returns r WHERE r.document_id = l.document_id)
--      exported (60,918 rows), `research returns fetch` (END now 2024-12-31; the 13.4M dsf
--      rows that overlap the 2026-10-18 pull are identical), `returns compute`, `returns mcap`.
--      document_event_returns_load.csv = event_returns.csv with mcap_pre joined on.
--
-- To rebuild, run in psql from a folder holding both load files:
--   psql "$DATABASE_URL" -f db/fixes/2026-11-16_links_and_returns_2022_2024.sql

BEGIN;

\copy public.document_crsp_links (document_id,permno,permco,ncusip8,crsp_ticker,comnam,route,method,note,title_score) FROM 'document_crsp_links_load.csv' WITH (FORMAT csv, HEADER)

\copy public.document_event_returns (document_id,permno,call_day,call_timing,day0,is_earnings_call,mm_est_days,mm_alpha,mm_beta,n_m20_m2,ret_m20_m2,bhar_m20_m2,car_mm_m20_m2,n_p0_p0,ret_p0_p0,bhar_p0_p0,car_mm_p0_p0,n_p0_p1,ret_p0_p1,bhar_p0_p1,car_mm_p0_p1,n_m1_p1,ret_m1_p1,bhar_m1_p1,car_mm_m1_p1,n_m1_p2,ret_m1_p2,bhar_m1_p2,car_mm_m1_p2,n_p0_p2,ret_p0_p2,bhar_p0_p2,car_mm_p0_p2,n_p2_p21,ret_p2_p21,bhar_p2_p21,car_mm_p2_p21,n_p22_p42,ret_p22_p42,bhar_p22_p42,car_mm_p22_p42,n_p43_p63,ret_p43_p63,bhar_p43_p63,car_mm_p43_p63,n_p2_p61,ret_p2_p61,bhar_p2_p61,car_mm_p2_p61,n_p2_p121,ret_p2_p121,bhar_p2_p121,car_mm_p2_p121,mcap_pre) FROM 'document_event_returns_load.csv' WITH (FORMAT csv, HEADER)

COMMIT;

-- Check: both counts 413,033, and no returns row without a link.
--   SELECT (SELECT COUNT(*) FROM public.document_crsp_links),
--          (SELECT COUNT(*) FROM public.document_event_returns),
--          (SELECT COUNT(*) FROM public.document_event_returns r
--            WHERE NOT EXISTS (SELECT 1 FROM public.document_crsp_links l WHERE l.document_id = r.document_id));
