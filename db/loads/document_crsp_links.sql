-- Load public.document_crsp_links from the output of `research link route-a`
-- (proposed_links_all.csv, which already includes the route B rows).
--
-- Reconstructed 2026-11-03 by comparing the live table with that CSV. The original
-- load on 2026-10-17 went through the DB tool's bulk CSV import, which logged no SQL.
-- The rule below reproduces every row and column of the live table:
--   * only rows with status = 'matched' (350,716 route A + 1,399 route B = 352,115)
--   * route B rows: method 'title_name:<status>' is stored as method 'title_name' and
--     note 'route_b_<status>'; permco comes from CRSP stocknames (one permco per permno)
--
-- Run in psql as a role that can write the table. The \copy paths are relative, so
-- run it from the private folder holding proposed_links_all.csv and stocknames.csv:
--   cd data/private/derived/crsp_linking && cp ../../crsp/stocknames.csv .
--   psql "$DATABASE_URL" -f ../../../../db/loads/document_crsp_links.sql

BEGIN;

CREATE TEMP TABLE stage_links (
    document_id text, company_id text, company_name text, ric text, root text, share_class text,
    exch_suffix text, delist text, call_date text, event_title text, route text, after_crsp_end text,
    lseg_cusip8s text, status text, method text, permno text, permco text, ncusip8 text,
    crsp_ticker text, comnam text, shrcd text, exchcd text, title_score text, company_score text,
    n_candidates text, note text, candidates text, other_methods text, title_status text,
    title_permno text, title_comnam text
) ON COMMIT DROP;
\copy stage_links FROM 'proposed_links_all.csv' WITH (FORMAT csv, HEADER)

CREATE TEMP TABLE stage_permco (permno integer PRIMARY KEY, permco integer) ON COMMIT DROP;
CREATE TEMP TABLE stage_stocknames (permno text, namedt text, nameenddt text, shrcd text, exchcd text,
    siccd text, ncusip text, ticker text, comnam text, shrcls text, permco text, hexcd text, cusip text,
    st_date text, end_date text, namedum text) ON COMMIT DROP;
\copy stage_stocknames FROM 'stocknames.csv' WITH (FORMAT csv, HEADER)
INSERT INTO stage_permco SELECT DISTINCT permno::integer, permco::integer FROM stage_stocknames;

INSERT INTO public.document_crsp_links
    (document_id, permno, permco, ncusip8, crsp_ticker, comnam, route, method, note, title_score)
SELECT
    s.document_id::bigint,
    s.permno::integer,
    COALESCE(NULLIF(s.permco, '')::integer, p.permco),
    NULLIF(s.ncusip8, ''),
    NULLIF(s.crsp_ticker, ''),
    NULLIF(s.comnam, ''),
    s.route,
    CASE WHEN s.route = 'B' THEN split_part(s.method, ':', 1) ELSE s.method END,
    CASE WHEN s.route = 'B' THEN 'route_b_' || split_part(s.method, ':', 2) ELSE NULLIF(s.note, '') END,
    NULLIF(s.title_score, '')::double precision
FROM stage_links s
LEFT JOIN stage_permco p ON p.permno = s.permno::integer
WHERE s.status = 'matched';

COMMIT;
