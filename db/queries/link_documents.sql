-- Input to `research link route-a` (--docs): every document with its company row's
-- ticker (a Refinitiv RIC) and the call date in UTC and in New York time.
-- Run with PGTZ=America/New_York so event_date prints as it did in the original export.
SELECT
    d.id AS document_id,
    d.company_id,
    c.name AS company_name,
    c.ticker,
    d.event_title,
    d.event_date,
    (d.event_date AT TIME ZONE 'UTC')::date AS call_date_utc,
    (d.event_date AT TIME ZONE 'America/New_York')::date AS call_date_ny
FROM public.documents d
JOIN public.companies c ON c.id = d.company_id
ORDER BY d.id
