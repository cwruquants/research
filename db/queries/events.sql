-- Input to `research returns fetch` and `research returns compute` (--events):
-- every linked call with its PERMNO and event time.
-- Run with PGTZ=America/New_York so event_date prints as it did in the original export.
SELECT l.document_id, l.permno, d.event_date, d.event_title
FROM public.document_crsp_links l
JOIN public.documents d ON d.id = l.document_id
ORDER BY l.document_id
