-- Applied once on 2026-10-16: set documents.year to the UTC year of event_date for the
-- 2,720 reviewed rows whose year came from the batch folder instead of the call date.
-- Recorded from the DB write log; re-running is a no-op (the count/fingerprint guard
-- fails once the rows are fixed). Before/after values: data/private/db_fixes/.

-- [2026-10-16T20:59]
-- Set documents.year to the UTC year of event_date for the 2,720 reviewed rows.
-- Guarded: changes nothing unless the set matches the reviewed count and fingerprint.
WITH year_fix AS (
    SELECT
        id,
        year                                                  AS old_year,
        EXTRACT(YEAR FROM event_date AT TIME ZONE 'UTC')::int AS new_year
    FROM public.documents
    WHERE event_date IS NOT NULL
      AND year <> EXTRACT(YEAR FROM event_date AT TIME ZONE 'UTC')::int
),
guard AS (
    SELECT
        count(*)                                                                    AS n,
        md5(string_agg(id || ':' || old_year || ':' || new_year, ',' ORDER BY id))  AS fp
    FROM year_fix
),
updated AS (
    UPDATE public.documents d
    SET year = f.new_year
    FROM year_fix f, guard g
    WHERE d.id = f.id
      AND d.year = f.old_year
      AND g.n = 2720
      AND g.fp = '78bfcb2cd4aacfc8406b63866c8b7384'
    RETURNING f.old_year, f.new_year
)
SELECT
    old_year,
    new_year,
    count(*) AS docs
FROM updated
GROUP BY old_year, new_year
ORDER BY old_year, new_year;
