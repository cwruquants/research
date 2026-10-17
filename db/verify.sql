-- Compare the counts with the saved Supabase baseline after the restore.
SELECT 'companies' AS table_name, count(*) AS rows FROM public.companies
UNION ALL SELECT 'documents', count(*) FROM public.documents
UNION ALL SELECT 'keyword_sets', count(*) FROM public.keyword_sets
UNION ALL SELECT 'exposure_runs', count(*) FROM public.exposure_runs
UNION ALL SELECT 'keyword_matches', count(*) FROM public.keyword_matches;

-- Each identity sequence must be at or ahead of the highest imported id.
SELECT 'companies' AS table_name,
       coalesce((SELECT max(id) FROM public.companies) <= (SELECT last_value FROM public.companies_id_seq), true) AS sequence_ok
UNION ALL SELECT 'documents',
       coalesce((SELECT max(id) FROM public.documents) <= (SELECT last_value FROM public.documents_id_seq), true)
UNION ALL SELECT 'keyword_sets',
       coalesce((SELECT max(id) FROM public.keyword_sets) <= (SELECT last_value FROM public.keyword_sets_id_seq), true)
UNION ALL SELECT 'exposure_runs',
       coalesce((SELECT max(id) FROM public.exposure_runs) <= (SELECT last_value FROM public.exposure_runs_id_seq), true)
UNION ALL SELECT 'keyword_matches',
       coalesce((SELECT max(id) FROM public.keyword_matches) <= (SELECT last_value FROM public.keyword_matches_id_seq), true);

SELECT conname, convalidated
FROM pg_constraint
WHERE connamespace = 'public'::regnamespace AND contype = 'f'
ORDER BY conname;
