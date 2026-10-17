BEGIN;
-- The owner is the only role that can ALTER a table. Existing grants to research_app and research_ro carry over.
ALTER TABLE public.companies       OWNER TO research_agent;
ALTER TABLE public.documents       OWNER TO research_agent;
ALTER TABLE public.exposure_runs   OWNER TO research_agent;
ALTER TABLE public.keyword_matches OWNER TO research_agent;
ALTER TABLE public.keyword_sets    OWNER TO research_agent;
GRANT CREATE ON SCHEMA public TO research_agent;
-- Tables the agent creates later must stay usable by the pipeline and by collaborators.
ALTER DEFAULT PRIVILEGES FOR ROLE research_agent IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO research_app;
ALTER DEFAULT PRIVILEGES FOR ROLE research_agent IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO research_app;
ALTER DEFAULT PRIVILEGES FOR ROLE research_agent IN SCHEMA public GRANT SELECT ON TABLES TO research_ro;
COMMIT;
