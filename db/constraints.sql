-- Apply after the data restore. Primary and unique constraints build their
-- indexes once; foreign keys validate the copied relationships.

ALTER TABLE public.companies
    ADD CONSTRAINT companies_pkey PRIMARY KEY (id),
    ADD CONSTRAINT companies_name_ticker_key UNIQUE NULLS NOT DISTINCT (name, ticker);

ALTER TABLE public.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (id),
    ADD CONSTRAINT documents_source_file_unique UNIQUE (source_file);

ALTER TABLE public.keyword_sets
    ADD CONSTRAINT keyword_sets_pkey PRIMARY KEY (id),
    ADD CONSTRAINT keyword_sets_name_key UNIQUE (name);

ALTER TABLE public.exposure_runs
    ADD CONSTRAINT exposure_runs_pkey PRIMARY KEY (id),
    ADD CONSTRAINT exposure_runs_doc_kwset_unique UNIQUE (document_id, keyword_set_id);

ALTER TABLE public.document_crsp_links
    ADD CONSTRAINT document_crsp_links_pkey PRIMARY KEY (document_id);

ALTER TABLE public.keyword_matches
    ADD CONSTRAINT keyword_matches_pkey PRIMARY KEY (id),
    ADD CONSTRAINT keyword_matches_unique UNIQUE (exposure_run_id, keyword, match_type, matched_text, "position");

CREATE INDEX idx_documents_company ON public.documents (company_id);
CREATE INDEX idx_documents_event_date ON public.documents (event_date);
CREATE INDEX idx_documents_year ON public.documents (year);
CREATE INDEX idx_document_crsp_links_permno ON public.document_crsp_links (permno);
CREATE INDEX idx_exposure_doc ON public.exposure_runs (document_id);
CREATE INDEX idx_exposure_kw ON public.exposure_runs (keyword_set_id);
CREATE INDEX idx_matches_keyword ON public.keyword_matches (keyword);
CREATE INDEX idx_matches_run ON public.keyword_matches (exposure_run_id);

ALTER TABLE public.documents
    ADD CONSTRAINT documents_company_id_fkey FOREIGN KEY (company_id)
    REFERENCES public.companies(id) ON DELETE CASCADE;
ALTER TABLE public.exposure_runs
    ADD CONSTRAINT exposure_runs_document_id_fkey FOREIGN KEY (document_id)
    REFERENCES public.documents(id) ON DELETE CASCADE,
    ADD CONSTRAINT exposure_runs_keyword_set_id_fkey FOREIGN KEY (keyword_set_id)
    REFERENCES public.keyword_sets(id) ON DELETE CASCADE;
ALTER TABLE public.keyword_matches
    ADD CONSTRAINT keyword_matches_exposure_run_id_fkey FOREIGN KEY (exposure_run_id)
    REFERENCES public.exposure_runs(id) ON DELETE CASCADE;
ALTER TABLE public.document_crsp_links
    ADD CONSTRAINT document_crsp_links_document_id_fkey FOREIGN KEY (document_id)
    REFERENCES public.documents(id) ON DELETE CASCADE;
