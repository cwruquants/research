-- Adds document_crsp_links: one row per earnings call linked to a CRSP security
-- (PERMNO) on the call date, from src/linking/crsp_link.py. Calls with no link
-- (foreign listings, no CRSP match, name mismatches, ambiguous) have no row.
-- Returns join on permno; ncusip8 is the CRSP NCUSIP in effect on the call date.
-- Created by the research agent for the CRSP linking work, 2026-10-18.

CREATE TABLE public.document_crsp_links (
    document_id bigint NOT NULL,
    permno integer NOT NULL,
    permco integer,
    ncusip8 text,
    crsp_ticker text,
    comnam text,
    route text NOT NULL,
    method text NOT NULL,
    note text,
    title_score double precision,
    linked_at timestamptz DEFAULT now(),
    CONSTRAINT document_crsp_links_pkey PRIMARY KEY (document_id),
    CONSTRAINT document_crsp_links_document_id_fkey FOREIGN KEY (document_id)
        REFERENCES public.documents(id) ON DELETE CASCADE,
    CONSTRAINT document_crsp_links_route_check CHECK (route IN ('A', 'B'))
);

CREATE INDEX idx_document_crsp_links_permno ON public.document_crsp_links (permno);
