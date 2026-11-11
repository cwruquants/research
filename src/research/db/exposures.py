"""Read and write exposure_runs and keyword_matches."""

from typing import Dict, List, Optional, Tuple

import psycopg

MISSING_RUNS_SQL = """
    SELECT d.id, d.source_file, d.year, d.event_title, ks.id, ks.name
    FROM public.documents d
    CROSS JOIN public.keyword_sets ks
    WHERE ks.name = ANY(%(names)s)
        AND (%(years)s::int[] IS NULL OR d.year = ANY(%(years)s::int[]))
        AND d.source_file IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM public.exposure_runs er
            WHERE er.document_id = d.id AND er.keyword_set_id = ks.id
        )
    ORDER BY d.id, ks.id
"""

INSERT_RUNS_SQL = """
    INSERT INTO public.exposure_runs (
        document_id, keyword_set_id, matching_method, cosine_threshold, runtime_seconds,
        total_keywords_searched, total_keywords_with_matches, total_direct_matches,
        total_cosine_matches, total_sentences_in_document, total_sentences_with_match,
        pct_sentences_with_match, run_timestamp
    )
    SELECT * FROM unnest(
        %s::bigint[], %s::bigint[], %s::text[], %s::float8[], %s::float8[],
        %s::int[], %s::int[], %s::int[],
        %s::int[], %s::int[], %s::int[],
        %s::float8[], %s::text[]
    )
    ON CONFLICT (document_id, keyword_set_id) DO NOTHING
    RETURNING id, document_id, keyword_set_id
"""

COPY_MATCHES_SQL = """
    COPY public.keyword_matches (exposure_run_id, keyword, match_type, matched_text, context, position)
    FROM STDIN
"""


def keyword_set_sizes(conn: psycopg.Connection, names: List[str]) -> Dict[str, int]:
    """keyword_sets.total_keywords for each named set that exists."""
    with conn.cursor() as cur:
        cur.execute("SELECT name, total_keywords FROM public.keyword_sets WHERE name = ANY(%s)", [names])
        return dict(cur.fetchall())


def register_keyword_set(conn: psycopg.Connection, name: str, source_file: str, total_keywords: int) -> None:
    """Add a keyword_sets row for a set that has never been run."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.keyword_sets (name, source_file, total_keywords) VALUES (%s, %s, %s)",
            [name, source_file, total_keywords],
        )


def missing_runs(conn: psycopg.Connection, names: List[str], years: Optional[List[int]]) -> List[dict]:
    """Documents with at least one missing run for the named sets, with the sets each one needs."""
    with conn.cursor() as cur:
        cur.execute(MISSING_RUNS_SQL, {"names": names, "years": years})
        docs: Dict[int, dict] = {}
        for document_id, source_file, year, event_title, keyword_set_id, name in cur.fetchall():
            doc = docs.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "source_file": source_file,
                    "year": year,
                    "event_title": event_title,
                    "keyword_sets": [],
                },
            )
            doc["keyword_sets"].append((keyword_set_id, name))
    return list(docs.values())


def write_runs(
    conn, run_rows: List[tuple], match_rows: Dict[Tuple[int, int], List[tuple]]
) -> Tuple[int, int]:
    """
    Insert exposure runs and the matches of the runs that were new, in one transaction.
    run_rows follow INSERT_RUNS_SQL's columns; match_rows are keyed by (document_id,
    keyword_set_id). Returns (runs written, matches written).
    """
    if not run_rows:
        return 0, 0
    matches_written = 0
    columns = [list(col) for col in zip(*run_rows)]
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(INSERT_RUNS_SQL, columns)
            new_runs = cur.fetchall()
            with cur.copy(COPY_MATCHES_SQL) as copy:
                for run_id, document_id, keyword_set_id in new_runs:
                    for row in match_rows[(document_id, keyword_set_id)]:
                        copy.write_row((run_id, *row))
                        matches_written += 1
    return len(new_runs), matches_written
