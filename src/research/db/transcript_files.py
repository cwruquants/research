"""Read and write transcript_files, the table of raw vendor transcript XMLs."""

from typing import Dict, List, Optional, Set, Tuple

import psycopg

from research.transcripts.files import TranscriptFile

# Rows that repeat a (source_file, sha256) already stored, or stored earlier in the same
# statement, are dropped: a byte-identical second copy is never kept
INSERT_FILES_SQL = """
    INSERT INTO public.transcript_files (
        source_file, sha256, byte_length, xml, last_update,
        story_action, story_version, story_type, folder_year, archive, member
    )
    SELECT * FROM unnest(
        %s::text[], %s::bytea[], %s::int[], %s::text[], %s::timestamptz[],
        %s::text[], %s::text[], %s::text[], %s::int[], %s::text[], %s::text[]
    )
    ON CONFLICT (source_file, sha256) DO NOTHING
"""

# The stored text must reproduce the file: same bytes, so the same hash and length
CHANGED_ROWS_SQL = """
    SELECT count(*)
    FROM public.transcript_files
    WHERE archive = ANY(%s)
        AND (sha256(convert_to(xml, 'UTF8')) <> sha256 OR octet_length(convert_to(xml, 'UTF8')) <> byte_length)
"""


def loaded_members(conn: psycopg.Connection, archive: str) -> Set[str]:
    """Paths inside one archive that already have a row."""
    with conn.cursor() as cur:
        cur.execute("SELECT member FROM public.transcript_files WHERE archive = %s", [archive])
        return {member for (member,) in cur.fetchall()}


def insert_files(conn: psycopg.Connection, files: List[TranscriptFile]) -> int:
    """Insert a batch in one transaction; returns how many rows were new."""
    if not files:
        return 0
    columns = [
        [getattr(f, name) for f in files]
        for name in (
            "source_file",
            "sha256",
            "byte_length",
            "xml",
            "last_update",
            "story_action",
            "story_version",
            "story_type",
            "folder_year",
            "archive",
            "member",
        )
    ]
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(INSERT_FILES_SQL, columns)
        return cur.rowcount


def stored_revisions(conn: psycopg.Connection) -> Set[Tuple[str, bytes]]:
    """(source_file, sha256) of every row."""
    with conn.cursor() as cur:
        cur.execute("SELECT source_file, sha256 FROM public.transcript_files")
        return {(source_file, bytes(sha256)) for source_file, sha256 in cur.fetchall()}


def rows_loaded_from(conn: psycopg.Connection, archives: List[str]) -> Set[Tuple[str, bytes]]:
    """(source_file, sha256) of the rows whose first copy came from these archives."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_file, sha256 FROM public.transcript_files WHERE archive = ANY(%s)", [archives]
        )
        return {(source_file, bytes(sha256)) for source_file, sha256 in cur.fetchall()}


def changed_rows(conn: psycopg.Connection, archives: List[str]) -> int:
    """Rows from these archives whose stored text no longer hashes to the stored SHA-256."""
    with conn.cursor() as cur:
        cur.execute(CHANGED_ROWS_SQL, [archives])
        return cur.fetchone()[0]


# Calls whose latest published transcript has (with_document) or has no documents row yet
LATEST_NAMES_WITH_DOCUMENT_SQL = """
    SELECT l.source_file, l.folder_year
    FROM public.transcript_files_latest l
    JOIN public.documents d ON d.source_file = l.source_file
    WHERE %(years)s::int[] IS NULL OR d.year = ANY(%(years)s::int[])
    ORDER BY l.source_file
"""

LATEST_NAMES_WITHOUT_DOCUMENT_SQL = """
    SELECT l.source_file, l.folder_year
    FROM public.transcript_files_latest l
    WHERE NOT EXISTS (SELECT 1 FROM public.documents d WHERE d.source_file = l.source_file)
        AND (%(years)s::int[] IS NULL OR l.folder_year = ANY(%(years)s::int[]))
    ORDER BY l.source_file
"""


def latest_transcript_names(
    conn: psycopg.Connection, with_document: bool, years=None
) -> List[Tuple[str, Optional[int]]]:
    """(source_file, folder_year) of the latest published transcript of each call."""
    sql = LATEST_NAMES_WITH_DOCUMENT_SQL if with_document else LATEST_NAMES_WITHOUT_DOCUMENT_SQL
    with conn.cursor() as cur:
        cur.execute(sql, {"years": list(years) if years else None})
        return cur.fetchall()


def latest_xml(conn: psycopg.Connection, source_files: List[str]) -> Dict[str, str]:
    """The XML of the latest published transcript of each named call that has one."""
    if not source_files:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source_file, xml FROM public.transcript_files_latest WHERE source_file = ANY(%s)",
            [list(source_files)],
        )
        return dict(cur.fetchall())
