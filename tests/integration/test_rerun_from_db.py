"""The rerun path against a throwaway Postgres (TEST_DATABASE_URL) with the schema: ingest and
backfill read the latest revision from transcript_files. Rows are removed afterwards."""

import zipfile
from uuid import uuid4

import psycopg
import pytest

from research.db.exposures import register_keyword_set
from research.matching.direct import MatchingAgent
from research.pipelines import backfill_exposures
from research.pipelines import ingest as ingest_module
from research.pipelines.load_transcript_files import load_sources

YEAR = 2031  # no real call is dated here, so the year filters pick only this test's rows


def _revision(fixture: bytes, stamp: str, extra: str = "") -> bytes:
    xml = fixture.decode("utf-8")
    xml = xml.replace('<Event Id="', f'<Event lastUpdate="{stamp}" Id="', 1)
    xml = xml.replace('storyType="transcript"', 'action="publish" storyType="transcript"', 1)
    xml = xml.replace("02-Feb-17 1:30pm GMT", f"02-Feb-{YEAR % 100} 1:30pm GMT")
    if extra:
        xml = xml.replace("Gross margin improved to 34%.", f"Gross margin improved to 34%. {extra}")
    return xml.encode("utf-8")


@pytest.mark.db
def test_ingest_and_backfill_read_the_latest_revision(test_database_url, transcripts, tmp_path):
    tag = uuid4().hex[:10]
    known, new = f"9{tag}1_T.xml", f"9{tag}2_T.xml"
    fixture = (transcripts / "spaced_T.xml").read_bytes()
    older = _revision(fixture, "Monday, January 4, 2032 at 9:00:00am GMT")
    latest = _revision(
        fixture, "Monday, March 1, 2032 at 9:00:00am GMT", extra="Demand was strong and pricing was good."
    )
    archive = tmp_path / f"{tag}.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(f"{YEAR}/{known}", older)
        z.writestr(f"{YEAR + 1}/{known}", latest)
        z.writestr(f"{YEAR}/{new}", older)
    load_sources([str(archive)], test_database_url)

    words = tmp_path / "words.csv"
    words.write_text("risk\ntaxes\n", encoding="utf-8")
    set_name = f"test_set_{tag}"
    try:
        # A documents row made from the older revision, as the stored rows were
        first = ingest_module.ingest(
            None, None, database_url=test_database_url, from_db=True, years=[YEAR, YEAR + 1], num_threads=1
        )
        assert first["num_uploaded"] == 2
        with psycopg.connect(test_database_url) as conn:
            conn.execute(
                "UPDATE public.documents SET num_words = 1, lm_score = -99 WHERE source_file = %s", [known]
            )

        rerun = ingest_module.ingest(
            None,
            None,
            database_url=test_database_url,
            from_db=True,
            measures_only=True,
            years=[YEAR],
            num_threads=2,
        )
        assert rerun["num_uploaded"] == 2
        expected = ingest_module.analyze_document(known, xml=latest)["document_attr"]
        with psycopg.connect(test_database_url) as conn:
            num_words, lm = conn.execute(
                "SELECT num_words, lm_score FROM public.documents WHERE source_file = %s", [known]
            ).fetchone()
            assert num_words == expected["num_words"] and lm == expected["LM"]
            assert num_words > ingest_module.analyze_document(known, xml=older)["document_attr"]["num_words"]
            document_id, doc_year = conn.execute(
                "SELECT id, year FROM public.documents WHERE source_file = %s", [known]
            ).fetchone()
            assert doc_year == YEAR

        # Backfill: one batch read from the table, on a connection that is not in autocommit
        with psycopg.connect(test_database_url) as conn:
            register_keyword_set(conn, set_name, "words.csv", 2)
            set_id = conn.execute(
                "SELECT id FROM public.keyword_sets WHERE name = %s", [set_name]
            ).fetchone()[0]
        conn = psycopg.connect(test_database_url)
        try:
            batch = [
                {
                    "document_id": document_id,
                    "source_file": known,
                    "keyword_sets": [(set_id, set_name)],
                    "location": ("db", known, None),
                },
                {
                    "document_id": document_id,
                    "source_file": f"9{tag}9_T.xml",
                    "keyword_sets": [(set_id, set_name)],
                    "location": ("db", f"9{tag}9_T.xml", None),
                },
            ]
            result = backfill_exposures.process_batch(
                batch, "TS", agents={set_name: MatchingAgent(keywords_path=str(words))}, conn=conn
            )
        finally:
            conn.close()  # the run must already be committed: closing rolls back anything open
        assert result["runs_written"] == 1
        assert [reason for _, _, reason in result["failures"]] == ["no published transcript"]
        with psycopg.connect(test_database_url) as conn:
            sentences = conn.execute(
                "SELECT total_sentences_in_document FROM public.exposure_runs "
                "WHERE document_id = %s AND keyword_set_id = %s",
                [document_id, set_id],
            ).fetchone()
            assert sentences == (expected["num_sentences"],)
    finally:
        with psycopg.connect(test_database_url) as conn:
            conn.execute("DELETE FROM public.documents WHERE source_file IN (%s, %s)", [known, new])
            conn.execute("DELETE FROM public.keyword_sets WHERE name = %s", [set_name])
            conn.execute("DELETE FROM public.transcript_files WHERE archive = %s", [archive.name])
