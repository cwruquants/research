"""--measures-only against a throwaway Postgres (TEST_DATABASE_URL); rows are removed afterwards."""

from uuid import uuid4

import psycopg
import pytest

from research.db.documents import PostgresUploader


def _document(company, source_file, title, measures, event_type=(None, None)):
    return {
        "doc_attrs": {
            "document": {
                "company_name": company,
                "company_ticker": "NEW.N",
                "file_name": source_file,
                "event_title": title,
                "start_date": "02-Feb-21 1:30PM GMT",
                "city": "Elsewhere",
                "event_type_id": event_type[0],
                "event_type": event_type[1],
            },
            "document_attr": measures,
        }
    }


@pytest.mark.db
def test_update_measures_changes_only_the_measures(test_database_url):
    prefix = f"measures_test_{uuid4().hex}"
    source_file = f"{prefix}_T.xml"
    uploader = PostgresUploader(test_database_url, max_connections=1)
    try:
        uploader.initialize_company_cache()
        first = uploader.upload_document(
            _document(
                f"{prefix} Original Inc",
                source_file,
                "Original title",
                {
                    "num_words": 100,
                    "sentiment": 0.2,
                    "LM": -3,
                    "flesch_reading_ease": 50.0,
                },
            ),
            2017,
        )
        assert first["status"] == "success"
        with psycopg.connect(test_database_url) as conn:
            before = conn.execute(
                "SELECT company_id, year, event_title, event_date, city FROM public.documents "
                "WHERE source_file = %s",
                [source_file],
            ).fetchone()
            companies_before = conn.execute("SELECT count(*) FROM public.companies").fetchone()[0]

        # The file now names a different company and title; sentiment was not computed
        result = uploader.update_measures(
            _document(
                f"{prefix} Renamed Owner",
                source_file,
                "Revised title",
                {
                    "num_words": 250,
                    "num_sentences": 20,
                    "flesch_reading_ease": 60.0,
                    "LM_positive": 7,
                    "LM_negative": 9,
                },
                event_type=("1", "Earnings Conference Call"),
            )
        )
        assert result["status"] == "success" and result["document_id"] == first["document_id"]

        missing = uploader.update_measures(_document("x", f"{prefix}_none_T.xml", "t", {"num_words": 1}))
        assert missing["status"] == "skipped"

        with psycopg.connect(test_database_url) as conn:
            after = conn.execute(
                "SELECT company_id, year, event_title, event_date, city FROM public.documents "
                "WHERE source_file = %s",
                [source_file],
            ).fetchone()
            measures = conn.execute(
                "SELECT num_words, num_sentences, flesch_reading_ease, sentiment, lm_score "
                "FROM public.documents WHERE source_file = %s",
                [source_file],
            ).fetchone()
            assert after == before  # metadata untouched
            assert measures == (250, 20, 60.0, 0.2, -3.0)  # new measures; sentiment kept
            filled = conn.execute(
                "SELECT lm_positive, lm_negative, event_type_id, event_type FROM public.documents "
                "WHERE source_file = %s",
                [source_file],
            ).fetchone()
            assert filled == (7, 9, 1, "Earnings Conference Call")  # the file's event type is filled in
            assert conn.execute("SELECT count(*) FROM public.companies").fetchone()[0] == companies_before
            assert (
                conn.execute(
                    "SELECT count(*) FROM public.documents WHERE source_file LIKE %s", [f"{prefix}%"]
                ).fetchone()[0]
                == 1
            )
    finally:
        with psycopg.connect(test_database_url) as conn:
            conn.execute("DELETE FROM public.documents WHERE source_file LIKE %s", [f"{prefix}%"])
            conn.execute("DELETE FROM public.companies WHERE name LIKE %s", [f"{prefix}%"])
        uploader.finalize()
