"""Integration checks against a dedicated local test database."""

from concurrent.futures import ThreadPoolExecutor
import os
from uuid import uuid4

import psycopg
import pytest

from src.analysis.postgres_uploader import PostgresUploader


def _document(company: str, source_file: str, score: float):
    return {
        "doc_attrs": {
            "document": {
                "company_name": company,
                "company_ticker": None,
                "file_name": source_file,
                "event_title": "Test call",
            },
            "document_attr": {"sentiment": score, "num_words": 100},
        }
    }


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not set")
def test_upsert_and_concurrent_uploads():
    database_url = os.environ["TEST_DATABASE_URL"]
    prefix = f"uploader_test_{uuid4().hex}"
    company = f"{prefix}_company"
    uploader = PostgresUploader(database_url, max_connections=4)
    try:
        uploader.initialize_company_cache()
        first_file = f"{prefix}_first.xml"
        first = uploader.upload_document(_document(company, first_file, 0.2), 2020)
        second = uploader.upload_document(_document(company, first_file, 0.8), 2020)
        assert first["status"] == second["status"] == "success"
        assert first["document_id"] == second["document_id"]

        files = [f"{prefix}_{n}.xml" for n in range(4)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda path: uploader.upload_document(
                _document(company, path, 0.5), 2020), files))
        assert all(result["status"] == "success" for result in results)

        with psycopg.connect(database_url) as conn:
            company_count = conn.execute(
                "SELECT count(*) FROM public.companies WHERE name = %s", (company,)
            ).fetchone()[0]
            document_count = conn.execute(
                """SELECT count(*) FROM public.documents d
                   JOIN public.companies c ON c.id = d.company_id
                   WHERE c.name = %s""", (company,)
            ).fetchone()[0]
            score = conn.execute(
                "SELECT sentiment FROM public.documents WHERE source_file = %s",
                (first_file,),
            ).fetchone()[0]
        assert company_count == 1
        assert document_count == 5
        assert score == 0.8
    finally:
        uploader.finalize()
