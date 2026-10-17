"""Checks that uploaders store the call-date year rather than the batch year."""

from contextlib import contextmanager
from pathlib import Path
import threading

import pytest

from src.analysis.document_year import resolve_document_year
from src.analysis.supabase_uploader import SupabaseUploader

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "earnings_calls" / "2016"


def _document(start_date, source_file="1020948_T.xml"):
    return {
        "doc_attrs": {
            "document": {
                "company_name": "Alexander & Baldwin Inc",
                "company_ticker": "ALEX",
                "file_name": source_file,
                "event_title": "Q1 2005 Alexander & Baldwin Earnings Conference Call",
                "start_date": start_date,
            },
            "document_attr": {"sentiment": 0.1, "num_words": 100},
        }
    }


class _FakeSupabase:
    """Records the rows passed to documents.upsert()."""

    def __init__(self):
        self.rows = []

    def table(self, name):
        assert name == "documents"
        return self

    def upsert(self, rows, on_conflict):
        assert on_conflict == "source_file"
        self.rows.extend(rows)
        return self

    def execute(self):
        return type("Result", (), {"data": [{"id": len(self.rows)}]})()


def _supabase_uploader():
    client = _FakeSupabase()
    uploader = SupabaseUploader(client)
    uploader.company_cache[("Alexander & Baldwin Inc", "ALEX")] = 7
    return uploader, client


def test_resolve_uses_event_date_year():
    assert resolve_document_year("2005-04-29T13:00:00+00:00", 2020) == (2005, False)


@pytest.mark.parametrize("event_date", [None, ""])
def test_resolve_falls_back_without_event_date(event_date):
    assert resolve_document_year(event_date, 2021) == (2021, True)


@pytest.mark.parametrize(
    "start_date, expected",
    [
        ("29-Apr-05 01:00PM GMT", 2005),
        ("31-Dec-20 11:59PM GMT", 2020),
        ("01-Jan-21 12:00AM GMT", 2021),
    ],
)
def test_supabase_uploader_stores_call_year(start_date, expected):
    uploader, client = _supabase_uploader()
    result = uploader.upload_document(_document(start_date), 1999)
    assert result["status"] == "success"
    assert client.rows[0]["year"] == expected
    assert result["year"] == expected
    assert result["year_fallback"] is False


@pytest.mark.parametrize("start_date", [None, "not a date"])
def test_supabase_uploader_falls_back_to_batch_year(start_date):
    uploader, client = _supabase_uploader()
    result = uploader.upload_document(_document(start_date), 2021)
    assert client.rows[0]["event_date"] is None
    assert client.rows[0]["year"] == 2021
    assert result["year_fallback"] is True


def test_postgres_uploader_stores_call_year():
    pytest.importorskip("psycopg")
    pytest.importorskip("psycopg_pool")
    from src.analysis.postgres_uploader import PostgresUploader

    statements = []

    class _Conn:
        def execute(self, statement, params):
            statements.append((statement, params))
            return type("Cursor", (), {"fetchone": lambda self: (11,)})()

    class _Pool:
        @contextmanager
        def connection(self):
            yield _Conn()

    uploader = PostgresUploader.__new__(PostgresUploader)
    uploader.pool = _Pool()
    uploader.company_cache = {("Alexander & Baldwin Inc", "ALEX"): 7}
    uploader._cache_lock = threading.Lock()

    dated = uploader.upload_document(_document("29-Apr-05 01:00PM GMT"), 2020)
    undated = uploader.upload_document(_document(None, "undated.xml"), 2020)

    assert (dated["year"], dated["year_fallback"]) == (2005, False)
    assert (undated["year"], undated["year_fallback"]) == (2020, True)
    year_index = statements[0][0].split("(")[1].split(")")[0].split(", ").index("year")
    assert statements[0][1][year_index] == 2005
    assert statements[1][1][year_index] == 2020


@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="sample transcripts not present")
def test_sample_transcripts_round_trip_to_their_own_year():
    """Parser output fed through the uploader gives the startDate's year."""
    from src.document.decompose_transcript import get_company_metadata

    uploader = SupabaseUploader(None)
    files = sorted(SAMPLE_DIR.glob("*.xml"))
    assert files
    for path in files:
        start_date = get_company_metadata(path)["startDate"]
        event_date = uploader._parse_date(start_date)
        assert event_date is not None, (path.name, start_date)
        year, fallback = resolve_document_year(event_date, 1999)
        assert not fallback
        assert year == int(event_date[:4])
        assert year in (2015, 2016, 2017), (path.name, start_date)


def test_offset_start_date_uses_utc_year(tmp_path):
    """An 8pm US Eastern call on Dec 31 is already the next year in UTC."""
    from src.document.decompose_transcript import get_company_metadata

    xml = tmp_path / "offset_T.xml"
    xml.write_text(
        "<Event><eventTitle>Q4 2020 Test Earnings Call</eventTitle>"
        "<startDate>2020-12-31T20:00:00-05:00</startDate></Event>"
    )
    uploader, client = _supabase_uploader()
    start_date = get_company_metadata(xml)["startDate"]
    result = uploader.upload_document(_document(start_date), 2020)
    assert start_date == "01-Jan-21 01:00AM GMT"
    assert client.rows[0]["year"] == 2021
    assert result["year_fallback"] is False
