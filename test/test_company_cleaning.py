"""Checks that PostgresUploader cleans company names and tickers before writing rows."""

from contextlib import contextmanager
import threading

import pytest

from src.analysis.postgres_uploader import PostgresUploader
from src.document.decompose_transcript import clean_text, clean_ticker


class _FakeCursor:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class _FakeConnection:
    """Records every statement and answers RETURNING id with an incrementing id."""

    def __init__(self, calls):
        self.calls = calls

    def execute(self, statement, params=None):
        self.calls.append((statement, params))
        return _FakeCursor((len(self.calls),))


class _FakePool:
    def __init__(self):
        self.calls = []

    @contextmanager
    def connection(self):
        yield _FakeConnection(self.calls)


def _uploader():
    """An uploader wired to a fake pool, so no database is opened."""
    uploader = PostgresUploader.__new__(PostgresUploader)
    uploader.batch_size = 500
    uploader.pool = _FakePool()
    uploader.company_cache = {}
    uploader._cache_lock = threading.Lock()
    return uploader


def _document(**metadata):
    document = {"file_name": "1020948_T.xml", "start_date": "29-Apr-05 01:00PM GMT"}
    document.update(metadata)
    return {"doc_attrs": {"document": document, "document_attr": {"sentiment": 0.1}}}


def _company_params(uploader):
    return [params for statement, params in uploader.pool.calls if "public.companies" in statement]


def _document_row(uploader):
    statement, params = next(
        (statement, params) for statement, params in uploader.pool.calls if "public.documents" in statement
    )
    columns = statement.split("(", 1)[1].split(")", 1)[0].split(", ")
    return dict(zip(columns, params))


@pytest.mark.parametrize("raw", [None, "", "   ", "\n  ", "\\n  ", "\\n", " \\t\\r ", "\t\n"])
def test_clean_ticker_empty_forms_become_none(raw):
    assert clean_ticker(raw) is None


@pytest.mark.parametrize("raw, expected", [("ALEX", "ALEX"), ("  LLOY.L\n", "LLOY.L"), ("**TEL", "**TEL")])
def test_clean_ticker_keeps_real_tickers(raw, expected):
    assert clean_ticker(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    (None, None), ("", None), ("  \n ", None),
    ("QUAKER OATS                   ", "QUAKER OATS"), ("Lexent Inc. ", "Lexent Inc."),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize("ticker", [None, "\n  ", "\\n  ", "   "])
def test_empty_tickers_are_stored_as_null(ticker):
    uploader = _uploader()
    result = uploader.upload_document(_document(company_name="Lexent Inc. ", company_ticker=ticker), 2005)
    assert result["status"] == "success"
    assert _company_params(uploader) == [("Lexent Inc.", None)]


def test_padded_name_and_ticker_are_stripped():
    uploader = _uploader()
    uploader.upload_document(_document(company_name="  Alexander & Baldwin Inc\n", company_ticker=" ALEX "), 2005)
    assert _company_params(uploader) == [("Alexander & Baldwin Inc", "ALEX")]


def test_numeric_ticker_from_toml_is_kept_as_text():
    uploader = _uploader()
    uploader.upload_document(_document(company_name="Toyota Motor Corp", company_ticker=7203), 2005)
    assert _company_params(uploader) == [("Toyota Motor Corp", "7203")]


@pytest.mark.parametrize("name", [None, "", "  \n  "])
def test_missing_name_is_skipped_not_stored_as_none_string(name):
    uploader = _uploader()
    result = uploader.upload_document(_document(company_name=name, company_ticker="ALEX"), 2005)
    assert result == {"status": "skipped", "reason": "No company name"}
    assert uploader.pool.calls == []


def test_cleaned_variants_share_one_company_row():
    uploader = _uploader()
    for i, (name, ticker) in enumerate([("Lexent Inc.", None), ("Lexent Inc. ", "\\n  "), ("Lexent Inc.\n", "\n  ")]):
        uploader.upload_document(_document(company_name=name, company_ticker=ticker, file_name=f"{i}.xml"), 2005)
    assert _company_params(uploader) == [("Lexent Inc.", None)]
    assert uploader.company_cache == {("Lexent Inc.", None): 1}


def test_missing_city_and_title_are_null_not_none_string():
    uploader = _uploader()
    uploader.upload_document(_document(company_name="Lexent Inc.", city=None, event_title="  \n"), 2005)
    row = _document_row(uploader)
    assert row["city"] is None
    assert row["event_title"] is None


def test_city_and_title_are_stripped():
    uploader = _uploader()
    uploader.upload_document(
        _document(company_name="Lexent Inc.", city=" Honolulu\n", event_title=" Q1 2005 Earnings Call "), 2005
    )
    row = _document_row(uploader)
    assert row["city"] == "Honolulu"
    assert row["event_title"] == "Q1 2005 Earnings Call"
