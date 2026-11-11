"""Write companies and documents rows to Postgres."""

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psycopg
from psycopg_pool import ConnectionPool

from research.db.document_year import resolve_document_year
from research.transcripts.parse import clean_text, clean_ticker


def _as_str(value: Any) -> Optional[str]:
    """str() for metadata values (TOML may hold numeric tickers) that keeps None as None."""
    return None if value is None else str(value)


class PostgresUploader:
    """Upsert companies and documents rows from analyze_document() output, or update only their measures."""

    def __init__(self, database_url: str, batch_size: int = 500, max_connections: int = 4):
        self.batch_size = batch_size
        self.pool = ConnectionPool(conninfo=database_url, min_size=1, max_size=max_connections, open=True)
        self.company_cache = {}
        self._cache_lock = threading.Lock()

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse dates like '25-Jul-02 01:15PM GMT' to an ISO string."""
        if not date_str:
            return None
        for tz in [" GMT", " EST", " EDT", " CST", " CDT", " PST", " PDT"]:
            date_str = date_str.replace(tz, "")
        date_str = date_str.strip()
        for fmt in ["%d-%b-%y %I:%M%p", "%d-%b-%y %I:%M %p", "%d-%b-%y %H:%M"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                continue
        return None

    def _safe_int(self, val) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    def _safe_float(self, val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def initialize_company_cache(self):
        """Load every existing company in one query."""
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT id, name, ticker FROM public.companies").fetchall()
        with self._cache_lock:
            self.company_cache = {(name, ticker): company_id for company_id, name, ticker in rows}
            return len(self.company_cache)

    def _get_or_create_company(self, name: str, ticker: Optional[str]) -> int:
        key = (name, ticker)
        with self._cache_lock:
            company_id = self.company_cache.get(key)
        if company_id is not None:
            return company_id

        with self.pool.connection() as conn:
            company_id = conn.execute(
                """INSERT INTO public.companies (name, ticker) VALUES (%s, %s)
                   ON CONFLICT (name, ticker) DO UPDATE SET name = EXCLUDED.name
                   RETURNING id""",
                (name, ticker),
            ).fetchone()[0]
        with self._cache_lock:
            self.company_cache[key] = company_id
        return company_id

    # documents column -> (key in analyze_document's measures, converter)
    _MEASURES = {
        "num_sentences": ("num_sentences", "int"),
        "num_words": ("num_words", "int"),
        "sentiment": ("sentiment", "float"),
        "ml_score": ("ML", "float"),
        "lm_score": ("LM", "float"),
        "flesch_reading_ease": ("flesch_reading_ease", "float"),
        "flesch_kincaid_grade": ("flesch_kincaid_grade", "float"),
        "smog_index": ("smog_index", "float"),
        "coleman_liau_index": ("coleman_liau_index", "float"),
        "automated_readability": ("automated_readability_index", "float"),
        "dale_chall_score": ("dale_chall_readability_score", "float"),
        "difficult_words": ("difficult_words", "int"),
        "linsear_write_formula": ("linsear_write_formula", "float"),
        "gunning_fog": ("gunning_fog", "float"),
        "text_standard": ("text_standard", "text"),
    }

    def _measure_columns(self, doc_attr: Dict[str, Any], only_present: bool = False) -> Dict[str, Any]:
        """
        The text-measure columns of a documents row. With only_present, a measure that was
        not computed (sentiment when ingest ran without --sentiment) is left out, so an
        update keeps the stored value instead of writing NULL over it.
        """
        row = {}
        for column, (key, kind) in self._MEASURES.items():
            if only_present and key not in doc_attr:
                continue
            value = doc_attr.get(key)
            if kind == "int":
                row[column] = self._safe_int(value)
            elif kind == "float":
                row[column] = self._safe_float(value)
            else:
                row[column] = str(value) if value not in (None, "") else None
        return row

    def _update_measures_once(self, data: Dict[str, Any]) -> Dict[str, Any]:
        doc_attrs = data["doc_attrs"]
        source_file = str(doc_attrs["document"].get("file_name", ""))
        if not source_file:
            return {"status": "skipped", "reason": "No source file"}
        row = self._measure_columns(doc_attrs["document_attr"], only_present=True)
        if not row:
            return {"status": "skipped", "reason": "No measures to write", "source_file": source_file}
        # Column names come only from _MEASURES; values are parameters.
        assignments = ", ".join(f"{column} = %s" for column in row)
        statement = f"UPDATE public.documents SET {assignments} WHERE source_file = %s RETURNING id, year"
        with self.pool.connection() as conn:
            found = conn.execute(statement, (*row.values(), source_file)).fetchone()
        if found is None:
            return {"status": "skipped", "reason": "No documents row to update", "source_file": source_file}
        return {
            "status": "success",
            "document_id": found[0],
            "source_file": source_file,
            "year": found[1],
            "year_fallback": False,
        }

    def _upload_once(self, data: Dict[str, Any], year: int) -> Dict[str, Any]:
        doc_attrs = data["doc_attrs"]
        doc_metadata = doc_attrs["document"]
        doc_attr = doc_attrs["document_attr"]

        # Cleaned here because metadata may come from TOML files written before
        # the parser stripped empty tags, and str(None) would store "None".
        company_name = clean_text(_as_str(doc_metadata.get("company_name")))
        company_ticker = clean_ticker(_as_str(doc_metadata.get("company_ticker")))
        if not company_name:
            return {"status": "skipped", "reason": "No company name"}

        source_file = str(doc_metadata.get("file_name", ""))
        if not source_file:
            return {"status": "skipped", "reason": "No source file"}

        company_id = self._get_or_create_company(company_name, company_ticker)
        date_str = str(doc_metadata.get("start_date", "")) if doc_metadata.get("start_date") else ""
        event_date = self._parse_date(date_str) if date_str else None
        doc_year, year_fallback = resolve_document_year(event_date, year)

        doc_row = {
            "company_id": company_id,
            "year": doc_year,
            "source_file": source_file,
            "event_title": clean_text(_as_str(doc_metadata.get("event_title"))),
            "event_date": event_date,
            "city": clean_text(_as_str(doc_metadata.get("city"))),
            **self._measure_columns(doc_attr),
        }

        # Column names come only from the fixed keys above; values are parameters.
        columns = ", ".join(doc_row)
        placeholders = ", ".join(["%s"] * len(doc_row))
        updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in doc_row if column != "source_file")
        statement = (
            f"INSERT INTO public.documents ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT (source_file) DO UPDATE SET {updates} RETURNING id"
        )
        with self.pool.connection() as conn:
            document_id = conn.execute(statement, tuple(doc_row.values())).fetchone()[0]
        return {
            "status": "success",
            "document_id": document_id,
            "company_id": company_id,
            "source_file": source_file,
            "year": doc_year,
            "year_fallback": year_fallback,
        }

    def upload_document(self, data: Dict[str, Any], year: int) -> Dict[str, Any]:
        """Upload one document, retrying a transient connection failure."""
        return self._with_retries(lambda: self._upload_once(data, year), data)

    def update_measures(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overwrite only the text measures of an existing documents row, matched on
        source_file. Company, year, title, date and city are never touched, and no row or
        company is ever created: a file with no row is skipped.
        """
        return self._with_retries(lambda: self._update_measures_once(data), data)

    def _with_retries(self, attempt_once, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run one write, retrying a transient connection failure."""
        source_file = str(data.get("doc_attrs", {}).get("document", {}).get("file_name", ""))
        for attempt in range(4):
            try:
                return attempt_once()
            except psycopg.OperationalError as exc:
                if attempt == 3:
                    return {"status": "error", "reason": str(exc), "source_file": source_file}
                time.sleep((1, 3, 10)[attempt])
            except Exception as exc:
                return {"status": "error", "reason": str(exc), "source_file": source_file}
        raise AssertionError("unreachable")

    def finalize(self):
        """Close the connection pool after all uploads finish."""
        self.pool.close()
