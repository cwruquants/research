"""Upload document analysis directly to the local PostgreSQL database."""

from datetime import datetime, timezone
import threading
import time
from typing import Any, Dict, Optional

import psycopg
from psycopg_pool import ConnectionPool


class PostgresUploader:
    """Upload sentiment analysis results with the Supabase uploader's interface."""

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

    def _upload_once(self, data: Dict[str, Any], year: int) -> Dict[str, Any]:
        doc_attrs = data["doc_attrs"]
        doc_metadata = doc_attrs["document"]
        doc_attr = doc_attrs["document_attr"]

        company_name = str(doc_metadata.get("company_name", ""))
        company_ticker = doc_metadata.get("company_ticker")
        if company_ticker:
            company_ticker = str(company_ticker)
        if not company_name:
            return {"status": "skipped", "reason": "No company name"}

        source_file = str(doc_metadata.get("file_name", ""))
        if not source_file:
            return {"status": "skipped", "reason": "No source file"}

        company_id = self._get_or_create_company(company_name, company_ticker)
        date_str = str(doc_metadata.get("start_date", "")) if doc_metadata.get("start_date") else ""
        event_date = self._parse_date(date_str) if date_str else None

        doc_row = {
            "company_id": company_id,
            "year": year,
            "source_file": source_file,
            "event_title": str(doc_metadata.get("event_title", "")) or None,
            "event_date": event_date,
            "city": str(doc_metadata.get("city", "")) or None,
            "num_sentences": self._safe_int(doc_attr.get("num_sentences")),
            "num_words": self._safe_int(doc_attr.get("num_words")),
            "sentiment": self._safe_float(doc_attr.get("sentiment")),
            "ml_score": self._safe_float(doc_attr.get("ML")),
            "lm_score": self._safe_float(doc_attr.get("LM")),
            "hiv4_score": self._safe_float(doc_attr.get("HIV4")),
            "flesch_reading_ease": self._safe_float(doc_attr.get("flesch_reading_ease")),
            "flesch_kincaid_grade": self._safe_float(doc_attr.get("flesch_kincaid_grade")),
            "smog_index": self._safe_float(doc_attr.get("smog_index")),
            "coleman_liau_index": self._safe_float(doc_attr.get("coleman_liau_index")),
            "automated_readability": self._safe_float(doc_attr.get("automated_readability_index")),
            "dale_chall_score": self._safe_float(doc_attr.get("dale_chall_readability_score")),
            "difficult_words": self._safe_int(doc_attr.get("difficult_words")),
            "linsear_write_formula": self._safe_float(doc_attr.get("linsear_write_formula")),
            "gunning_fog": self._safe_float(doc_attr.get("gunning_fog")),
            "text_standard": str(doc_attr.get("text_standard", "")) or None,
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
        return {"status": "success", "document_id": document_id,
                "company_id": company_id, "source_file": source_file}

    def upload_document(self, data: Dict[str, Any], year: int) -> Dict[str, Any]:
        """Upload one document, retrying a transient connection failure."""
        source_file = str(data.get("doc_attrs", {}).get("document", {}).get("file_name", ""))
        for attempt in range(4):
            try:
                return self._upload_once(data, year)
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
