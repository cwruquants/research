"""
Direct Supabase Uploader
========================

Uploads analysis results directly to Supabase without creating intermediate files.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path


class SupabaseUploader:
    """Upload sentiment analysis results directly to Supabase."""
    
    def __init__(self, supabase_client, batch_size: int = 500):
        """
        Initialize the uploader.
        
        Args:
            supabase_client: Supabase client instance
            batch_size: Number of records to upload per batch
        """
        self.supabase = supabase_client
        self.batch_size = batch_size
        self.company_cache = {}  # (name, ticker) → company_id
        self.company_insert_buffer = []  # Buffer for batch inserts
        
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse dates like '25-Jul-02 01:15PM GMT' → ISO string."""
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
        """Load all existing companies from Supabase into cache."""
        all_companies = []
        page_size = 1000
        offset = 0
        
        while True:
            result = self.supabase.table("companies").select("id, name, ticker").range(offset, offset + page_size - 1).execute()
            if not result.data:
                break
            all_companies.extend(result.data)
            if len(result.data) < page_size:
                break
            offset += page_size
        
        for row in all_companies:
            self.company_cache[(row["name"], row["ticker"])] = row["id"]
        
        return len(self.company_cache)
    
    def _get_or_create_company(self, name: str, ticker: Optional[str]) -> Optional[int]:
        """
        Get company_id from cache or create new company.
        Uses buffered batch inserts for efficiency.
        
        Args:
            name: Company name
            ticker: Company ticker (can be None)
            
        Returns:
            company_id or None if creation failed
        """
        key = (name, ticker)
        
        # Check cache first
        if key in self.company_cache:
            return self.company_cache[key]
        
        # Add to insert buffer
        self.company_insert_buffer.append({"name": name, "ticker": ticker})
        
        # If buffer is full, flush it
        if len(self.company_insert_buffer) >= self.batch_size:
            self._flush_company_buffer()
        
        # After flush, check cache again
        return self.company_cache.get(key)
    
    def _flush_company_buffer(self):
        """Insert buffered companies and update cache."""
        if not self.company_insert_buffer:
            return
        
        # Insert companies
        result = self.supabase.table("companies").upsert(
            self.company_insert_buffer, 
            on_conflict="name,ticker"
        ).execute()
        
        # Update cache with returned IDs
        for row in result.data:
            self.company_cache[(row["name"], row["ticker"])] = row["id"]
        
        # Clear buffer
        self.company_insert_buffer = []
    
    def upload_document(self, data: Dict[str, Any], year: int) -> Dict[str, Any]:
        """
        Upload a single document's analysis results directly to Supabase.
        
        Args:
            data: Result from Analyst.analyze_document_memory()
            year: Year of the earnings call
            
        Returns:
            dict: Upload result with document_id and status
        """
        doc_attrs = data["doc_attrs"]
        doc_metadata = doc_attrs["document"]
        doc_attr = doc_attrs["document_attr"]
        
        # Get or create company
        company_name = str(doc_metadata.get("company_name", ""))
        company_ticker = doc_metadata.get("company_ticker")
        if company_ticker:
            company_ticker = str(company_ticker)
        
        if not company_name:
            return {"status": "skipped", "reason": "No company name"}
        
        company_id = self._get_or_create_company(company_name, company_ticker)
        
        # If company is not in cache yet (buffer not flushed), flush now
        if company_id is None:
            self._flush_company_buffer()
            company_id = self.company_cache.get((company_name, company_ticker))
        
        if not company_id:
            return {"status": "error", "reason": "Could not create company"}
        
        # Prepare document row
        source_file = str(doc_metadata.get("file_name", ""))
        if not source_file:
            return {"status": "skipped", "reason": "No source file"}
        
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
        
        # Upload document
        try:
            result = self.supabase.table("documents").upsert(
                [doc_row], 
                on_conflict="source_file"
            ).execute()
            
            if result.data:
                document_id = result.data[0]["id"]
                return {
                    "status": "success",
                    "document_id": document_id,
                    "company_id": company_id,
                    "source_file": source_file,
                }
            else:
                return {"status": "error", "reason": "No data returned from insert"}
                
        except Exception as e:
            return {"status": "error", "reason": str(e), "source_file": source_file}
    
    def finalize(self):
        """Flush any remaining buffered data."""
        self._flush_company_buffer()
