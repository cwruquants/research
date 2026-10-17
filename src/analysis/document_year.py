"""Year stored on a document row, shared by the Supabase and Postgres uploaders."""

from datetime import datetime
from typing import Optional, Tuple


def resolve_document_year(event_date: Optional[str], fallback_year: int) -> Tuple[int, bool]:
    """
    Return the year to store for a document, and whether the fallback was used.

    The year is the calendar year of the call's start date in UTC, so a
    transcript filed in a later year's batch folder keeps its own year.
    Transcripts with no parseable start date take the batch year instead.

    Args:
        event_date: ISO timestamp from the uploader's ``_parse_date``, or None
        fallback_year: Year passed to the batch run

    Returns:
        Tuple[int, bool]: (year, True if fallback_year was used)
    """
    if event_date:
        return datetime.fromisoformat(event_date).year, False
    return int(fallback_year), True
