import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytz
from dateutil import parser as date_parser

from research.transcripts.models import DocumentAttr

__all__ = [
    "get_company_metadata",
    "clean_text",
    "clean_ticker",
    "spoken_text",
    "load_document",
]

# Headlines look like "Edited Transcript of LLOY.L M&A conference call or presentation ...".
# The ticker token is upper-case letters/digits plus RIC punctuation (., ^ delisting codes);
# anything else (e.g. a company name) is not treated as a ticker.
HEADLINE_TICKER_RE = re.compile(r"Transcript of ([A-Z0-9][A-Z0-9.^*&\-]*) ")


def clean_text(value: Optional[str]) -> Optional[str]:
    """Strip whitespace; empty tags become None instead of '' or '\\n  '."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def clean_ticker(value: Optional[str]) -> Optional[str]:
    """Like clean_text, but also treat escaped whitespace (a literal '\\n') as empty."""
    value = clean_text(value)
    if value is None or not re.sub(r"\\[nrt]|\s", "", value):
        return None
    return value


def _headline_ticker(headline: Optional[str]) -> Optional[str]:
    """Ticker as printed in the headline, i.e. as of the call date."""
    match = HEADLINE_TICKER_RE.search(headline or "")
    return match.group(1) if match else None


def get_company_metadata(xml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parses company metadata (name, ticker, date, quarter, event type) from the earnings call XML.
    Standardizes the start date to GMT.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    event_title = root.findtext("eventTitle", "")

    match = re.search(r"(Q[1-4])\s+(\d{4})", event_title)
    quarter, year = (match.group(1), match.group(2)) if match else (None, None)

    # Parse and standardize startDate to GMT
    start_date_raw = root.findtext("startDate")
    start_date_gmt = None
    if start_date_raw:
        try:
            # Parse the date string with timezone info
            dt = date_parser.parse(start_date_raw)
            # Convert to GMT/UTC
            if dt.tzinfo is None:
                # If no timezone, assume GMT
                dt = pytz.UTC.localize(dt)
            else:
                dt = dt.astimezone(pytz.UTC)
            # Format as standard GMT string
            start_date_gmt = dt.strftime("%d-%b-%y %I:%M%p GMT")
        except Exception:
            # If parsing fails, keep original
            start_date_gmt = start_date_raw

    data = {
        "eventTitle": event_title,
        "city": clean_text(root.findtext("city")),
        "companyName": clean_text(root.findtext("companyName")),
        "companyTicker": clean_ticker(root.findtext("companyTicker")),
        "headlineTicker": _headline_ticker(root.findtext(".//Headline")),
        "eventTypeId": clean_text(root.get("eventTypeId")),
        "eventTypeName": clean_text(root.get("eventTypeName")),
        "startDate": start_date_gmt,
        "quarter": quarter,
        "year": year,
    }

    return data


# Only the XML entities and numeric references, and only with their semicolon, so an
# abbreviation such as "P&L" or "AT&T" is never read as an entity
_ESCAPED_ENTITY = re.compile(r"&(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);")


def _unescape(text: str) -> str:
    """
    Undo the vendor's extra escaping of the body: "R&D" is still "R&amp;D" after the XML
    parser, and a few files escape a quote mark once more ("&amp;quot;").
    """
    for _ in range(3):
        unescaped = _ESCAPED_ENTITY.sub(lambda match: html.unescape(match.group(0)), text)
        if unescaped == text:
            break
        text = unescaped
    return text


def _body_lines(xml_source) -> list:
    """Lines of the <Body> text of a transcript XML (a path or a binary file object)."""
    root = ET.parse(xml_source).getroot()
    body_elem = root.find(".//Body")
    body = body_elem.text if body_elem is not None else None
    return _unescape(body).splitlines() if body is not None else []


# A section heading sits between a ==== rule and a ---- or ==== rule
_PARTICIPANT_HEADINGS = {"corporate participants", "conference call participants"}


def _spoken_sections(lines: list) -> List[str]:
    """
    The body's sections, one string each, without the participant lists and without the
    heading lines themselves.

    Sections are found by their rules, not by what the heading says. The vendor's headings
    vary ("Presentation", "Questions and Answers", "Transcript", "q and a", "Q-and-A",
    "Question and Answers", ...) and a call can repeat them, but every section other than
    the two participant lists is speech. Text before the first heading is the header block.
    """
    sections: List[List[str]] = []
    current: Optional[List[str]] = None  # None in the header block and in a participant list
    for i, line in enumerate(lines):
        is_heading = (
            0 < i < len(lines) - 1
            and line.strip()
            and re.fullmatch(r"=+", lines[i - 1].strip())
            and re.fullmatch(r"[-=]{5,}", lines[i + 1].strip())
        )
        if is_heading:
            if current:
                current.pop()  # the ==== rule above this heading
            current = None
            if line.strip().lower() not in _PARTICIPANT_HEADINGS:
                # The rule below the heading stays: it opens the first speaker block
                current = []
                sections.append(current)
        elif current is not None:
            current.append(line)
    return ["\n".join(section) for section in sections]


def _strip_speaker_lines(section: str) -> str:
    """
    One spoken section with its speaker blocks (rule, speaker name, rule) removed.

    Every rule line is taken to open a speaker block, so the two lines after it are dropped
    unread. Pass one section at a time, starting at its first speaker block: a rule that
    frames anything else (a section heading, say) leaves the skip one line off, and in a
    transcript with no blank lines between turns it then stays off to the end, keeping the
    speaker names and dropping what they said.
    """
    lines = section.splitlines()
    kept = []
    i = 0
    while i < len(lines):
        if re.fullmatch(r"[-=]{5,}", lines[i].strip()):
            i += 3  # rule, speaker name, rule
            continue
        kept.append(lines[i])
        i += 1
    return "\n".join(kept).strip()


def spoken_text(xml_source) -> str:
    """
    The spoken text of a transcript: every section except the participant lists (usually a
    Presentation and a Q&A), with speaker lines and rules removed and a blank line between
    sections. Word counts, readability, sentiment and keyword matching all start from this
    text, so they describe the same words.
    xml_source may be a path or a binary file object (e.g. a zip member).
    """
    # Each section is cleaned on its own: see _strip_speaker_lines
    cleaned = (_strip_speaker_lines(section) for section in _spoken_sections(_body_lines(xml_source)))
    return "\n\n".join(text for text in cleaned if text)


def load_document(file_path) -> DocumentAttr:
    """
    Load an XML earnings call transcript as a DocumentAttr of its spoken text (see
    spoken_text). file_path may be a path or a binary file object (e.g. a zip member).
    An unreadable file gives an empty document.
    """
    try:
        return DocumentAttr(document=spoken_text(file_path))
    except Exception as e:
        print(f"Error loading document: {e}")
        return DocumentAttr(document="")
