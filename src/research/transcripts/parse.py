import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Union

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


def _body_lines(xml_source) -> list:
    """Lines of the <Body> text of a transcript XML (a path or a binary file object)."""
    root = ET.parse(xml_source).getroot()
    body_elem = root.find(".//Body")
    body = body_elem.text if body_elem is not None else None
    return body.splitlines() if body is not None else []


def _presentation_from_lines(lines: list) -> str:
    # Flags and buffers
    in_presentation = False
    presentation_lines = []

    for line in lines:
        if re.search(r"^\s*Presentation\s*$", line):
            in_presentation = True
            continue
        if re.search(r"^\s*(Questions and Answers|Q&A|Operator)\s*$", line, re.IGNORECASE):
            break
        if in_presentation:
            presentation_lines.append(line)

    return "\n".join(presentation_lines).strip()


def _qa_from_lines(lines: list) -> str:
    in_qa = False
    qa_lines = []

    for line in lines:
        if not in_qa and re.search(r"^\s*(Questions and Answers|Q&A)\s*$", line, re.IGNORECASE):
            in_qa = True
            continue
        if in_qa:
            qa_lines.append(line)

    return "\n".join(qa_lines).strip()


# Section headings in the transcript body sit between a ==== rule and a ---- rule
_PARTICIPANT_HEADINGS = {"corporate participants", "conference call participants"}


def _spoken_sections_from_lines(lines: list) -> str:
    """
    Fallback for transcripts without the usual "Presentation" / "Questions and Answers"
    headings (e.g. older calls with a single "Transcript" section, or headings like
    "PRESENTATION", "q and a", "Q-and-A"): keep every section after the header block
    except the participant lists, dropping the heading lines themselves.
    """
    kept = []
    in_spoken_section = False
    i = 0
    while i < len(lines):
        is_heading = (
            0 < i < len(lines) - 1
            and lines[i].strip()
            and re.fullmatch(r"=+", lines[i - 1].strip())
            and re.fullmatch(r"[-=]{5,}", lines[i + 1].strip())
        )
        if is_heading:
            in_spoken_section = lines[i].strip().lower() not in _PARTICIPANT_HEADINGS
            if kept and re.fullmatch(r"=+", kept[-1].strip()):
                kept.pop()  # the ==== rule above the heading
            # Skip the heading but keep the ---- rule below it: _strip_speaker_lines
            # expects each speaker name framed by rules, as in the Presentation section
            i += 1
            continue
        if in_spoken_section:
            kept.append(lines[i])
        i += 1
    return "\n".join(kept).strip()


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
    The spoken text of a transcript: its Presentation and Q&A with speaker lines and rules
    removed, or every spoken section if it has no such headings. Word counts, readability,
    sentiment and keyword matching all start from this text, so they describe the same words.
    xml_source may be a path or a binary file object (e.g. a zip member).
    """
    lines = _body_lines(xml_source)
    # Each section is cleaned on its own: see _strip_speaker_lines
    text = (
        _strip_speaker_lines(_presentation_from_lines(lines))
        + "\n\n"
        + _strip_speaker_lines(_qa_from_lines(lines))
    ).strip()
    if not text:
        # No Presentation / Q&A headings found: fall back to the generic section layout
        text = _strip_speaker_lines(_spoken_sections_from_lines(lines))
    return text


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
