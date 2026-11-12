"""Read raw transcript files out of the vendor zips, ready to store in transcript_files.

Nothing here judges whether a file is a usable transcript: every vendor file that is
valid text is kept, stubs and withdrawn calls included. Only files the vendor did not
send (macOS resource files), empty files and files that cannot be stored as text are
left out, each with a reason.
"""

import hashlib
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Collection, Iterator, Optional, Tuple, Union

from research.transcripts.sources import _folder_year

MACOS_JUNK = "macOS resource file"
NOT_XML = "not an .xml file"
EMPTY = "empty file"
NOT_UTF8 = "not valid UTF-8 text"
HAS_NUL = "contains a NUL character, which Postgres text cannot hold"


@dataclass(frozen=True)
class TranscriptFile:
    """One raw file, with the few tag attributes needed to choose between revisions."""

    source_file: str
    sha256: bytes
    byte_length: int
    xml: str
    last_update: Optional[datetime]
    story_action: Optional[str]
    story_version: Optional[str]
    story_type: Optional[str]
    folder_year: Optional[int]
    archive: str
    member: str


@dataclass(frozen=True)
class Skipped:
    archive: str
    member: str
    reason: str


_EVENT_TAG = re.compile(r"<Event\b([^>]*)>")
_STORY_TAG = re.compile(r"<EventStory\b([^>]*)>")
_ATTRIBUTE = re.compile(r"""([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
# e.g. "Wednesday, January 9, 2019 at 4:23:16am GMT"
_LAST_UPDATE = re.compile(
    r"(\w+) (\d{1,2}), (\d{4}) at (\d{1,2}):(\d{2}):(\d{2})\s*(am|pm) GMT", re.IGNORECASE
)
_MONTHS = {
    name: number
    for number, name in enumerate(
        "january february march april may june july august september october november december".split(), 1
    )
}


def _tag_attributes(pattern: re.Pattern, xml: str) -> dict:
    """Attributes of the first tag matching pattern; read by regex so malformed XML still yields them."""
    tag = pattern.search(xml)
    if tag is None:
        return {}
    return {
        name: double if double is not None else single for name, double, single in _ATTRIBUTE.findall(tag[1])
    }


def parse_last_update(value: Optional[str]) -> Optional[datetime]:
    """The vendor's lastUpdate stamp as a UTC datetime, or None if it is missing or in another format."""
    stamp = _LAST_UPDATE.search(value or "")
    if stamp is None:
        return None
    month, day, year, hour, minute, second, half = stamp.groups()
    if month.lower() not in _MONTHS or not 1 <= int(hour) <= 12:
        return None
    hour24 = int(hour) % 12 + (12 if half.lower() == "pm" else 0)
    try:
        return datetime(
            int(year), _MONTHS[month.lower()], int(day), hour24, int(minute), int(second), tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _members(source: Path) -> Iterator[Tuple[str, Callable[[], bytes]]]:
    """(path inside the source, reader) for every file in a zip or a folder, in a stable order."""
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    yield info.filename, lambda info=info: archive.read(info)
    elif source.is_dir():
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            yield path.relative_to(source).as_posix(), path.read_bytes
    else:
        raise SystemExit(f"Source is neither a folder nor a zip file: {source}")


def read_source(
    source: Union[str, Path], already_loaded: Collection[str] = ()
) -> Iterator[Union[TranscriptFile, Skipped]]:
    """
    Every file in a zip or folder as a TranscriptFile, or a Skipped with the reason it
    cannot be stored. Members named in already_loaded are passed over without being read,
    so an interrupted load can resume.
    """
    source = Path(source)
    archive = source.name
    for member, read in _members(source):
        name = member.rsplit("/", 1)[-1]
        if "__MACOSX" in member.split("/") or name.startswith("._") or name == ".DS_Store":
            yield Skipped(archive, member, MACOS_JUNK)
            continue
        if not name.lower().endswith(".xml"):
            yield Skipped(archive, member, NOT_XML)
            continue
        if member in already_loaded:
            continue
        raw = read()
        if not raw:
            yield Skipped(archive, member, EMPTY)
            continue
        try:
            xml = raw.decode("utf-8")
        except UnicodeDecodeError:
            yield Skipped(archive, member, NOT_UTF8)
            continue
        if "\x00" in xml:
            yield Skipped(archive, member, HAS_NUL)
            continue
        event, story = _tag_attributes(_EVENT_TAG, xml), _tag_attributes(_STORY_TAG, xml)
        yield TranscriptFile(
            source_file=name,
            sha256=hashlib.sha256(raw).digest(),
            byte_length=len(raw),
            xml=xml,
            last_update=parse_last_update(event.get("lastUpdate")),
            story_action=story.get("action"),
            story_version=story.get("version"),
            story_type=story.get("storyType"),
            folder_year=_folder_year(("zip", archive, member)),
            archive=archive,
            member=member,
        )
