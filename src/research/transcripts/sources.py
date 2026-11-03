"""Find transcript XMLs by file name in folders and zip archives, read in place."""

import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from research.transcripts.parse import get_company_metadata

# A transcript location: ("dir", file path, None) or ("zip", archive path, member name)
Location = Tuple[str, str, Optional[str]]

# Open archives, kept per process so repeated reads don't reopen the zip
_archives: Dict[str, zipfile.ZipFile] = {}


def index_sources(sources: List[str]) -> Dict[str, List[Location]]:
    """Map transcript file name -> every location holding a file with that name."""
    index: Dict[str, List[Location]] = defaultdict(list)
    for source in sources:
        path = Path(source)
        if path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    name = member.rsplit("/", 1)[-1]
                    if name.lower().endswith(".xml"):
                        index[name].append(("zip", str(path), member))
        elif path.is_dir():
            for xml_path in path.rglob("*.xml"):
                index[xml_path.name].append(("dir", str(xml_path), None))
        else:
            raise SystemExit(f"Source is neither a folder nor a zip file: {source}")
    return index


def open_transcript(location: Location):
    """Open a transcript for binary reading, wherever it lives."""
    kind, path, member = location
    if kind == "zip":
        if path not in _archives:
            _archives[path] = zipfile.ZipFile(path)
        return _archives[path].open(member)
    return open(path, "rb")


def _folder_year(location: Location) -> Optional[int]:
    """The innermost year-named folder in a transcript's path (e.g. .../2021/123_T.xml -> 2021)."""
    kind, path, member = location
    parts = (member if kind == "zip" else path).replace("\\", "/").split("/")[:-1]
    years = [int(p) for p in parts if len(p) == 4 and p.isdigit()]
    return years[-1] if years else None


def _normalize_title(title: Optional[str]) -> str:
    return " ".join((title or "").split()).casefold()


def resolve_location(doc: dict, candidates: List[Location]) -> Tuple[Optional[Location], str]:
    """
    Pick the transcript for a document when its file name is in more than one folder
    (the same name in two year folders is usually a different transcript): the one whose
    eventTitle matches documents.event_title, then the one in the document's year folder.
    """
    if len(candidates) == 1:
        return candidates[0], ""
    want = _normalize_title(doc["event_title"])
    if want:
        titled = []
        for location in candidates:
            with open_transcript(location) as fh:
                if _normalize_title(get_company_metadata(fh)["eventTitle"]) == want:
                    titled.append(location)
        if len(titled) == 1:
            return titled[0], ""
        candidates = titled or candidates
    same_year = [loc for loc in candidates if _folder_year(loc) == doc["year"]]
    if len(same_year) == 1:
        return same_year[0], ""
    return (
        None,
        f"{len(candidates)} transcripts named {doc['source_file']}; none uniquely matches its title or year",
    )
