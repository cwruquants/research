"""Fail if a tracked file holds something that must not be in this public repo.

Checks every file git tracks for infrastructure details (private-network IPs, Windows
host names, personal home paths, database hosts, emails) and for licensed transcript
files. Besides each file's raw bytes, the members of zip-based files (.zip, .xlsx) and
UTF-16 text are scanned too, and archives or data files (.gz, .parquet, ...) may only sit
in ALLOWED_DATA.
CI runs it on every push; run it yourself before committing docs or SQL:

    uv run python scripts/check_public_safe.py
"""

import io
import re
import subprocess
import sys
import zipfile

PATTERNS = {
    "Tailscale/CGNAT IP address": re.compile(rb"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"),
    "private IP address": re.compile(
        rb"\b(10\.\d{1,3}|192\.168|172\.(1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"
    ),
    "Windows host name": re.compile(rb"\bDESKTOP-[A-Z0-9]{5,}\b", re.IGNORECASE),
    "Tailscale host name": re.compile(rb"\b[\w-]+\.[\w-]+\.ts\.net\b"),
    "home directory path": re.compile(
        rb"(?:/Users/|/home/|[A-Za-z]:\\\\?Users\\\\?)(?!<|Shared\b|runner\b)[A-Za-z0-9._-]+"
    ),
    "Supabase host": re.compile(rb"\b[a-z0-9]{20}\.supabase\.co\b|pooler\.supabase\.com"),
    "email address": re.compile(
        rb"\b[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com)[A-Za-z0-9-]+\.[A-Za-z.]{2,}\b"
    ),
    "vendor transcript": re.compile(
        rb"\(Thomson StreetEvents\) -- Edited Transcript|<eventTitle>(?!Q4 2016 Example Industries)"
    ),
}
# Files that must hold these strings to do their job
ALLOWED = {
    "scripts/check_public_safe.py": set(PATTERNS),
    "uv.lock": {"private IP address"},  # package versions such as 10.3.7.77
    "tests/unit/db/test_document_year.py": {"vendor transcript"},
}
DATA_FILE = re.compile(
    r"\.(zip|gz|tgz|bz2|tbz2|xz|txz|zst|lz4|7z|rar|tar|parquet|feather|arrow|pkl|pickle|npy|npz"
    r"|h5|dta|sas7bdat|rds|rdata|dump|db|sqlite|duckdb|xls|xlsx|xlsm|xlsb|ods|docx|pptx|pdf)$",
    re.IGNORECASE,
)
ALLOWED_DATA = ("data/word_sets/original_xlsx/",)  # published word lists
ZIP_BASED = (".zip", ".xlsx", ".xlsm", ".docx", ".pptx", ".ods")


def _utf16(data: bytes):
    """UTF-16 text (Windows PowerShell 5 redirects) re-encoded so the patterns can match it."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff") or (data and data.count(b"\0") > len(data) // 4):
        try:
            return data.decode("utf-16").encode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def _parts(path: str, data: bytes):
    """(label, bytes) to scan: the raw file, its UTF-16 reading, and each zip member."""
    for where, raw in [(path, data)] + list(_members(path, data)):
        yield where, raw
        decoded = _utf16(raw)
        if decoded is not None:
            yield where, decoded


def _members(path: str, data: bytes):
    if not path.lower().endswith(ZIP_BASED):
        return
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.namelist():
                yield f"{path}!{member}", archive.read(member)
    except Exception as exc:  # corrupt, encrypted or not really a zip
        yield f"{path}: unreadable archive ({exc.__class__.__name__})", b""


def main() -> int:
    files = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout.split(b"\0")
    problems = []
    for name in filter(None, files):
        path = name.decode()
        if path.lower().endswith(".xml") and not path.startswith("tests/fixtures/"):
            problems.append(f"{path}: transcript XML outside tests/fixtures")
        if DATA_FILE.search(path) and not path.startswith(ALLOWED_DATA):
            problems.append(f"{path}: archive or data file outside {', '.join(ALLOWED_DATA)}")
        try:
            data = open(path, "rb").read()
        except (FileNotFoundError, IsADirectoryError):
            continue
        for where, text in _parts(path, data):
            if "unreadable archive" in where:
                problems.append(where)
                continue
            if "!" in where and re.search(r"_T\.xml$", where, re.IGNORECASE):
                problems.append(f"{where}: transcript XML inside an archive")
            if "!" in where and DATA_FILE.search(where):
                problems.append(f"{where}: archive or data file inside an archive")
            for label, pattern in PATTERNS.items():
                if label in ALLOWED.get(path, ()):
                    continue
                for match in pattern.finditer(text):
                    line = text.count(b"\n", 0, match.start()) + 1
                    problems.append(f"{where}:{line}: {label}")
    problems = list(dict.fromkeys(problems))  # a hit can show up in two views of one file
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s) in {len(files) - 1} tracked files")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
