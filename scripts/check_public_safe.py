"""Fail if a tracked file holds something that must not be in this public repo.

Checks every file git tracks for infrastructure details (private-network IPs, Windows
host names, personal home paths, database hosts, emails) and for licensed transcript
files. CI runs it on every push; run it yourself before committing docs or SQL:

    uv run python scripts/check_public_safe.py
"""

import re
import subprocess
import sys

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
    "src/research/transcripts/parse.py": {"vendor transcript"},
    "tests/unit/db/test_document_year.py": {"vendor transcript"},
    "tests/unit/matching/test_phrase_matching.py": {"vendor transcript"},
}


def main() -> int:
    files = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout.split(b"\0")
    problems = []
    for name in filter(None, files):
        path = name.decode()
        if path.endswith(".xml") and not path.startswith("tests/fixtures/"):
            problems.append(f"{path}: transcript XML outside tests/fixtures")
        try:
            data = open(path, "rb").read()
        except (FileNotFoundError, IsADirectoryError):
            continue
        for label, pattern in PATTERNS.items():
            if label in ALLOWED.get(path, ()):
                continue
            for match in pattern.finditer(data):
                line = data.count(b"\n", 0, match.start()) + 1
                problems.append(f"{path}:{line}: {label}")
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s) in {len(files) - 1} tracked files")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
