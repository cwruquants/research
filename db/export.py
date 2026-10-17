# /// script
# requires-python = ">=3.10"
# dependencies = ["psycopg[binary]==3.3.5", "pyyaml==6.0.2"]
# ///
"""Stream a read-only query from the `research` database to a local CSV file.

Connects with the saved `research_ro` profile from postgres-mcp
(~/.postgres-mcp/connections.yaml) and reads its password from the macOS
Keychain, so no secret is ever typed, passed as an argument, or printed.

The session is forced read-only, the role itself has no write grants, and the
output file is never overwritten.

Usage:
    uv run db/export.py --out results/documents.csv --sql "select * from public.documents"
    uv run db/export.py --out results/q.csv.gz --sql-file my_query.sql
"""

from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
import time
from pathlib import Path

import psycopg
import yaml
from psycopg import sql

PROFILE = "research_ro"
CONFIG = Path.home() / ".postgres-mcp" / "connections.yaml"
KEYCHAIN_SERVICE = "postgres-mcp"


def load_profile() -> dict:
    """Return host/port/database for the research_ro profile, refusing any other role."""
    profiles = yaml.safe_load(CONFIG.read_text())["connections"]
    profile = next((p for p in profiles if p["name"] == PROFILE), None)
    if profile is None:
        sys.exit(f"Profile '{PROFILE}' not found in {CONFIG}")
    if profile["user"] != PROFILE or profile.get("access_mode") != "ro":
        sys.exit(f"Profile '{PROFILE}' must use user research_ro with access_mode ro; refusing to connect")
    return profile


def keychain_password() -> str:
    """Read the profile's password from the macOS Keychain without echoing it."""
    result = subprocess.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", PROFILE, "-w"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(
            "Could not read the research_ro password from the Keychain. Re-store it with:\n"
            f"  npx -y @microsoft/postgres-mcp@0.2.0 connection set-password {PROFILE}"
        )
    return result.stdout.rstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sql", help="query to export (SELECT, WITH, VALUES, TABLE)")
    source.add_argument("--sql-file", type=Path, help="file containing the query to export")
    parser.add_argument("--out", type=Path, required=True, help="output CSV path; .gz compresses; must not exist")
    args = parser.parse_args()

    query = (args.sql if args.sql is not None else args.sql_file.read_text()).strip().rstrip(";").strip()
    if args.out.exists():
        sys.exit(f"{args.out} already exists; choose a new path (files are never overwritten)")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    profile = load_profile()
    started = time.monotonic()
    with psycopg.connect(
        host=profile["host"],
        port=profile["port"],
        dbname=profile["database"],
        user=PROFILE,
        password=keychain_password(),
        options="-c default_transaction_read_only=on",
        application_name="research-db-export",
    ) as conn:
        conn.read_only = True
        copy_stmt = sql.SQL("COPY ({}) TO STDOUT WITH (FORMAT csv, HEADER)").format(sql.SQL(query))
        opener = gzip.open if args.out.suffix == ".gz" else open
        # "xb" creates the file exclusively, so a file that appears meanwhile is not clobbered either.
        # Open it outside the try so the cleanup below only ever removes a file this run created.
        out_file = opener(args.out, "xb")
        try:
            with out_file as out, conn.cursor() as cur:
                with cur.copy(copy_stmt) as copy:
                    for chunk in copy:
                        out.write(chunk)
                rows = cur.rowcount
        except BaseException:
            args.out.unlink(missing_ok=True)
            raise

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {rows} rows ({size_mb:.1f} MB) to {args.out} in {time.monotonic() - started:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
