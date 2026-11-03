"""WRDS connection details: host, the username from ~/.pgpass, and a COPY-to-DataFrame helper."""

import io
import sys
from pathlib import Path

import pandas as pd

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737


def wrds_user() -> str:
    """Username from the ~/.pgpass line for the WRDS host; libpq reads the password."""
    pgpass = Path.home() / ".pgpass"
    for line in pgpass.read_text().splitlines():
        parts = line.split(":")
        if len(parts) >= 5 and parts[0] == WRDS_HOST:
            return parts[3]
    sys.exit(f"No {WRDS_HOST} entry in {pgpass}; run wrds.Connection() once to create it.")


def copy_frame(conn, query: str) -> pd.DataFrame:
    buf = io.BytesIO()
    with conn.cursor() as cur:
        with cur.copy(f"COPY ({query}) TO STDOUT WITH CSV HEADER") as copy:
            for chunk in copy:
                buf.write(chunk)
    buf.seek(0)
    return pd.read_csv(buf)
