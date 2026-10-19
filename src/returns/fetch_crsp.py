# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["psycopg[binary]==3.3.5", "pandas==2.3.0", "pyarrow==21.0.0"]
# ///
"""
Pull the CRSP daily data event_returns.py needs from WRDS, in one connection.

Reads the linked PERMNOs from the events file (db/export.py output with a
permno column) and writes three Parquet files to --out-dir:

    dsi.parquet       date, vwretd       CRSP value-weighted index with dividends
    dsf.parquet       permno, date, ret  daily holding-period returns
    delist.parquet    permno, dlstdt, dlret, dlstcd

The WRDS username and password come from the WRDS entry in ~/.pgpass (the same
file the wrds package writes), so nothing secret is typed or printed. WRDS may
send a Duo push when the connection opens.

Usage
    uv run src/returns/fetch_crsp.py --events results/event_returns/events.csv.gz \\
        --out-dir results/event_returns/crsp
"""

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737

# Events run 2002-2021. The start leaves room for the 250-day estimation window,
# the end for the +121-day drift window after the last call.
START = "2000-06-01"
END = "2022-12-31"


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    permnos = sorted(pd.read_csv(args.events, usecols=["permno"])["permno"].dropna().astype(int).unique())
    permno_list = ",".join(str(p) for p in permnos)
    print(f"{len(permnos):,} PERMNOs, {START} to {END}")

    queries = {
        "dsi": f"select date, vwretd from crsp.dsi where date between '{START}' and '{END}' order by date",
        "dsf": f"select permno, date, ret from crsp.dsf where permno in ({permno_list}) "
               f"and date between '{START}' and '{END}'",
        "delist": f"select permno, dlstdt, dlret, dlstcd from crsp.dsedelist where permno in ({permno_list})",
    }
    with psycopg.connect(host=WRDS_HOST, port=WRDS_PORT, dbname="wrds", user=wrds_user(),
                         sslmode="require", connect_timeout=60) as conn:
        conn.read_only = True
        for name, query in queries.items():
            t0 = time.time()
            df = copy_frame(conn, query)
            df.to_parquet(out / f"{name}.parquet", index=False)
            print(f"{name}: {len(df):,} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
