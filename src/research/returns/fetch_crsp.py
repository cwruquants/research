"""
Pull the CRSP daily data event_returns.py needs from WRDS, in one connection.

Reads the linked PERMNOs from the events file (research export output with a
permno column) and writes three Parquet files to --out-dir:

    dsi.parquet       date, vwretd       CRSP value-weighted index with dividends
    dsf.parquet       permno, date, ret  daily holding-period returns
    delist.parquet    permno, dlstdt, dlret, dlstcd

The WRDS username and password come from the WRDS entry in ~/.pgpass (the same
file the wrds package writes), so nothing secret is typed or printed. WRDS may
send a Duo push when the connection opens.

Usage
    uv run research returns fetch --events results/event_returns/events.csv.gz \\
        --out-dir data/private/wrds/crsp
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import psycopg

from research.returns.wrds import WRDS_HOST, WRDS_PORT, copy_frame, wrds_user

# Events run 2002-2021. The start leaves room for the 250-day estimation window,
# the end for the +121-day drift window after the last call.
START = "2000-06-01"
END = "2022-12-31"


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
