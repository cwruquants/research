# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["psycopg[binary]==3.3.5", "pandas==2.3.0", "pyarrow==21.0.0"]
# ///
"""
Pre-call market cap for each call in event_returns.py's output, for
document_event_returns.mcap_pre.

mcap_pre is |prc| * shrout from CRSP dsf, in USD millions, on the last trading
day at least 2 calendar days before day 0, looking back at most 30 days. A
negative CRSP price is a bid/ask midpoint, so its absolute value is used.

Pulls permno, date, prc, shrout for the linked PERMNOs from WRDS in one
connection (credentials from ~/.pgpass, as in fetch_crsp.py), and writes a CSV
of document_id, mcap_pre for the calls that have one. Nothing is written to the DB.

Usage
    uv run src/returns/market_cap.py --returns results/event_returns/event_returns.csv.gz \\
        --out results/event_returns/mcap_pre.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).parent))
from fetch_crsp import WRDS_HOST, WRDS_PORT, copy_frame, wrds_user  # noqa: E402

LAG = pd.Timedelta(days=2)
TOLERANCE = pd.Timedelta(days=30)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--returns", required=True, help="event_returns.py output (document_id, permno, day0)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    calls = pd.read_csv(args.returns, usecols=["document_id", "permno", "day0"]).dropna(subset=["day0"])
    calls["size_date"] = pd.to_datetime(calls.day0).astype("datetime64[ns]") - LAG
    permnos = sorted(calls.permno.astype(int).unique())
    start = (calls.size_date.min() - TOLERANCE).date()
    end = calls.size_date.max().date()
    query = (f"select permno, date, prc, shrout from crsp.dsf where permno in ({','.join(map(str, permnos))}) "
             f"and date between '{start}' and '{end}' and prc is not null and shrout is not null")
    with psycopg.connect(host=WRDS_HOST, port=WRDS_PORT, dbname="wrds", user=wrds_user(),
                         sslmode="require", connect_timeout=120) as conn:
        conn.read_only = True
        dsf = copy_frame(conn, query)
    print(f"dsf: {len(dsf):,} rows for {dsf.permno.nunique():,} of {len(permnos):,} PERMNOs")

    dsf["mcap_pre"] = dsf.prc.abs() * dsf.shrout / 1000  # shrout is in thousands
    dsf = dsf[dsf.mcap_pre > 0]
    dsf["date"] = pd.to_datetime(dsf.date).astype("datetime64[ns]")
    merged = pd.merge_asof(calls.sort_values("size_date"), dsf.sort_values("date")[["permno", "date", "mcap_pre"]],
                           left_on="size_date", right_on="date", by="permno",
                           direction="backward", tolerance=TOLERANCE)
    out = merged.dropna(subset=["mcap_pre"])[["document_id", "mcap_pre"]].sort_values("document_id")
    out["mcap_pre"] = out.mcap_pre.round(4)
    out.to_csv(args.out, index=False)
    print(f"{len(out):,} of {len(calls):,} calls have mcap_pre -> {args.out}")


if __name__ == "__main__":
    main()
