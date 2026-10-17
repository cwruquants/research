"""
Ticker -> CUSIP match-rate diagnostic.

Maps every ticker in the research DB to a CUSIP through LSEG's two bridge files
(ticker -> InstrumentPI via QuoteFile, InstrumentPI -> CUSIP via CUSIPFile), the
same way the professor's 01_ticker_to_cusip.py does, and reports how many match,
how many are ambiguous, and why the rest fail. Nothing is written to the DB.

Inputs
    --tickers   CSV with a `ticker` column and, optionally, `n_docs` (documents per
                ticker) so rates can also be weighted by earnings calls. Export with:
                \\copy (SELECT c.ticker, count(d.id) AS n_docs FROM companies c
                        LEFT JOIN documents d ON d.company_id = c.id
                        GROUP BY c.ticker) TO 'db_tickers.csv' CSV HEADER
    --quote     LSEG QuoteFile (pipe-delimited, no header)
    --cusip     LSEG CUSIPFile (pipe-delimited, no header)

Usage
    python -I src/linking/ticker_cusip_match_rate.py --peek --quote Q.txt --cusip C.txt
    python -I src/linking/ticker_cusip_match_rate.py --tickers db_tickers.csv \\
        --quote Q.txt --cusip C.txt --out results/ticker_cusip_match.csv
"""

import argparse
import csv
import re
import sys

import pandas as pd

# 0-based column positions. The README's QuoteFile columns 2/4/5/9/24 and
# CUSIPFile columns 1/2 count from 1 (checked against the sample files with --peek:
# QuoteFile[1] joins to CUSIPFile[0]). QuoteFile[2] is the full RIC, including the
# ^ delisting code, which is the same format as the DB's companies.ticker.
QUOTE_COLS = {"InstrumentPI": 1, "RIC": 2, "AssetCategory": 3, "AssetType": 4, "Ticker": 8, "ExchangeCountry": 23}
CUSIP_COLS = {"InstrumentPI": 0, "CUSIP": 1}

# Refinitiv RIC suffixes for U.S. venues (incl. OTC). Anything else is a foreign
# listing and is expected to miss, since the QuoteFile filter keeps U.S. only.
US_SUFFIXES = {"", "OQ", "O", "N", "A", "P", "PK", "OB", "K", "Z"}

# Refinitiv marks retired RICs as ROOT^<month letter A-L><2-digit year>, e.g. OCR^H15.
DELIST_RE = re.compile(r"\^([A-L])(\d{2})$")
MONTHS = "ABCDEFGHIJKL"


def read_lseg(path, cols):
    """Read a header-less pipe file as text, keeping only the named positions."""
    kwargs = dict(sep="|", header=None, dtype=str, usecols=list(cols.values()),
                  keep_default_na=False, quoting=csv.QUOTE_NONE)
    try:
        df = pd.read_csv(path, **kwargs)
    except (pd.errors.ParserError, ValueError):
        # Very long fields occasionally break the C parser.
        df = pd.read_csv(path, engine="python", **kwargs)
    df = df.rename(columns={v: k for k, v in cols.items()})
    return df.apply(lambda s: s.str.strip())


def peek(path, n=3):
    """Print the first rows with 0-based column indices so the mapping can be checked."""
    print(f"\n== {path}")
    with open(path, encoding="utf-8", errors="replace") as f:
        for _ in range(n):
            line = f.readline().rstrip("\r\n")
            if not line:
                break
            print("  " + " | ".join(f"[{i}]{v}" for i, v in enumerate(line.split("|"))))


def parse_ticker(raw):
    """Split a DB ticker into (root, exchange suffix, delist yyyy-mm or '')."""
    t = (raw or "").strip().upper().lstrip("*")
    delist = ""
    m = DELIST_RE.search(t)
    if m:
        delist = f"20{m.group(2)}-{MONTHS.index(m.group(1)) + 1:02d}"
        t = t[: m.start()]
    root, _, suffix = t.partition(".")
    return root, suffix, delist


def readme_root(raw):
    """The README's normalisation: strip, upper-case, keep text left of the first '.'."""
    return (raw or "").strip().upper().split(".")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers")
    ap.add_argument("--quote", required=True)
    ap.add_argument("--cusip", required=True)
    ap.add_argument("--out", help="optional per-ticker CSV")
    ap.add_argument("--peek", action="store_true", help="show first rows of the LSEG files and exit")
    ap.add_argument("--quote-cols", help="override, e.g. InstrumentPI=1,AssetCategory=3,...")
    ap.add_argument("--cusip-cols", help="override, e.g. InstrumentPI=0,CUSIP=1")
    args = ap.parse_args()

    for spec, cols in ((args.quote_cols, QUOTE_COLS), (args.cusip_cols, CUSIP_COLS)):
        for pair in filter(None, (spec or "").split(",")):
            k, v = pair.split("=")
            cols[k] = int(v)

    if args.peek:
        peek(args.quote)
        peek(args.cusip)
        return
    if not args.tickers:
        ap.error("--tickers is required unless --peek")

    # --- LSEG bridge tables -------------------------------------------------
    quote = read_lseg(args.quote, QUOTE_COLS)
    n_quote = len(quote)
    quote = quote[(quote.AssetCategory == "ORD") & (quote.AssetType == "EQUITY")
                  & (quote.ExchangeCountry == "United States")]
    quote = quote.assign(ric=quote.RIC.str.upper(),
                         ric_root=[parse_ticker(r)[0] for r in quote.RIC],
                         readme_root=quote.Ticker.str.upper().str.split(".").str[0])

    def pis_by(col):
        q = quote[quote[col] != ""]
        return q.groupby(col).InstrumentPI.agg(lambda s: sorted(set(s)))

    pis_by_ric = pis_by("ric")                  # exact RIC, incl. ^ delist code
    pis_by_root = pis_by("ric_root")            # RIC with suffix and ^ code removed
    pis_by_readme = pis_by("readme_root")       # README: Ticker column, split on '.'

    cusip = read_lseg(args.cusip, CUSIP_COLS)
    cusip = cusip[cusip.CUSIP != ""]
    cusips_by_pi = cusip.groupby("InstrumentPI").CUSIP.agg(lambda s: sorted(set(s)))

    print(f"QuoteFile: {n_quote:,} rows, {len(quote):,} U.S. ordinary equity, "
          f"{len(pis_by_ric):,} distinct RICs, {len(pis_by_root):,} distinct roots, "
          f"{(quote.Ticker == '').sum():,} with empty Ticker column")
    print(f"CUSIPFile: {len(cusip):,} rows, {len(cusips_by_pi):,} distinct InstrumentPIs")
    if len(quote) == 0:
        sys.exit("No QuoteFile rows passed the U.S. equity filter: check column positions with --peek.")

    # --- Classify every DB ticker ------------------------------------------
    tickers = pd.read_csv(args.tickers, dtype={"ticker": str}, keep_default_na=False)
    if "n_docs" not in tickers:
        tickers["n_docs"] = 1

    def has_cusip(pis):
        return any(cusips_by_pi.get(pi) for pi in pis)

    rows = []
    for raw, n_docs in zip(tickers.ticker, tickers.n_docs):
        root, suffix, delist = parse_ticker(raw)
        ric = (raw or "").strip().upper().lstrip("*")

        # Exact RIC first (it pins the listing period); fall back to the bare root.
        pis, method = pis_by_ric.get(ric, []), "ric_exact"
        if not has_cusip(pis):
            pis, method = pis_by_root.get(root, []), "root"
        cusips = sorted({c for pi in pis for c in cusips_by_pi.get(pi, [])})
        cusip8s = sorted({c[:8] for c in cusips})

        if not root:
            status = "blank"
        elif suffix not in US_SUFFIXES:
            status = "foreign_listing"
        elif not pis:
            status = "no_instrument"
        elif not cusips:
            status = "instrument_no_cusip"
        elif len(cusip8s) > 1:
            status = "ambiguous"
        else:
            status = "matched"
        if status not in ("matched", "ambiguous"):
            method = ""

        rows.append(dict(ticker=raw, n_docs=n_docs, root=root, suffix=suffix, delist=delist,
                         status=status, method=method, n_instruments=len(pis),
                         n_cusip8=len(cusip8s), cusip8s=";".join(cusip8s),
                         readme_matched=has_cusip(pis_by_readme.get(readme_root(raw), [])),
                         ric_matched=has_cusip(pis_by_ric.get(ric, []))))

    res = pd.DataFrame(rows)

    # --- Report --------------------------------------------------------------
    def table(df, by):
        g = df.groupby(by).agg(tickers=("ticker", "size"), calls=("n_docs", "sum"))
        g["% tickers"] = (100 * g.tickers / g.tickers.sum()).round(1)
        g["% calls"] = (100 * g.calls / g.calls.sum()).round(1)
        return g.sort_values("tickers", ascending=False)

    print(f"\n{len(res):,} DB tickers, {res.n_docs.sum():,} calls\n")
    print("== Status (all tickers)")
    print(table(res, "status").to_string())

    us = res[~res.status.isin(["blank", "foreign_listing"])]
    print("\n== Status (U.S.-suffix tickers only)")
    print(table(us, "status").to_string())

    us = us.assign(has_delist=us.delist != "")
    print("\n== U.S. tickers: active vs. delisted (^ code)")
    print(table(us, ["has_delist", "status"]).to_string())

    linked = res[res.status.isin(["matched", "ambiguous"])]
    print("\n== Linked tickers by method")
    print(table(linked, ["method", "status"]).to_string())

    print("\n== Methods compared (tickers / calls with any CUSIP)")
    for label, mask in [("README (Ticker column, split on '.')", res.readme_matched),
                        ("Exact RIC only", res.ric_matched),
                        ("This script (exact RIC, then root)", res.status.isin(["matched", "ambiguous"]))]:
        print(f"  {label:<38} {mask.sum():>7,} / {res.n_docs[mask].sum():>9,}")

    if args.out:
        res.to_csv(args.out, index=False)
        print(f"\nPer-ticker results written to {args.out}")


if __name__ == "__main__":
    main()
