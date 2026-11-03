"""
Event title + call date -> CRSP PERMNO/NCUSIP, one document at a time.

Some company rows are named after a later owner or a conference host (TPG Capital,
RBC Capital Markets), so their calls are about other companies. The event title
still names the real company at call time ("Q1 2004 Primedia Earnings Conference
Call"). This step matches that name against CRSP stocknames and proposes a link
per document. Nothing is written to the DB.

The matching rules are in crsp_names.py.

Status
    matched          clean match; the title continues with a word like "Earnings"
    matched_prefix   the CRSP name is only the start of a longer phrase
    needs_review     a prefix match to a company no clean call on the same row
                     matched (e.g. "Intuit and Google Forge Alliance"), or a title
                     with a non-U.S. legal form matched to a CRSP name without one
    ambiguous        two different companies matched equally well
    no_match / no_name

Inputs
    --calls       CSV with document_id, company_id, company_name, call_date, event_title
    --stocknames  CRSP stocknames CSV (permno, namedt, nameenddt, ncusip, ticker,
                  comnam, permco, shrcd, st_date, end_date)

Usage
    uv run research link route-b --calls calls.csv \\
        --stocknames stocknames.csv --out proposed_links.csv
"""

import argparse

import pandas as pd

from research.linking.crsp_names import load_crsp, match_one, title_core, tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--stocknames", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_first, by_permno = load_crsp(args.stocknames)
    calls = pd.read_csv(args.calls, dtype=str, keep_default_na=False)

    out = []
    for c in calls.itertuples(index=False):
        d = pd.to_datetime(c.call_date).date()
        status, r, rows = match_one(c.event_title, d, by_first, by_permno)
        out.append(
            {
                "document_id": c.document_id,
                "company_id": c.company_id,
                "company_name": c.company_name,
                "call_date": c.call_date,
                "event_title": c.event_title,
                "title_name": " ".join(tokens(title_core(c.event_title))),
                "status": status,
                "permno": r.permno if r is not None else "",
                "ncusip": r.ncusip if r is not None else "",
                "ticker": r.ticker if r is not None else "",
                "comnam": r.comnam if r is not None else "",
                "candidates": "; ".join(sorted({f"{x.permno} {x.ticker} {x.comnam}" for x in rows})),
            }
        )
    res = pd.DataFrame(out)

    # A loose prefix match to a company that no clean call on the same row matched
    # is usually a second company or a host named first ("Vector Capital to Acquire
    # Printronix"); send it to review instead of linking it.
    clean = res[res["status"].str.startswith("matched") & ~res["status"].str.contains("prefix")]
    seen = set(zip(clean["company_id"], clean["permno"]))
    loose = res["status"].str.contains("prefix")
    unseen = [(c, p) not in seen for c, p in zip(res["company_id"], res["permno"])]
    res.loc[loose & pd.Series(unseen, index=res.index), "status"] = "needs_review"

    res.to_csv(args.out, index=False)
    print(res["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
