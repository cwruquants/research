"""
Event title + call date -> CRSP PERMNO/NCUSIP, one document at a time.

Some company rows are named after a later owner or a conference host (TPG Capital,
RBC Capital Markets), so their calls are about other companies. The event title
still names the real company at call time ("Q1 2004 Primedia Earnings Conference
Call"). This script matches that name against CRSP stocknames and proposes a link
per document. Nothing is written to the DB.

Matching
    The title's period label (Q1 2004, Full Year 2008, Preliminary 2004, ...) is
    dropped, then the title and every CRSP name are normalised: upper case,
    punctuation removed, legal suffixes like INC/CORP/HOLDINGS dropped, common CRSP
    abbreviations folded, spaced initials joined ("P D C" -> "PDC"). A CRSP name
    matches when its tokens are the leading tokens of the title; the longest match
    wins. In order of preference:

      1. a CRSP name in effect on the call date;
      2. any name that PERMNO ever had, if the PERMNO was listed on the call date
         (titles often use a name CRSP changed earlier or later, e.g. Macrovision
         calls in 2009 after CRSP had switched to MACROVISION SOLUTIONS);
      3. a name that ended within a year before the call (calls just after a
         delisting), never before the PERMNO's first listing.

    Several PERMNOs of one PERMCO (share classes) resolve to the longest-listed
    common share. Different PERMCOs tied on length are left ambiguous. NCUSIP and
    ticker always come from the stocknames row in effect on the call date when there
    is one, since CUSIPs change over time.

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
    python -I src/linking/title_crsp_match.py --calls calls.csv \\
        --stocknames stocknames.csv --out proposed_links.csv
"""

import argparse
import re
from collections import defaultdict
from datetime import timedelta

import pandas as pd

# Reporting-period words Thomson puts before the company name, stripped repeatedly
# from the start ("Q4/Year-End 2005", "2005 Q4", "Preliminary 2004", "Full Year 2008").
PERIOD_RE = re.compile(
    r"^\s*(?:(?:q[1-4]|[1-4]q|h[12]|fy\d{0,4}|\d{4}(?:/\d{2,4})?|preliminary|full\s+year|half\s+year|"
    r"first\s+half|second\s+half|interim|fiscal(?:\s+year)?|nine\s+months|year[\s-]+end)(?=\W|$)|[/-])",
    re.I,
)

# Words dropped from names on both sides, wherever they appear.
DROP = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "COS", "COMPANIES", "LTD",
    "LIMITED", "LLC", "LP", "PLC", "NV", "SA", "AG", "HOLDINGS", "HOLDING", "HLDGS", "GROUP",
    "GRP", "THE", "AND", "COM", "NEW", "DE", "DEL",
}

# Spelled-out word -> CRSP abbreviation, so both sides fold to the same token.
FOLD = {
    "INTERNATIONAL": "INTL", "TECHNOLOGIES": "TECH", "TECHNOLOGY": "TECH", "TECHS": "TECH",
    "SYSTEMS": "SYS", "SYSTEM": "SYS", "FINANCIAL": "FINL", "SERVICES": "SVCS", "SERVICE": "SVCS",
    "PHARMACEUTICALS": "PHARM", "PHARMACEUTICAL": "PHARM", "PHARMS": "PHARM",
    "COMMUNICATIONS": "COMM", "COMMUNICATION": "COMM", "COMMUNICATNS": "COMM", "COMMS": "COMM",
    "BANCORPORATION": "BANCORP", "MANAGEMENT": "MGMT", "ENTERPRISES": "ENT", "INDUSTRIES": "IND",
    "INDS": "IND", "LABORATORIES": "LABS", "AMERICA": "AMER", "AMERICAN": "AMERN",
    "NATIONAL": "NATL", "PRODUCTS": "PRODS", "RESOURCES": "RES", "SOLUTIONS": "SOLS",
    "ASSOCIATES": "ASSOC", "INVESTMENTS": "INVTS", "MEDICAL": "MED", "ENTERTAINMENT": "ENTMT",
    "ELECTRONICS": "ELECTRS", "INSURANCE": "INS", "SEMICONDUCTOR": "SEMICON",
    "MANUFACTURING": "MFG",
}

# Title words that may follow a company name; a match ending right before one is clean.
BOUNDARY = {
    "EARNINGS", "CONFERENCE", "CALL", "AT", "TO", "DISCUSSING", "DISCUSS", "ANNOUNCES", "ANNOUNCE",
    "PLANS", "MERGER", "ACQUISITION", "ANALYST", "ANALYSTS", "INVESTOR", "INVESTORS", "ANNUAL",
    "SHAREHOLDER", "SHAREHOLDERS", "BUSINESS", "GUIDANCE", "SALES", "PRESENTATION", "SPECIAL",
    "WEBCAST", "MEETING", "PRE", "PRERECORDED", "UPDATE", "FINL", "RESULTS", "STRATEGIC",
    "INVESTMENT", "FY", "Q1", "Q2", "Q3", "Q4", "PRELIMINARY", "OPERATING", "FIRST", "SECOND",
    "THIRD", "FOURTH", "STATUS", "CORPORATE", "VIRTUAL", "FIRESIDE", "HOSTED",
    "ANNOUNCEMENT",
}

# Legal forms that mark a non-U.S. company. A title that carries one ("Crane Group
# Limited") should not link to a CRSP name that lacks it ("CRANE CO").
FOREIGN_FORMS = {"LTD", "LIMITED", "PLC", "NV", "AG", "SA", "PTY", "SPA", "BV", "AB", "ASA", "SE"}

LISTED_SLACK = timedelta(days=30)
NEAR_DATE = timedelta(days=365)


def tokens(text, possessive=False, keep_drop=False):
    """Normalise a company name or title into comparison tokens."""
    t = (text or "").upper().replace("&", " ").replace(".", " ")
    if possessive:
        t = re.sub(r"[`'’]S\b", "", t)
    t = re.sub(r"[`'’]", "", t)
    t = re.sub(r"[^A-Z0-9 ]+", " ", t)
    # CRSP spells initials apart ("P D C ENERGY", "O CHARLEYS", "N V"); titles write
    # them together ("PDC Energy", "O'Charleys"). Join a run of single letters into
    # one word, and glue a lone initial onto the word after it.
    words, run = [], ""
    for w in t.split():
        if len(w) == 1 and w.isalpha():
            run += w
            continue
        if len(run) == 1:
            w = run + w
        elif run:
            words.append(run)
        words.append(w)
        run = ""
    if run:
        words.append(run)
    out = []
    for w in words:
        w = FOLD.get(w, w)
        if keep_drop or w not in DROP:
            out.append(w)
    return out


def foreign_mismatch(core, comnam):
    """True when the title names a non-U.S. legal form and the CRSP name has none."""
    # Only look at the name part of the title, up to the first event word.
    name = []
    for w in tokens(core, keep_drop=True):
        if w in BOUNDARY:
            break
        name.append(w)
    title_forms = FOREIGN_FORMS.intersection(name)
    crsp_forms = FOREIGN_FORMS.intersection(tokens(comnam, keep_drop=True))
    return bool(title_forms) and not crsp_forms


def title_core(title):
    """Drop the leading reporting-period label from an event title."""
    t = title or ""
    while True:
        s = PERIOD_RE.sub("", t, count=1)
        if s == t:
            return t.strip()
        t = s


def load_crsp(path):
    sn = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("namedt", "nameenddt", "st_date", "end_date"):
        sn[col] = pd.to_datetime(sn[col]).dt.date
    sn["toks"] = sn["comnam"].map(lambda s: tuple(tokens(s)))
    sn = sn[sn["toks"].map(len) > 0]
    by_first = defaultdict(list)
    by_permno = defaultdict(list)
    for r in sn.itertuples(index=False):
        by_first[r.toks[0]].append(r)
        by_permno[r.permno].append(r)
    return by_first, by_permno


def prefix_len(toks, r):
    """Length of the CRSP name if it is the start of the title, else 0."""
    n = len(r.toks)
    if tuple(toks[:n]) == r.toks:
        return n
    # CRSP cuts names at 32 characters; accept a truncated last token.
    if n > 1 and len(r.comnam) >= 30 and len(toks) >= n and tuple(toks[: n - 1]) == r.toks[:-1] \
            and toks[n - 1].startswith(r.toks[-1]):
        return n
    return 0


def row_on(by_permno, permno, d):
    """The stocknames row of a PERMNO in effect on date d, else the nearest one."""
    rows = by_permno[permno]
    for r in rows:
        if r.namedt <= d <= r.nameenddt:
            return r
    return min(rows, key=lambda r: min(abs((r.namedt - d).days), abs((r.nameenddt - d).days)))


def match_tokens(toks, d, by_first, by_permno):
    hits = [(prefix_len(toks, r), r) for r in by_first.get(toks[0], [])]
    hits = [(n, r) for n, r in hits if n]
    passes = [
        ("", lambda r: r.namedt <= d <= r.nameenddt),
        ("_other_name", lambda r: r.st_date - LISTED_SLACK <= d <= r.end_date + LISTED_SLACK),
        # Only after the PERMNO first listed: a pre-IPO call has no stock to link.
        ("_near_date", lambda r: r.st_date <= d <= r.nameenddt + NEAR_DATE),
    ]
    for tag, ok in passes:
        found = [(n, r) for n, r in hits if ok(r)]
        if found:
            best = max(n for n, _ in found)
            permnos = sorted({r.permno for n, r in found if n == best})
            return tag, best, [row_on(by_permno, p, d) for p in permnos]
    return None, 0, []


def match_one(title, d, by_first, by_permno):
    """Return (status, chosen stocknames row or None, candidate rows)."""
    core = title_core(title)
    variants = [tokens(core), tokens(core, possessive=True)]
    variants = [v for v in variants if v]
    if not variants:
        return "no_name", None, []

    tag, best, rows, toks = None, 0, [], variants[0]
    for v in variants:
        t, n, rs = match_tokens(v, d, by_first, by_permno)
        if t is not None and n > best:
            tag, best, rows, toks = t, n, rs, v
    if tag is None:
        return "no_match", None, []
    if len({r.permco for r in rows}) > 1:
        return "ambiguous", None, rows
    by_listing = {r.permno: min(x.st_date for x in by_permno[r.permno]) for r in rows}
    rows.sort(key=lambda r: (r.shrcd not in ("10", "11"), by_listing[r.permno], r.permno))
    nxt = toks[best] if len(toks) > best else ""
    clean = nxt == "" or nxt in BOUNDARY or re.fullmatch(r"\d{4}", nxt) is not None
    if foreign_mismatch(core, rows[0].comnam):
        return "needs_review", rows[0], rows
    status = ("matched" if clean else "matched_prefix") + tag
    if len(rows) > 1:
        status += "_multiclass"
    return status, rows[0], rows


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
        out.append({
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
        })
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
