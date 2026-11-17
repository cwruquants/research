"""LSEG bridge files: QuoteFile (RIC -> InstrumentPI) and CUSIPFile (InstrumentPI -> CUSIP)."""

import csv
import re

import pandas as pd

# 0-based column positions. The README's QuoteFile columns 2/4/5/9/24 and
# CUSIPFile columns 1/2 count from 1 (checked against the sample files with --peek:
# QuoteFile[1] joins to CUSIPFile[0]). QuoteFile[2] is the full RIC, including the
# ^ delisting code, which is the same format as the DB's companies.ticker.
QUOTE_COLS = {
    "InstrumentPI": 1,
    "RIC": 2,
    "AssetCategory": 3,
    "AssetType": 4,
    "Ticker": 8,
    "ExchangeCountry": 23,
}
CUSIP_COLS = {"InstrumentPI": 0, "CUSIP": 1}

# Refinitiv RIC suffixes for U.S. venues (incl. OTC). Anything else is a foreign
# listing and is expected to miss, since the QuoteFile filter keeps U.S. only.
US_SUFFIXES = {"", "OQ", "O", "N", "A", "P", "PK", "OB", "K", "Z"}

# Refinitiv marks retired RICs as ROOT^<month letter A-L><2-digit year>, e.g. OCR^H15.
DELIST_RE = re.compile(r"\^([A-L])(\d{2})$")
MONTHS = "ABCDEFGHIJKL"
# Single-letter "suffixes" that are share classes after a U.S. root (BRK.A is
# Berkshire class A, not AMEX; BF.B is Brown-Forman class B).
CLASS_SUFFIXES = {"A", "B"}


def read_lseg(path, cols):
    """Read a header-less pipe file as text, keeping only the named positions."""
    kwargs = dict(
        sep="|",
        header=None,
        dtype=str,
        usecols=list(cols.values()),
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )
    try:
        df = pd.read_csv(path, **kwargs)
    except (pd.errors.ParserError, ValueError):
        # Very long fields occasionally break the C parser.
        df = pd.read_csv(path, engine="python", **kwargs)
    df = df.rename(columns={v: k for k, v in cols.items()})
    return df.apply(lambda s: s.str.strip())


def parse_ric(raw):
    """Split a RIC into (root, share class, exchange suffix, delist yyyy-mm, foreign)."""
    t = (raw or "").strip().lstrip("*")
    delist = ""
    m = DELIST_RE.search(t.upper())
    if m:
        delist = f"20{m.group(2)}-{MONTHS.index(m.group(1)) + 1:02d}"
        t = t[: m.start()]
    parts = t.split(".")
    root, rest = parts[0], [p.upper() for p in parts[1:]]
    suffix = ""
    if rest and (rest[-1] not in CLASS_SUFFIXES or len(rest) > 1):
        suffix = rest.pop()
    cls = rest[0] if len(rest) == 1 and len(rest[0]) == 1 else ""
    # Some RICs spell the class with a space: "ERIC B.ST", "FMER PR A".
    root, *words = root.split() or [""]
    if words:
        cls = cls or ("PREF" if words[0].upper() == "PR" else words[0].upper())
    # Refinitiv writes share classes and preferreds in lower case: LGFa, BRKb, AMLp, INN_u.
    m = re.fullmatch(r"([A-Z0-9]+)_?([a-z]+)", root)
    if m:
        root, low = m.group(1), m.group(2)
        if low[0] in "pu":
            cls = cls or "PREF"  # preferred share or unit
        elif len(low) == 1:
            cls = cls or low.upper()
    foreign = bool(suffix) and suffix not in US_SUFFIXES
    return root.upper(), cls, suffix, delist, foreign


class LsegBridge:
    """RIC -> CUSIP8s through the LSEG files: QuoteFile (U.S. ordinary equity lines only)
    gives the RIC's InstrumentPIs, CUSIPFile their CUSIPs."""

    def __init__(self, quote_path, cusip_path):
        quote = read_lseg(quote_path, QUOTE_COLS)
        quote = quote[
            (quote.AssetCategory == "ORD")
            & (quote.AssetType == "EQUITY")
            & (quote.ExchangeCountry == "United States")
        ]
        quote = quote.assign(ric=quote.RIC.str.upper(), ric_root=[parse_ric(r)[0] for r in quote.RIC])
        self.pis_by_ric = quote.groupby("ric").InstrumentPI.agg(lambda s: sorted(set(s))).to_dict()
        self.pis_by_root = quote.groupby("ric_root").InstrumentPI.agg(lambda s: sorted(set(s))).to_dict()
        cus = read_lseg(cusip_path, CUSIP_COLS)
        cus = cus[cus.CUSIP != ""]
        self.c8_by_pi = cus.groupby("InstrumentPI").CUSIP.agg(lambda s: sorted({c[:8] for c in s})).to_dict()

    def cusip8s(self, raw, root):
        """The CUSIP8s of the exact RIC's instruments, else of the RIC root's."""
        ric = (raw or "").strip().upper().lstrip("*")
        for pis in (self.pis_by_ric.get(ric, []), self.pis_by_root.get(root, [])):
            c8 = sorted({c for pi in pis for c in self.c8_by_pi.get(pi, [])})
            if c8:
                return c8
        return []
