"""LSEG bridge files: QuoteFile (RIC -> InstrumentPI) and CUSIPFile (InstrumentPI -> CUSIP)."""

import csv
import re

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
