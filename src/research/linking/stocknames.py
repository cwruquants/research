"""
CRSP stocknames, read once and indexed for both kinds of linking.

route_a looks rows up by ticker, NCUSIP, CUSIP and PERMNO; title matching
(crsp_names.match_one, used by route_a and route_b) looks them up by the first token
of the normalised company name. Each row is a stocknames record with its dates
parsed and `toks`, the comparison tokens of its comnam.
"""

from collections import defaultdict
from datetime import date
from functools import cached_property

import pandas as pd

from research.linking.crsp_names import row_on, tokens

# The last date in the stocknames and daily stock files the links were made with.
CRSP_END = date(2024, 12, 31)


def on(rows, d):
    """The rows whose name period covers date d."""
    return [r for r in rows if r.namedt <= d <= r.nameenddt]


class StockNames:
    def __init__(self, path):
        sn = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in ("namedt", "nameenddt", "st_date", "end_date"):
            sn[col] = pd.to_datetime(sn[col]).dt.date
        sn["toks"] = sn["comnam"].map(lambda s: tuple(tokens(s)))
        self.rows = list(sn.itertuples(index=False))

        self.by_ticker = defaultdict(list)
        self.by_ncusip = defaultdict(list)
        self.permnos_by_cusip = defaultdict(set)
        self.permnos_by_cusip6 = defaultdict(set)
        self.by_permno = defaultdict(list)
        # Title matching uses only rows whose name leaves tokens after normalising.
        self.by_first_token = defaultdict(list)
        self.named_by_permno = defaultdict(list)
        for r in self.rows:
            if r.ticker:
                self.by_ticker[r.ticker].append(r)
            if r.ncusip:
                self.by_ncusip[r.ncusip[:8]].append(r)
                self.permnos_by_cusip[r.ncusip[:8]].add(r.permno)
            if r.cusip:
                self.permnos_by_cusip[r.cusip[:8]].add(r.permno)
            for c in (r.ncusip, r.cusip):
                if c:
                    self.permnos_by_cusip6[c[:6]].add(r.permno)
            self.by_permno[r.permno].append(r)
            if r.toks:
                self.by_first_token[r.toks[0]].append(r)
                self.named_by_permno[r.permno].append(r)
        self._anchors = {}

    @cached_property
    def comnams(self):
        """PERMNO -> sorted (first date, comnam) of every name it had."""
        return {
            p: sorted({(min(x.namedt for x in rs if x.comnam == r.comnam), r.comnam) for r in rs})
            for p, rs in self.by_permno.items()
        }

    @cached_property
    def span(self):
        """PERMNO -> (first, last) listing date."""
        return {
            p: (min(r.st_date for r in rs), max(r.end_date for r in rs)) for p, rs in self.by_permno.items()
        }

    def row_on(self, permno, d):
        """The PERMNO's row in effect on date d, else the nearest one."""
        return row_on(self.by_permno, permno, d)

    def listed(self, permno, d):
        st, end = self.span[permno]
        return st <= d <= end

    def anchor(self, root, delist):
        """
        The PERMNOs that had the ticker root when its RIC was last used: in its
        delisting month (yyyy-mm), or on CRSP_END for a live RIC. Takes the name period
        covering that date, else the last one before it.
        """
        key = (root, delist)
        if key not in self._anchors:
            if delist:
                y, m = map(int, delist.split("-"))
                d = date(y, m, 1)
            else:
                d = CRSP_END
            rows = self.by_ticker.get(root, [])
            hit = on(rows, d)
            if not hit and rows:
                before = [r for r in rows if r.namedt <= d]
                if before:
                    last = max(r.nameenddt for r in before)
                    hit = [r for r in before if r.nameenddt == last]
            self._anchors[key] = sorted({r.permno for r in hit})
        return self._anchors[key]
