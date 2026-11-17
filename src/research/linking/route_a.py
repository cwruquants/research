"""
Earnings call -> CRSP PERMNO, one document at a time, by ticker and call date.

Each document carries its company's Refinitiv RIC (AAPL.OQ, OCR^H15, LGFa.N). The
RIC is the company row's latest code, and Thomson relabels old calls with the
current owner, so the link is made per document on the call date and every link
is checked against the company the event title names. Nothing is written to the DB.

Matching, per document (call date = event date in New York time)
    The RIC is split into root, share class, exchange suffix and ^ delisting month.
    Non-U.S. suffixes are left unlinked (foreign_listing), except that a call whose
    title cleanly names a U.S. common stock (shrcd 10/11) is linked by title
    (note=foreign_ric_us_company: SanDisk filed as SNDK*.MX); ADRs never are. For
    U.S. RICs up to three
    candidate links are built, in this order:

      ticker_date       CRSP stocknames rows whose ticker is the root and whose name
                        period covers the call date.
      lseg_cusip_date   RIC -> InstrumentPI (LSEG QuoteFile, exact RIC then root)
                        -> CUSIP8 (CUSIPFile) -> stocknames rows with that NCUSIP on
                        the call date. Failing that, a PERMNO that ever carried the
                        CUSIP8 (NCUSIP or header CUSIP) and was listed on the call
                        date (lseg_cusip_permno), then the same for the issuer's
                        CUSIP6 (lseg_cusip6, a new issue number after a reorg).
      ticker_anchor     the PERMNO that had the root as its ticker when the RIC was
                        last used (its delisting month, or CRSP's last date for
                        live RICs), if that PERMNO was listed on the call date.
                        Catches ticker changes (Harris Corp calls under LHX).

    Within a method, a share class from the RIC (BRK.A, LGFa.N) picks the CRSP
    class, then common shares (shrcd 10/11) are preferred; several classes of one
    company with no class in the RIC resolve to the unlettered or class A line
    (note=share_class_pick). The first method whose CRSP name fits the call wins.

    When no method fits, the event title is matched to CRSP names on the call date
    with crsp_names.py (method=title_name): Thomson files old calls under
    the current owner (Grant Prideco calls under NOV) and tickers get reused (EPC
    was EPCOS before Edgewell). Clean title matches are accepted, as are prefix
    matches whose CRSP name fits the company row or that clean matches on the
    same row already link (note=title_prefix_same_row). A title that names no
    company ("UBS Global Life Sciences Conference") cannot contradict the ticker
    and is accepted when the CRSP name fits the company row (note=title_unnamed).

Name check
    The CRSP names the PERMNO had by the call date are compared with the event
    title (the part before "at <host>" / "hosted by"): the weighted share of CRSP
    name words found in the title, or of the title's name words found in the CRSP
    name, with both first words required to agree. Words match when equal, cut
    off, abbreviated in order (INVT ~ INVESTMENT) or misspelled; initials and
    spaced letters are joined (B B & T ~ BB&T). Accepted at >= 0.6, or >= 0.5 when
    every distinctive word of the title's name is in the CRSP name. A title that is
    only the CRSP initials or the RIC root (IBM, DHI) counts for the RIC's company.

Status
    matched          one PERMNO, name check passed
    multiple         several PERMNOs of different companies on the call date
    name_mismatch    one PERMNO, but its CRSP name does not fit the title and the
                     title names no CRSP company cleanly
    no_crsp_match    U.S. RIC (or unparseable ticker), nothing found in CRSP
    foreign_listing  non-U.S. exchange suffix
    blank            no ticker

Documents already linked by title (route_b.py, acquirer and broker-host
rows) are taken from --route-b and merged in with route=B.

Usage
    uv run research link route-a --docs data/private/derived/crsp_linking/documents_export.csv.gz \\
        --stocknames data/private/crsp/stocknames.csv \\
        --quote data/private/lseg/QuoteFile.txt --cusip data/private/lseg/CUSIPFile.txt \\
        --route-b data/private/derived/acquirer_linking/proposed_links.csv \\
        --out data/private/derived/crsp_linking/proposed_links_all.csv
"""

import argparse
import re
from datetime import date

import pandas as pd

from research.linking.crsp_names import match_one
from research.linking.lseg import LsegBridge, parse_ric
from research.linking.names import (
    NAME_OK,
    name_score,
    names_a_company,
    part_in_name,
    same_name,
    title_parts,
    variants,
)
from research.linking.stocknames import CRSP_END, StockNames, on

COMMON = ("10", "11")
# match_one statuses accepted as a title link without further checks.
CLEAN_TITLE = {
    "matched",
    "matched_other_name",
    "matched_near_date",
    "matched_multiclass",
    "matched_other_name_multiclass",
    "matched_near_date_multiclass",
}
# Route B's statuses as route A statuses; the rest that start "matched" are matched.
ROUTE_B_STATUS = {
    "needs_review": "name_mismatch",
    "ambiguous": "multiple",
    "no_match": "no_crsp_match",
    "no_name": "no_crsp_match",
}


def pick(rows, cls):
    """Narrow stocknames rows to one PERMNO where the share class or shrcd allows."""
    if cls and cls != "PREF":
        same = [r for r in rows if r.shrcls == cls]
        rows = same or rows
    common = [r for r in rows if r.shrcd in COMMON]
    rows = common or rows
    by_permno = {}
    for r in rows:
        by_permno.setdefault(r.permno, r)
    return list(by_permno.values())


def crsp_fields(r):
    """The CRSP columns of a link to stocknames row r."""
    return dict(
        permno=r.permno,
        permco=r.permco,
        ncusip8=r.ncusip[:8],
        crsp_ticker=r.ticker,
        comnam=r.comnam,
        shrcd=r.shrcd,
        exchcd=r.exchcd,
    )


def with_replaced(method, permno, other_methods):
    """other_methods with the link being replaced (method:permno) put first."""
    return "; ".join(filter(None, [f"{method}:{permno}", other_methods]))


def title_fallback(crsp, rec, title, d):
    """When the ticker gives nothing or the wrong company, link by the company
    the event title names (crsp_names.py) if that match is clean, or a
    prefix match whose CRSP name also fits the company row (Limited Brands July
    Sales under L Brands)."""
    st, r, _ = match_one(title, d, crsp)
    rec.update(
        title_status=st,
        title_permno=r.permno if r is not None else "",
        title_comnam=r.comnam if r is not None else "",
    )
    accept = st in CLEAN_TITLE or (
        st.startswith("matched_prefix")
        and "near_date" not in st
        and name_score(crsp, r.permno, rec["company_name"], variants(rec["company_name"])) >= NAME_OK
    )
    if accept:
        if rec.get("permno"):
            rec["other_methods"] = with_replaced(rec["method"], rec["permno"], rec.get("other_methods", ""))
        # company_score is left as it was: it still describes the ticker's PERMNO.
        rec.update(
            status="matched",
            method="title_name",
            **crsp_fields(r),
            title_score=1.0,
            n_candidates=1,
            candidates="",
            note=rec.get("note", ""),
        )
    return rec


def candidates(crsp, lseg, ric, root, cls, delist, d):
    """
    (tries, lseg CUSIP8s): up to three (method, stocknames rows) candidate links for a
    U.S. RIC on date d, in order of preference (see the module docstring).
    """
    tries = []
    rows = pick(on(crsp.by_ticker.get(root, []), d), cls)
    if rows:
        tries.append(("ticker_date", rows))
    c8s = lseg.cusip8s(ric, root)
    if c8s:
        rows = pick([r for c8 in c8s for r in on(crsp.by_ncusip.get(c8, []), d)], cls)
        if rows:
            tries.append(("lseg_cusip_date", rows))
        else:
            ps = sorted({p for c8 in c8s for p in crsp.permnos_by_cusip.get(c8, ()) if crsp.listed(p, d)})
            rows = pick([crsp.row_on(p, d) for p in ps], cls)
            if rows:
                tries.append(("lseg_cusip_permno", rows))
            else:
                # Same issuer (CUSIP6), new issue number after a reorganisation.
                c6s = {c8[:6] for c8 in c8s}
                ps = sorted(
                    {p for c6 in c6s for p in crsp.permnos_by_cusip6.get(c6, ()) if crsp.listed(p, d)}
                )
                rows = pick([crsp.row_on(p, d) for p in ps], cls)
                if rows:
                    tries.append(("lseg_cusip6", rows))
    ps = [p for p in crsp.anchor(root, delist) if crsp.listed(p, d)]
    rows = pick([crsp.row_on(p, d) for p in ps], cls)
    if rows:
        tries.append(("ticker_anchor", rows))
    return tries, c8s


class Judge:
    """Whether a candidate link's CRSP name fits one call's event title and company row."""

    def __init__(self, crsp, title, company_name, root, delist, d):
        self.crsp, self.title, self.company_name, self.d = crsp, title, company_name, d
        self.parts = title_parts(title)
        self.named = any(self.parts)
        # A title that is just the ticker ("DHI at JPMorgan ...") names the RIC's
        # own company, i.e. the PERMNO the ticker anchors to.
        self.own = (
            set(crsp.anchor(root, delist))
            if any(p == [root] for p in self.parts) and len(root) >= 3
            else set()
        )

    def title_fit(self, permno):
        ts = name_score(self.crsp, permno, self.title, self.parts, self.d, title=True)
        return max(ts, 0.8) if permno in self.own else ts

    def __call__(self, rows):
        """
        (fits, title score, company score, best row, rows): the best-fitting row and
        whether it fits, and the rows narrowed to those whose name fits when that
        leaves one company.
        """
        crsp, d = self.crsp, self.d
        sc = [
            (
                self.title_fit(r.permno),
                name_score(crsp, r.permno, self.company_name, variants(self.company_name)),
                r,
            )
            for r in rows
        ]
        sc.sort(key=lambda x: (-x[0], -x[1]))
        ts, cs, r = sc[0]
        ok = (
            ts >= NAME_OK
            or (not self.named and cs >= NAME_OK)
            or (ts >= 0.5 and part_in_name(crsp, r.permno, self.parts, d))
        )
        if len(rows) > 1:
            fit = [x for t, k, x in sc if t >= NAME_OK or (not self.named and k >= NAME_OK)]
            if fit and len({x.permco for x in fit}) == 1:
                rows = fit
        return ok, ts, cs, r, rows


def choose(tries, judge):
    """
    The first method whose best row fits, as (method, rows, fits, title score,
    company score, row). With none, the first method, judged again on its rows as the
    first pass narrowed them. Each judged method's rows in tries are replaced by its
    narrowed rows, which other_methods then lists.
    """
    for i, (method, rows) in enumerate(tries):
        ok, ts, cs, r, rows = judge(rows)
        tries[i] = (method, rows)
        if ok:
            return method, rows, ok, ts, cs, r
    method, rows = tries[0]
    ok, ts, cs, r, rows = judge(rows)
    tries[0] = (method, rows)
    return method, rows, ok, ts, cs, r


def link_document(crsp, lseg, c):
    """The proposed link of one documents export row (see the module docstring)."""
    d_call = date.fromisoformat(c.call_date_ny)
    after_end = d_call > CRSP_END
    d = min(d_call, CRSP_END)
    root, cls, suffix, delist, foreign = parse_ric(c.ticker)
    rec = dict(
        document_id=c.document_id,
        company_id=c.company_id,
        company_name=c.company_name,
        ric=c.ticker,
        root=root,
        share_class=cls,
        exch_suffix=suffix,
        delist=delist,
        call_date=c.call_date_ny,
        event_title=c.event_title,
        route="A",
        after_crsp_end=int(after_end),
    )

    if foreign:
        rec.update(status="foreign_listing")
        # A U.S. company whose only RIC is a foreign line (SanDisk as SNDK*.MX):
        # link by a clean title match to U.S. common stock, never to an ADR, and
        # only when the company row carries the same name, so foreign names that
        # collide with U.S. ones (Vector Ltd vs VECTOR GROUP LTD) and U.S.
        # targets filed under a foreign acquirer (Genentech under Roche) stay out.
        st, r, _ = match_one(c.event_title, d, crsp)
        if st in CLEAN_TITLE and r.shrcd in COMMON and same_name(c.company_name, r.comnam):
            rec.update(
                status="matched",
                method="title_name",
                **crsp_fields(r),
                title_score=1.0,
                n_candidates=1,
                title_status=st,
                title_permno=r.permno,
                title_comnam=r.comnam,
                note="foreign_ric_us_company",
            )
        return rec
    if not root or not re.fullmatch(r"[A-Z0-9]{1,8}", root):
        rec.update(status="blank" if not root else "no_crsp_match", note="" if not root else "invalid_ticker")
        return title_fallback(crsp, rec, c.event_title, d)

    tries, c8s = candidates(crsp, lseg, c.ticker, root, cls, delist, d)
    rec["lseg_cusip8s"] = ";".join(c8s)
    if not tries:
        rec.update(status="no_crsp_match")
        return title_fallback(crsp, rec, c.event_title, d)

    judge = Judge(crsp, c.event_title, c.company_name, root, delist, d)
    method, rows, ok, ts, cs, r = choose(tries, judge)
    note = ""
    if len(rows) > 1 and len({x.permco for x in rows}) == 1:
        # Share classes of one company and no class in the RIC: take the
        # unlettered or class A line (GEF over GEF.B, MKC over MKC.V).
        r = min(rows, key=lambda x: (x.shrcls not in ("", "A"), x.shrcls, crsp.span[x.permno][0]))
        note = "share_class_pick"
    if len(rows) > 1 and not note:
        status = "multiple"
    else:
        status = "matched" if ok else "name_mismatch"
    rec.update(
        status=status,
        method=method,
        **crsp_fields(r),
        title_score=ts,
        company_score=cs,
        n_candidates=len(rows),
        note=note,
        candidates="; ".join(f"{x.permno} {x.ticker} {x.shrcls} {x.comnam}".replace("  ", " ") for x in rows),
        other_methods="; ".join(
            f"{m}:{'/'.join(sorted({x.permno for x in rs}))}" for m, rs in tries if m != method
        ),
    )
    if status == "name_mismatch":
        rec = title_fallback(crsp, rec, c.event_title, d)
        # A title that names no company ("Connections 2014: Investor Session",
        # "UBS Global Life Sciences Conference") cannot contradict the ticker;
        # accept when the CRSP name fits the company row.
        if rec["status"] == "name_mismatch" and not names_a_company(c.event_title) and cs >= NAME_OK:
            rec.update(status="matched", note="title_unnamed")
    return rec


def accept_prefix_same_row(crsp, res):
    """
    Link unlinked calls whose title prefix match ("Aeropostale Reports November
    Sales") names a PERMNO that clean title matches on the same company row already
    link, as route B does.
    """
    title_linked = res[res.method == "title_name"]
    seen = set(zip(title_linked.company_id, title_linked.permno))
    later = (
        res.status.isin(["name_mismatch", "no_crsp_match", "blank"])
        & res.title_status.str.startswith("matched_prefix")
        & pd.Series([(co, p) in seen for co, p in zip(res.company_id, res.title_permno)], index=res.index)
    )
    for i in res.index[later]:
        r = crsp.row_on(res.at[i, "title_permno"], min(date.fromisoformat(res.at[i, "call_date"]), CRSP_END))
        if res.at[i, "permno"]:
            res.at[i, "other_methods"] = with_replaced(
                res.at[i, "method"], res.at[i, "permno"], res.at[i, "other_methods"]
            )
        link = dict(
            status="matched",
            method="title_name",
            **crsp_fields(r),
            title_score=1.0,
            n_candidates=1,
            candidates="",
            note="title_prefix_same_row",
        )
        res.loc[i, list(link)] = list(link.values())
    return res


def route_b_links(route_b, docs):
    """Route B's title links (acquirer and broker-host rows) in route A's columns."""
    rb = route_b.merge(
        docs[["document_id", "ticker"]].rename(columns={"ticker": "ric"}), on="document_id", how="left"
    )
    return pd.DataFrame(
        dict(
            document_id=rb.document_id,
            company_id=rb.company_id,
            company_name=rb.company_name,
            ric=rb.ric,
            call_date=rb.call_date,
            event_title=rb.event_title,
            route="B",
            status=[ROUTE_B_STATUS.get(s, "matched" if s.startswith("matched") else s) for s in rb.status],
            method="title_name:" + rb.status,
            permno=rb.permno,
            ncusip8=rb.ncusip.str[:8],
            crsp_ticker=rb.ticker,
            comnam=rb.comnam,
            candidates=rb.candidates,
            after_crsp_end=[int(date.fromisoformat(x) > CRSP_END) for x in rb.call_date],
        )
    )


def report(res):
    n = len(res)
    print(f"{n:,} documents ({(res.route == 'B').sum():,} from route B)\n")
    print(
        res.status.value_counts()
        .to_frame("docs")
        .assign(pct=lambda x: (100 * x.docs / n).round(1))
        .to_string()
    )
    print()
    print(res[res.permno.fillna("") != ""].groupby(["status", "method"]).size().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--stocknames", required=True)
    ap.add_argument("--quote", required=True)
    ap.add_argument("--cusip", required=True)
    ap.add_argument("--route-b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    crsp = StockNames(args.stocknames)
    lseg = LsegBridge(args.quote, args.cusip)
    docs = pd.read_csv(args.docs, dtype=str, keep_default_na=False)
    route_b = pd.read_csv(args.route_b, dtype=str, keep_default_na=False)
    b_ids = set(route_b.document_id)

    res = pd.DataFrame(
        [link_document(crsp, lseg, c) for c in docs.itertuples(index=False) if c.document_id not in b_ids]
    ).fillna("")
    res = accept_prefix_same_row(crsp, res)
    res = pd.concat([res, route_b_links(route_b, docs)], ignore_index=True)
    res = res.sort_values("document_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    res.to_csv(args.out, index=False)
    report(res)


if __name__ == "__main__":
    main()
