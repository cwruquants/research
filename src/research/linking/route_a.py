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
from research.linking.lseg import CUSIP_COLS, QUOTE_COLS, parse_ric, read_lseg
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

    quote = read_lseg(args.quote, QUOTE_COLS)
    quote = quote[
        (quote.AssetCategory == "ORD")
        & (quote.AssetType == "EQUITY")
        & (quote.ExchangeCountry == "United States")
    ]
    quote = quote.assign(ric=quote.RIC.str.upper(), ric_root=[parse_ric(r)[0] for r in quote.RIC])
    pis_by_ric = quote.groupby("ric").InstrumentPI.agg(lambda s: sorted(set(s))).to_dict()
    pis_by_root = quote.groupby("ric_root").InstrumentPI.agg(lambda s: sorted(set(s))).to_dict()
    cus = read_lseg(args.cusip, CUSIP_COLS)
    cus = cus[cus.CUSIP != ""]
    c8_by_pi = cus.groupby("InstrumentPI").CUSIP.agg(lambda s: sorted({c[:8] for c in s})).to_dict()

    def lseg_cusips(raw, root):
        ric = (raw or "").strip().upper().lstrip("*")
        for pis in (pis_by_ric.get(ric, []), pis_by_root.get(root, [])):
            c8 = sorted({c for pi in pis for c in c8_by_pi.get(pi, [])})
            if c8:
                return c8
        return []

    docs = pd.read_csv(args.docs, dtype=str, keep_default_na=False)
    route_b = pd.read_csv(args.route_b, dtype=str, keep_default_na=False)
    b_ids = set(route_b.document_id)

    clean_title = {
        "matched",
        "matched_other_name",
        "matched_near_date",
        "matched_multiclass",
        "matched_other_name_multiclass",
        "matched_near_date_multiclass",
    }

    def title_fallback(rec, title, d):
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
        accept = st in clean_title or (
            st.startswith("matched_prefix")
            and "near_date" not in st
            and name_score(crsp, r.permno, rec["company_name"], variants(rec["company_name"])) >= NAME_OK
        )
        if accept:
            if rec.get("permno"):
                rec["other_methods"] = "; ".join(
                    filter(None, [f"{rec['method']}:{rec['permno']}", rec.get("other_methods", "")])
                )
            rec.update(
                status="matched",
                method="title_name",
                permno=r.permno,
                permco=r.permco,
                ncusip8=r.ncusip[:8],
                crsp_ticker=r.ticker,
                comnam=r.comnam,
                shrcd=r.shrcd,
                exchcd=r.exchcd,
                title_score=1.0,
                n_candidates=1,
                candidates="",
                note=rec.get("note", ""),
            )
        return rec

    out = []
    for c in docs.itertuples(index=False):
        if c.document_id in b_ids:
            continue
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
        t_parts = title_parts(c.event_title)
        t_named = any(t_parts)

        if foreign:
            rec.update(status="foreign_listing")
            # A U.S. company whose only RIC is a foreign line (SanDisk as SNDK*.MX):
            # link by a clean title match to U.S. common stock, never to an ADR, and
            # only when the company row carries the same name, so foreign names that
            # collide with U.S. ones (Vector Ltd vs VECTOR GROUP LTD) and U.S.
            # targets filed under a foreign acquirer (Genentech under Roche) stay out.
            st, r, _ = match_one(c.event_title, d, crsp)
            if st in clean_title and r.shrcd in COMMON and same_name(c.company_name, r.comnam):
                rec.update(
                    status="matched",
                    method="title_name",
                    permno=r.permno,
                    permco=r.permco,
                    ncusip8=r.ncusip[:8],
                    crsp_ticker=r.ticker,
                    comnam=r.comnam,
                    shrcd=r.shrcd,
                    exchcd=r.exchcd,
                    title_score=1.0,
                    n_candidates=1,
                    title_status=st,
                    title_permno=r.permno,
                    title_comnam=r.comnam,
                    note="foreign_ric_us_company",
                )
            out.append(rec)
            continue
        if not root or not re.fullmatch(r"[A-Z0-9]{1,8}", root):
            rec.update(
                status="blank" if not root else "no_crsp_match", note="" if not root else "invalid_ticker"
            )
            out.append(title_fallback(rec, c.event_title, d))
            continue

        tries = []  # (method, [rows])
        rows = pick(on(crsp.by_ticker.get(root, []), d), cls)
        if rows:
            tries.append(("ticker_date", rows))
        c8s = lseg_cusips(c.ticker, root)
        rec["lseg_cusip8s"] = ";".join(c8s)
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

        if not tries:
            rec.update(status="no_crsp_match")
            out.append(title_fallback(rec, c.event_title, d))
            continue

        # A title that is just the ticker ("DHI at JPMorgan ...") names the RIC's
        # own company, i.e. the PERMNO the ticker anchors to.
        own = (
            set(crsp.anchor(root, delist)) if any(p == [root] for p in t_parts) and len(root) >= 3 else set()
        )

        def title_fit(permno):
            ts = name_score(crsp, permno, c.event_title, t_parts, d, title=True)
            return max(ts, 0.8) if permno in own else ts

        def judge(rows):
            sc = [
                (title_fit(r.permno), name_score(crsp, r.permno, c.company_name, variants(c.company_name)), r)
                for r in rows
            ]
            sc.sort(key=lambda x: (-x[0], -x[1]))
            ts, cs, r = sc[0]
            ok = (
                ts >= NAME_OK
                or (not t_named and cs >= NAME_OK)
                or (ts >= 0.5 and part_in_name(crsp, r.permno, t_parts, d))
            )
            if len(rows) > 1:
                # Keep only the candidates whose name fits, if that settles it.
                fit = [x for t, k, x in sc if t >= NAME_OK or (not t_named and k >= NAME_OK)]
                if fit and len({x.permco for x in fit}) == 1:
                    rows[:] = fit
            return ok, ts, cs, r

        chosen = None
        for method, rows in tries:
            ok, ts, cs, r = judge(rows)
            if ok:
                chosen = (method, rows, ok, ts, cs, r)
                break
        if chosen is None:
            method, rows = tries[0]
            chosen = (method, rows, *judge(rows))
        method, rows, ok, ts, cs, r = chosen
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
            permno=r.permno,
            permco=r.permco,
            ncusip8=r.ncusip[:8],
            crsp_ticker=r.ticker,
            comnam=r.comnam,
            shrcd=r.shrcd,
            exchcd=r.exchcd,
            title_score=ts,
            company_score=cs,
            n_candidates=len(rows),
            note=note,
            candidates="; ".join(
                f"{x.permno} {x.ticker} {x.shrcls} {x.comnam}".replace("  ", " ") for x in rows
            ),
            other_methods="; ".join(
                f"{m}:{'/'.join(sorted({x.permno for x in rs}))}" for m, rs in tries if m != method
            ),
        )
        if status == "name_mismatch":
            rec = title_fallback(rec, c.event_title, d)
            # A title that names no company ("Connections 2014: Investor Session",
            # "UBS Global Life Sciences Conference") cannot contradict the ticker;
            # accept when the CRSP name fits the company row.
            if rec["status"] == "name_mismatch" and not names_a_company(c.event_title) and cs >= NAME_OK:
                rec.update(status="matched", note="title_unnamed")
        out.append(rec)

    res = pd.DataFrame(out).fillna("")

    # A title prefix match ("Aeropostale Reports November Sales") to a PERMNO that
    # clean title matches already link on the same company row is accepted too,
    # as route B does.
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
            res.at[i, "other_methods"] = "; ".join(
                filter(None, [f"{res.at[i, 'method']}:{res.at[i, 'permno']}", res.at[i, "other_methods"]])
            )
        res.loc[
            i,
            [
                "status",
                "method",
                "permno",
                "permco",
                "ncusip8",
                "crsp_ticker",
                "comnam",
                "shrcd",
                "exchcd",
                "title_score",
                "n_candidates",
                "candidates",
                "note",
            ],
        ] = [
            "matched",
            "title_name",
            r.permno,
            r.permco,
            r.ncusip[:8],
            r.ticker,
            r.comnam,
            r.shrcd,
            r.exchcd,
            1.0,
            1,
            "",
            "title_prefix_same_row",
        ]

    # Route B: title-matched calls on acquirer and broker-host rows.
    b_status = {
        "needs_review": "name_mismatch",
        "ambiguous": "multiple",
        "no_match": "no_crsp_match",
        "no_name": "no_crsp_match",
    }
    rb = route_b.merge(
        docs[["document_id", "ticker"]].rename(columns={"ticker": "ric"}), on="document_id", how="left"
    )
    b = pd.DataFrame(
        dict(
            document_id=rb.document_id,
            company_id=rb.company_id,
            company_name=rb.company_name,
            ric=rb.ric,
            call_date=rb.call_date,
            event_title=rb.event_title,
            route="B",
            status=[s if s in b_status else ("matched" if s.startswith("matched") else s) for s in rb.status],
            method="title_name:" + rb.status,
            permno=rb.permno,
            ncusip8=rb.ncusip.str[:8],
            crsp_ticker=rb.ticker,
            comnam=rb.comnam,
            candidates=rb.candidates,
            after_crsp_end=[int(date.fromisoformat(x) > CRSP_END) for x in rb.call_date],
        )
    )
    b["status"] = [b_status.get(s, s) for s in b.status]
    res = pd.concat([res, b], ignore_index=True)
    res = res.sort_values("document_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    res.to_csv(args.out, index=False)

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


if __name__ == "__main__":
    main()
