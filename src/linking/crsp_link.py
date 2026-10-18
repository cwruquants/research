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
    with title_crsp_match.py (method=title_name): Thomson files old calls under
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

Documents already linked by title (title_crsp_match.py, acquirer and broker-host
rows) are taken from --route-b and merged in with route=B.

Usage
    python -I src/linking/crsp_link.py --docs documents_export.csv.gz \\
        --stocknames stocknames.csv --quote QuoteFile.txt --cusip CUSIPFile.txt \\
        --route-b proposed_links.csv --out proposed_links_all.csv
"""

import argparse
import os
import re
import unicodedata
from difflib import SequenceMatcher
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ticker_cusip_match_rate import CUSIP_COLS, DELIST_RE, MONTHS, QUOTE_COLS, US_SUFFIXES, read_lseg  # noqa: E402
from title_crsp_match import BOUNDARY, DROP, FOLD, load_crsp, match_one, title_core, tokens  # noqa: E402

CRSP_END = date(2024, 12, 31)
COMMON = ("10", "11")
NAME_OK = 0.6
NAME_LEAD = timedelta(days=30)
# "Dell Technologies Inc Investor Call Hosted by Morgan Stanley": the host is not the subject.
HOSTED_BY = re.compile(r"\b(?:hosted|sponsored|moderated)(?:\s+by)?\b.*$", re.I)
# "Salesforce.com Inc at Bank of America ... Conference": score only the part before "at".
AT_HOST = re.compile(r"^(.*?\w.*?)\s+at\s+.*$", re.I)
# Single-letter "suffixes" that are share classes after a U.S. root (BRK.A is
# Berkshire class A, not AMEX; BF.B is Brown-Forman class B).
CLASS_SUFFIXES = {"A", "B"}
# Words too common to count as evidence that a title names the company.
WEAK = {"AMERN", "AMER", "NATL", "INTL", "FIRST", "UNITED", "GENERAL", "BANCORP", "BANK", "TRUST",
        "FINL", "IND", "SYS", "TECH", "SVCS", "RES", "ENERGY", "CAPITAL", "US", "USA"}


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


class Crsp:
    def __init__(self, path):
        sn = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in ("namedt", "nameenddt", "st_date", "end_date"):
            sn[col] = pd.to_datetime(sn[col]).dt.date
        self.rows = list(sn.itertuples(index=False))
        self.by_ticker = defaultdict(list)
        self.by_ncusip = defaultdict(list)
        self.permnos_by_cusip = defaultdict(set)
        self.permnos_by_cusip6 = defaultdict(set)
        self.by_permno = defaultdict(list)
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
        self.comnams = {p: sorted({(min(x.namedt for x in rs if x.comnam == r.comnam), r.comnam) for r in rs})
                        for p, rs in self.by_permno.items()}
        self.tickers = {p: {r.ticker for r in rs if r.ticker} for p, rs in self.by_permno.items()}
        self.span = {p: (min(r.st_date for r in rs), max(r.end_date for r in rs)) for p, rs in self.by_permno.items()}

    def on(self, rows, d):
        return [r for r in rows if r.namedt <= d <= r.nameenddt]

    def row_on(self, permno, d):
        rows = self.by_permno[permno]
        hit = self.on(rows, d)
        if hit:
            return hit[0]
        return min(rows, key=lambda r: min(abs((r.namedt - d).days), abs((r.nameenddt - d).days)))

    def listed(self, permno, d):
        st, end = self.span[permno]
        return st <= d <= end


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


# Legal forms ignored when comparing a company row name with a CRSP name exactly.
LEGAL = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "LLC", "LP",
         "PLC", "THE", "AND", "DE", "DEL", "NEW"}


def same_name(a, b):
    """True when two names have the same words apart from legal forms (SANDISK LLC
    ~ SANDISK CORP, but EQUITABLE GROUP is not EQUITABLE HOLDINGS)."""
    def words(s):
        return {w for w in tokens(norm(s), keep_drop=True) if w not in LEGAL}
    return bool(words(a)) and words(a) == words(b)


def acronym(name_toks):
    return "".join(w[0] for w in name_toks if w)


def norm(text):
    """Strip accents so tokens() keeps letters like the e in Estee."""
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()


def raw_words(text):
    return re.findall(r"[A-Z0-9]+", norm(text).upper())


def simple_tokens(text):
    """Like tokens() but without joining single letters, which can glue an
    initial onto the wrong word ("JOHN B SANFILIPPO" -> BSANFILIPPO)."""
    out = []
    for w in raw_words(re.sub(r"[`'’]", "", norm(text))):
        w = FOLD.get(w, w)
        if w not in DROP:
            out.append(w)
    return out


def variants(text):
    return [v for v in (tokens(norm(text)), simple_tokens(text)) if v]


def is_subsequence(short, long_):
    it = iter(long_)
    return all(ch in it for ch in short)


def same_word(a, b, first=False):
    """Equal, or one a cut-off of the other (CRSP truncates and abbreviates: MACHS,
    EST). A name's first word gets no 3-letter cut-offs (PAR is not PARAMETRIC)."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= (4 if first else 3) and long_.startswith(short):
        return True
    if len(short) >= 5 and long_.endswith(short):  # JCPENNEY ~ PENNEY
        return True
    n = len(os.path.commonprefix([a, b]))
    if len(short) >= 4 and n >= 4 and n >= 0.75 * len(short):
        return True
    # CRSP abbreviations keep the letters in order: INVT ~ INVESTMENT, CKG ~ COOKING.
    if a[0] == b[0] and len(short) >= (4 if first else 3) and len(long_) <= 3 * len(short) \
            and is_subsequence(short, long_):
        return True
    # Spelling slips: BANKFINACIAL ~ BANKFINANCIAL.
    return (len(short) >= 5 and a[0] == b[0] and len(long_) - len(short) <= 3
            and SequenceMatcher(None, a, b).ratio() >= 0.8)


def coverage(name, words):
    """Weighted share of name tokens found in words, 0 unless the name's first
    token is found. Runs of name tokens that spell one word count too
    (DU PONT ~ DUPONT)."""
    hit = [any(same_word(t, w, first=i == 0) for w in words) for i, t in enumerate(name)]
    for i in range(len(name)):
        for j in range(i + 2, min(len(name), i + 4) + 1):
            if "".join(name[i:j]) in words:
                hit[i:j] = [True] * (j - i)
    if not hit[0]:
        return 0.0
    weight = [0.5 if t in WEAK or len(t) <= 2 else 1.0 for t in name]
    return sum(w for w, h in zip(weight, hit) if h) / sum(weight)


def part_in_name(crsp, permno, parts, d):
    """True when every distinctive word of a title name part is in a CRSP name
    the PERMNO had by date d (DuPont ~ DU PONT E I DE NEMOURS)."""
    for since, comnam in crsp.comnams[permno]:
        if since > d + NAME_LEAD:
            continue
        for name in variants(comnam):
            joined = {"".join(name[i:j]) for i in range(len(name)) for j in range(i + 1, min(len(name), i + 4) + 1)}
            for part in parts:
                strong = [t for t in part if t not in WEAK and len(t) > 2]
                if strong and all(t in joined or any(same_word(t, w) for w in name) for t in strong):
                    return True
    return False


def spelled_in(compact, words):
    """True when a run of consecutive words spells compact (BBT in "BB T")."""
    for i in range(len(words)):
        acc = ""
        for w in words[i: i + 5]:
            acc += w
            if acc == compact:
                return True
            if not compact.startswith(acc):
                break
    return False


def name_score(crsp, permno, text, name_part=None, d=None, title=False):
    """How well the PERMNO's CRSP names fit a text: the better of the share of a
    CRSP name found in the text and the share of the text's name part found in
    the CRSP name (both first words must agree), over every name the PERMNO had by
    date d and two tokenisations."""
    if title:
        text = AT_HOST.sub(r"\1", HOSTED_BY.sub("", norm(text)))
    texts = variants(text)
    if not texts:
        return 0.0
    raw = raw_words(text)
    parts = [p for p in (name_part or []) if p]
    best = 0.0
    for since, comnam in crsp.comnams[permno]:
        if d is not None and since > d + NAME_LEAD:
            continue  # a later name (BANK OF NEW YORK MELLON) says nothing about a 2006 call
        for name in variants(comnam):
            for words in map(set, texts):
                best = max(best, coverage(name, words))
            for part in parts:
                if any(same_word(name[0], w, first=True) for w in part):
                    best = max(best, coverage(part, set(name)))
            # A title that names the company by its initials only ("IBM").
            ac = acronym(name)
            if len(name) >= 2 and len(ac) >= 2 and any(p == [ac] for p in parts):
                best = max(best, 0.9)
        compact = "".join(raw_words(comnam))
        for drop in sorted(DROP, key=len, reverse=True):
            if compact.endswith(drop) and len(compact) - len(drop) >= 3:
                compact = compact[: -len(drop)]
                break
        if len(compact) >= 3 and spelled_in(compact, raw):
            best = max(best, 1.0)
    return round(min(best, 1.0), 2)


ORDINAL_PERIOD = re.compile(
    r"^\s*(?:first|second|third|fourth|1st|2nd|3rd|4th)[\s-]+(?:fiscal\s+)?(?:quarter|half)\b", re.I)
MONTHS_WORDS = {"JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER",
                "OCTOBER", "NOVEMBER", "DECEMBER"}
# BOUNDARY without the ordinals, which start names too (First Health, Fourth Shift).
NAME_STOP = (BOUNDARY - {"FIRST", "SECOND", "THIRD", "FOURTH"}) | MONTHS_WORDS
# A title that names the company it is about: "<name> Earnings ...", "<name> at <conference>".
NAMED_CALL = re.compile(r"^(?P<name>.+?)\s+(?:earnings|results|sales|reports?|announces?|to\s+discuss|at|investor|analyst)\b", re.I)


def strip_period(title):
    """Drop leading period labels, including "Fourth Quarter and Full Year 2008"."""
    t = norm(title)
    while True:
        s = re.sub(r"^\s*(?:and|&|,|:)\s+", "", ORDINAL_PERIOD.sub("", title_core(t)), flags=re.I)
        if s == t:
            return t
        t = s


def title_parts(title):
    """Name-part tokens of the title (period label dropped, cut at event words), per tokenisation."""
    core = strip_period(title)
    out = []
    for v in variants(core):
        part = []
        for w in v:
            if w in NAME_STOP:
                break
            part.append(w)
        out.append(part)
    return out


def names_a_company(title):
    """True when the title has the shape "<name> Earnings/at/..." with a real name."""
    m = NAMED_CALL.match(strip_period(title))
    if not m:
        return False
    return any(w not in NAME_STOP and not w.isdigit() for w in tokens(m.group("name")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--stocknames", required=True)
    ap.add_argument("--quote", required=True)
    ap.add_argument("--cusip", required=True)
    ap.add_argument("--route-b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    crsp = Crsp(args.stocknames)

    quote = read_lseg(args.quote, QUOTE_COLS)
    quote = quote[(quote.AssetCategory == "ORD") & (quote.AssetType == "EQUITY")
                  & (quote.ExchangeCountry == "United States")]
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

    # ticker -> sorted name periods, for anchoring a RIC at its last-used date
    anchor_cache = {}

    def anchor(root, delist):
        key = (root, delist)
        if key not in anchor_cache:
            if delist:
                y, m = map(int, delist.split("-"))
                d = date(y, m, 1)
            else:
                d = CRSP_END
            rows = crsp.by_ticker.get(root, [])
            # the name period covering the anchor date, else the last one before it
            hit = crsp.on(rows, d)
            if not hit and rows:
                before = [r for r in rows if r.namedt <= d]
                if before:
                    last = max(r.nameenddt for r in before)
                    hit = [r for r in before if r.nameenddt == last]
            anchor_cache[key] = sorted({r.permno for r in hit})
        return anchor_cache[key]

    docs = pd.read_csv(args.docs, dtype=str, keep_default_na=False)
    route_b = pd.read_csv(args.route_b, dtype=str, keep_default_na=False)
    b_ids = set(route_b.document_id)

    tb_first, tb_permno = load_crsp(args.stocknames)
    clean_title = {"matched", "matched_other_name", "matched_near_date", "matched_multiclass",
                   "matched_other_name_multiclass", "matched_near_date_multiclass"}

    def title_fallback(rec, title, d):
        """When the ticker gives nothing or the wrong company, link by the company
        the event title names (title_crsp_match.py) if that match is clean, or a
        prefix match whose CRSP name also fits the company row (Limited Brands July
        Sales under L Brands)."""
        st, r, _ = match_one(title, d, tb_first, tb_permno)
        rec.update(title_status=st, title_permno=r.permno if r is not None else "",
                   title_comnam=r.comnam if r is not None else "")
        accept = st in clean_title or (
            st.startswith("matched_prefix") and "near_date" not in st
            and name_score(crsp, r.permno, rec["company_name"], variants(rec["company_name"])) >= NAME_OK)
        if accept:
            if rec.get("permno"):
                rec["other_methods"] = "; ".join(filter(None, [f"{rec['method']}:{rec['permno']}",
                                                               rec.get("other_methods", "")]))
            rec.update(status="matched", method="title_name", permno=r.permno, permco=r.permco,
                       ncusip8=r.ncusip[:8], crsp_ticker=r.ticker, comnam=r.comnam, shrcd=r.shrcd,
                       exchcd=r.exchcd, title_score=1.0, n_candidates=1, candidates="", note=rec.get("note", ""))
        return rec

    out = []
    for c in docs.itertuples(index=False):
        if c.document_id in b_ids:
            continue
        d_call = date.fromisoformat(c.call_date_ny)
        after_end = d_call > CRSP_END
        d = min(d_call, CRSP_END)
        root, cls, suffix, delist, foreign = parse_ric(c.ticker)
        rec = dict(document_id=c.document_id, company_id=c.company_id, company_name=c.company_name,
                   ric=c.ticker, root=root, share_class=cls, exch_suffix=suffix, delist=delist,
                   call_date=c.call_date_ny, event_title=c.event_title, route="A",
                   after_crsp_end=int(after_end))
        t_parts = title_parts(c.event_title)
        t_named = any(t_parts)

        if foreign:
            rec.update(status="foreign_listing")
            # A U.S. company whose only RIC is a foreign line (SanDisk as SNDK*.MX):
            # link by a clean title match to U.S. common stock, never to an ADR, and
            # only when the company row carries the same name, so foreign names that
            # collide with U.S. ones (Vector Ltd vs VECTOR GROUP LTD) and U.S.
            # targets filed under a foreign acquirer (Genentech under Roche) stay out.
            st, r, _ = match_one(c.event_title, d, tb_first, tb_permno)
            if st in clean_title and r.shrcd in COMMON and same_name(c.company_name, r.comnam):
                rec.update(status="matched", method="title_name", permno=r.permno, permco=r.permco,
                           ncusip8=r.ncusip[:8], crsp_ticker=r.ticker, comnam=r.comnam, shrcd=r.shrcd,
                           exchcd=r.exchcd, title_score=1.0, n_candidates=1, title_status=st,
                           title_permno=r.permno, title_comnam=r.comnam, note="foreign_ric_us_company")
            out.append(rec)
            continue
        if not root or not re.fullmatch(r"[A-Z0-9]{1,8}", root):
            rec.update(status="blank" if not root else "no_crsp_match", note="" if not root else "invalid_ticker")
            out.append(title_fallback(rec, c.event_title, d))
            continue

        tries = []  # (method, [rows])
        rows = pick(crsp.on(crsp.by_ticker.get(root, []), d), cls)
        if rows:
            tries.append(("ticker_date", rows))
        c8s = lseg_cusips(c.ticker, root)
        rec["lseg_cusip8s"] = ";".join(c8s)
        if c8s:
            rows = pick([r for c8 in c8s for r in crsp.on(crsp.by_ncusip.get(c8, []), d)], cls)
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
                    ps = sorted({p for c6 in c6s for p in crsp.permnos_by_cusip6.get(c6, ()) if crsp.listed(p, d)})
                    rows = pick([crsp.row_on(p, d) for p in ps], cls)
                    if rows:
                        tries.append(("lseg_cusip6", rows))
        ps = [p for p in anchor(root, delist) if crsp.listed(p, d)]
        rows = pick([crsp.row_on(p, d) for p in ps], cls)
        if rows:
            tries.append(("ticker_anchor", rows))

        if not tries:
            rec.update(status="no_crsp_match")
            out.append(title_fallback(rec, c.event_title, d))
            continue

        # A title that is just the ticker ("DHI at JPMorgan ...") names the RIC's
        # own company, i.e. the PERMNO the ticker anchors to.
        own = set(anchor(root, delist)) if any(p == [root] for p in t_parts) and len(root) >= 3 else set()

        def title_fit(permno):
            ts = name_score(crsp, permno, c.event_title, t_parts, d, title=True)
            return max(ts, 0.8) if permno in own else ts

        def judge(rows):
            sc = [(title_fit(r.permno),
                   name_score(crsp, r.permno, c.company_name, variants(c.company_name)), r)
                  for r in rows]
            sc.sort(key=lambda x: (-x[0], -x[1]))
            ts, cs, r = sc[0]
            ok = (ts >= NAME_OK or (not t_named and cs >= NAME_OK)
                  or (ts >= 0.5 and part_in_name(crsp, r.permno, t_parts, d)))
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
        rec.update(status=status, method=method, permno=r.permno, permco=r.permco, ncusip8=r.ncusip[:8],
                   crsp_ticker=r.ticker, comnam=r.comnam, shrcd=r.shrcd, exchcd=r.exchcd,
                   title_score=ts, company_score=cs, n_candidates=len(rows), note=note,
                   candidates="; ".join(f"{x.permno} {x.ticker} {x.shrcls} {x.comnam}".replace("  ", " ") for x in rows),
                   other_methods="; ".join(f"{m}:{'/'.join(sorted({x.permno for x in rs}))}"
                                           for m, rs in tries if m != method))
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
    later = (res.status.isin(["name_mismatch", "no_crsp_match", "blank"])
             & res.title_status.str.startswith("matched_prefix")
             & pd.Series([(co, p) in seen for co, p in zip(res.company_id, res.title_permno)], index=res.index))
    for i in res.index[later]:
        r = crsp.row_on(res.at[i, "title_permno"], min(date.fromisoformat(res.at[i, "call_date"]), CRSP_END))
        if res.at[i, "permno"]:
            res.at[i, "other_methods"] = "; ".join(filter(None, [f"{res.at[i, 'method']}:{res.at[i, 'permno']}",
                                                                 res.at[i, "other_methods"]]))
        res.loc[i, ["status", "method", "permno", "permco", "ncusip8", "crsp_ticker", "comnam", "shrcd", "exchcd",
                    "title_score", "n_candidates", "candidates", "note"]] = [
            "matched", "title_name", r.permno, r.permco, r.ncusip[:8], r.ticker, r.comnam, r.shrcd, r.exchcd,
            1.0, 1, "", "title_prefix_same_row"]

    # Route B: title-matched calls on acquirer and broker-host rows.
    b_status = {"needs_review": "name_mismatch", "ambiguous": "multiple", "no_match": "no_crsp_match",
                "no_name": "no_crsp_match"}
    rb = route_b.merge(docs[["document_id", "ticker"]].rename(columns={"ticker": "ric"}), on="document_id", how="left")
    b = pd.DataFrame(dict(
        document_id=rb.document_id, company_id=rb.company_id, company_name=rb.company_name, ric=rb.ric,
        call_date=rb.call_date, event_title=rb.event_title, route="B",
        status=[s if s in b_status else ("matched" if s.startswith("matched") else s) for s in rb.status],
        method="title_name:" + rb.status, permno=rb.permno, ncusip8=rb.ncusip.str[:8], crsp_ticker=rb.ticker, comnam=rb.comnam, candidates=rb.candidates,
        after_crsp_end=[int(date.fromisoformat(x) > CRSP_END) for x in rb.call_date]))
    b["status"] = [b_status.get(s, s) for s in b.status]
    res = pd.concat([res, b], ignore_index=True)
    res = res.sort_values("document_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    res.to_csv(args.out, index=False)

    n = len(res)
    print(f"{n:,} documents ({(res.route == 'B').sum():,} from route B)\n")
    print(res.status.value_counts().to_frame("docs").assign(pct=lambda x: (100 * x.docs / n).round(1)).to_string())
    print()
    print(res[res.permno.fillna("") != ""].groupby(["status", "method"]).size().to_string())


if __name__ == "__main__":
    main()
