"""
Name similarity between event titles, company rows and CRSP names (used by route_a).

A CRSP name fits a text when enough of its words are found there, or enough of the
text's name words are found in it, with both first words required to agree. Words
match when equal, cut off, abbreviated in order (INVT ~ INVESTMENT) or misspelled;
initials and spaced letters are joined (B B & T ~ BB&T). See route_a's docstring.
"""

import os
import re
import unicodedata
from datetime import timedelta
from difflib import SequenceMatcher

from research.linking.crsp_names import BOUNDARY, DROP, FOLD, title_core, tokens

NAME_OK = 0.6
NAME_LEAD = timedelta(days=30)
# "Dell Technologies Inc Investor Call Hosted by Morgan Stanley": the host is not the subject.
HOSTED_BY = re.compile(r"\b(?:hosted|sponsored|moderated)(?:\s+by)?\b.*$", re.I)
# "Salesforce.com Inc at Bank of America ... Conference": score only the part before "at".
AT_HOST = re.compile(r"^(.*?\w.*?)\s+at\s+.*$", re.I)
# Words too common to count as evidence that a title names the company.
WEAK = {
    "AMERN",
    "AMER",
    "NATL",
    "INTL",
    "FIRST",
    "UNITED",
    "GENERAL",
    "BANCORP",
    "BANK",
    "TRUST",
    "FINL",
    "IND",
    "SYS",
    "TECH",
    "SVCS",
    "RES",
    "ENERGY",
    "CAPITAL",
    "US",
    "USA",
}


# Legal forms ignored when comparing a company row name with a CRSP name exactly.
LEGAL = {
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "CO",
    "COMPANY",
    "LTD",
    "LIMITED",
    "LLC",
    "LP",
    "PLC",
    "THE",
    "AND",
    "DE",
    "DEL",
    "NEW",
}


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
    if (
        a[0] == b[0]
        and len(short) >= (4 if first else 3)
        and len(long_) <= 3 * len(short)
        and is_subsequence(short, long_)
    ):
        return True
    # Spelling slips: BANKFINACIAL ~ BANKFINANCIAL.
    return (
        len(short) >= 5
        and a[0] == b[0]
        and len(long_) - len(short) <= 3
        and SequenceMatcher(None, a, b).ratio() >= 0.8
    )


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
            joined = {
                "".join(name[i:j]) for i in range(len(name)) for j in range(i + 1, min(len(name), i + 4) + 1)
            }
            for part in parts:
                strong = [t for t in part if t not in WEAK and len(t) > 2]
                if strong and all(t in joined or any(same_word(t, w) for w in name) for t in strong):
                    return True
    return False


def spelled_in(compact, words):
    """True when a run of consecutive words spells compact (BBT in "BB T")."""
    for i in range(len(words)):
        acc = ""
        for w in words[i : i + 5]:
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
    r"^\s*(?:first|second|third|fourth|1st|2nd|3rd|4th)[\s-]+(?:fiscal\s+)?(?:quarter|half)\b", re.I
)
MONTHS_WORDS = {
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
}
# BOUNDARY without the ordinals, which start names too (First Health, Fourth Shift).
NAME_STOP = (BOUNDARY - {"FIRST", "SECOND", "THIRD", "FOURTH"}) | MONTHS_WORDS
# A title that names the company it is about: "<name> Earnings ...", "<name> at <conference>".
NAMED_CALL = re.compile(
    r"^(?P<name>.+?)\s+(?:earnings|results|sales|reports?|announces?|to\s+discuss|at|investor|analyst)\b",
    re.I,
)


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
