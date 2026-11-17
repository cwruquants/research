"""
Route A's name helpers on synthetic names. These pin the matching rules the module
docstring describes, so a refactor of route_a.py (or crsp_names.py) that changes them
fails here before it shows up as a diff in the linking golden output.
"""

from datetime import date

import pytest

from research.linking.route_a import (
    coverage,
    name_score,
    names_a_company,
    norm,
    parse_ric,
    same_name,
    same_word,
    spelled_in,
    strip_period,
    title_parts,
)


@pytest.mark.parametrize(
    "ric, expected",
    [
        ("AAPL.OQ", ("AAPL", "", "OQ", "", False)),
        ("AAPL", ("AAPL", "", "", "", False)),
        ("OCR^H15", ("OCR", "", "", "2015-08", False)),
        ("WCIMQ.PK^I09", ("WCIMQ", "", "PK", "2009-09", False)),
        ("*SNDK.MX", ("SNDK", "", "MX", "", True)),
        # share classes: a single-letter suffix after a U.S. root, lower case, or a space
        ("BRK.A", ("BRK", "A", "", "", False)),
        ("BF.B.N", ("BF", "B", "N", "", False)),
        ("LGFa.N", ("LGF", "A", "N", "", False)),
        ("BRKb", ("BRK", "B", "", "", False)),
        ("ERIC B.ST", ("ERIC", "B", "ST", "", True)),
        ("FMER PR A", ("FMER", "PREF", "", "", False)),
        ("AMLp", ("AML", "PREF", "", "", False)),
        ("INN_u.N", ("INN", "PREF", "N", "", False)),
        ("", ("", "", "", "", False)),
        (None, ("", "", "", "", False)),
    ],
)
def test_parse_ric(ric, expected):
    assert parse_ric(ric) == expected


@pytest.mark.parametrize(
    "a, b, first, expected",
    [
        ("MACHINES", "MACHINES", False, True),
        ("MACHS", "MACHINES", False, True),  # CRSP cuts words off
        ("INVT", "INVESTMENT", False, True),  # and abbreviates them in order
        ("CKG", "COOKING", False, True),
        ("PENNEY", "JCPENNEY", False, True),  # a long enough word at the end
        ("BANKFINACIAL", "BANKFINANCIAL", False, True),  # spelling slips
        ("PAR", "PARAMETRIC", False, True),
        ("PAR", "PARAMETRIC", True, False),  # but a name's first word gets no 3-letter cut-off
        ("APPLE", "AMAZON", False, False),
        ("GAS", "GRAPE", False, False),  # no S after the A
        # The in-order abbreviation rule is loose: GAS is read as short for GLASS.
        # Pinned as it is; tightening it would change links.
        ("GAS", "GLASS", False, True),
    ],
)
def test_same_word(a, b, first, expected):
    assert same_word(a, b, first=first) is expected
    assert same_word(b, a, first=first) is expected


def test_same_name_ignores_legal_forms_only():
    assert same_name("SANDISK LLC", "SanDisk Corp")
    assert same_name("The Example Company", "EXAMPLE INC")
    assert not same_name("EQUITABLE GROUP", "EQUITABLE HOLDINGS")
    assert not same_name("INC", "CORP")  # no words left to compare


def test_coverage():
    assert coverage(["EXAMPLE", "WIDGETS"], {"EXAMPLE", "WIDGETS", "Q4"}) == 1.0
    # the first word must be found, or nothing counts
    assert coverage(["EXAMPLE", "WIDGETS"], {"WIDGETS"}) == 0.0
    # weak words weigh half: 1 of 1.5
    assert coverage(["EXAMPLE", "FIRST"], {"EXAMPLE"}) == pytest.approx(2 / 3)
    # a run of name words that spells one word counts (DU PONT ~ DUPONT)
    assert coverage(["DU", "PONT"], {"DUPONT"}) == 1.0


def test_spelled_in():
    assert spelled_in("BBT", ["B", "B", "T", "CORP"])
    assert spelled_in("BBT", ["Q4", "BB", "T"])
    assert not spelled_in("BBT", ["B", "T", "B"])


def test_norm_strips_accents():
    assert norm("Estée Lauder") == "Estee Lauder"
    assert norm(None) == ""


def test_strip_period_drops_leading_period_labels():
    assert strip_period("Fourth Quarter and Full Year 2008 Example Widgets Earnings Call").startswith(
        "Example"
    )
    assert strip_period("First Health Group Earnings Call").startswith("First Health")


def test_title_parts_cut_at_event_words():
    assert ["EXAMPLE", "WIDGETS"] in title_parts("Q4 2016 Example Widgets Inc Earnings Conference Call")


def test_names_a_company():
    assert names_a_company("Q4 2016 Example Widgets Inc Earnings Conference Call")
    assert names_a_company("Example Widgets Inc at Example Bank Technology Conference")
    assert not names_a_company("Example Bank Global Healthcare Conference")


class FakeCrsp:
    """The one thing name_score reads: each PERMNO's CRSP names with their first dates."""

    def __init__(self, comnams):
        self.comnams = comnams


CRSP = FakeCrsp(
    {
        1: [(date(1990, 1, 1), "EXAMPLE WIDGETS INC")],
        2: [(date(1990, 1, 1), "OTHER GADGETS CORP")],
        3: [(date(1990, 1, 1), "INTERNATIONAL BUSINESS MACHINES")],
        # renamed in 2010: the later name says nothing about a 2006 call
        4: [(date(1990, 1, 1), "OLD NAME HOLDINGS"), (date(2010, 1, 1), "EXAMPLE WIDGETS INC")],
    }
)
TITLE = "Q4 2016 Example Widgets Inc Earnings Conference Call"


def test_name_score_fits_the_right_company():
    parts = title_parts(TITLE)
    assert name_score(CRSP, 1, TITLE, parts, title=True) == 1.0
    assert name_score(CRSP, 2, TITLE, parts, title=True) == 0.0


def test_name_score_ignores_the_host():
    title = "Other Gadgets Corp at Example Widgets Technology Conference"
    assert name_score(CRSP, 2, title, title_parts(title), title=True) == 1.0
    assert name_score(CRSP, 1, title, title_parts(title), title=True) == 0.0


def test_name_score_takes_initials_for_the_company():
    title = "IBM Earnings Conference Call"
    assert name_score(CRSP, 3, title, title_parts(title), title=True) >= 0.9


def test_name_score_ignores_names_from_after_the_call():
    title = "Example Widgets Inc Earnings Conference Call"
    parts = title_parts(title)
    assert name_score(CRSP, 4, title, parts, date(2006, 5, 1), title=True) == 0.0
    assert name_score(CRSP, 4, title, parts, date(2012, 5, 1), title=True) == 1.0
