from xml.etree.ElementTree import ParseError
from xml.sax.saxutils import escape

import pytest

from research.transcripts.parse import clean_ticker, get_company_metadata, load_document, spoken_text
from research.transcripts.sources import index_sources, open_transcript, resolve_location


def test_metadata(transcripts):
    meta = get_company_metadata(transcripts / "spaced_T.xml")
    assert meta["companyName"] == "Example Industries Inc"
    assert meta["companyTicker"] == "EXMPL.OQ"
    assert meta["headlineTicker"] == "EXMPL.OQ"
    assert meta["startDate"] == "02-Feb-17 01:30PM GMT"
    assert (meta["quarter"], meta["year"]) == ("Q4", "2016")


@pytest.mark.parametrize("name", ["spaced_T.xml", "tight_T.xml", "no_headings_T.xml"])
def test_load_document_keeps_spoken_text_only(transcripts, name):
    text = load_document(transcripts / name).text
    assert "Revenue grew 8% this quarter" in text  # presentation
    assert "contracts in place through the summer" in text  # an answer in Q&A
    assert "Dana Park" not in text and "[" not in text  # speaker lines
    assert "Corporate Participants" not in text


def test_tight_and_spaced_layouts_give_the_same_words(transcripts):
    spaced = load_document(transcripts / "spaced_T.xml").text.split()
    tight = load_document(transcripts / "tight_T.xml").text.split()
    assert spaced == tight


BAR, RULE = "=" * 80, "-" * 80
OPENING = [
    ("Operator   [1]", "Welcome to the call."),
    ("Dana Park, Example Inc - CEO   [2]", "Revenue grew 8%."),
]
CLOSING = [("Lee Moran, Example Inc - CFO   [3]", "Margin improved. We expect more orders.")]
QUESTIONS = [("Sam Ortiz, Harbor - Analyst   [4]", "How exposed are you?"), ("Dana Park   [5]", "A fifth.")]
FOLLOW_UP = [
    ("Sam Ortiz, Harbor - Analyst   [6]", "And on taxes?"),
    ("Lee Moran   [7]", "Two cents a share."),
]


def _transcript(tmp_path, sections, tight):
    """A synthetic transcript whose body has these (heading, [(speaker line, speech), ...]) sections."""
    gap = "" if tight else "\n"
    body = (
        "Example Inc Q4 2016 Earnings Call\n\nTEXT version of Transcript\n\n"
        f"{BAR}\nCorporate Participants\n{BAR}\n   *  Dana Park\n      Example Inc - CEO\n\n"
        f"{BAR}\nConference Call Participants\n{BAR}\n   *  Sam Ortiz\n      Harbor - Analyst\n\n"
    )
    for heading, turns in sections:
        body += f"{BAR}\n{heading}\n"
        body += "".join(f"{RULE}\n{speaker}\n{RULE}\n{gap}{speech}\n{gap}" for speaker, speech in turns)
    path = tmp_path / "layout_T.xml"
    path.write_text(f"<Event><EventStory><Body>{escape(body)}</Body></EventStory></Event>", encoding="utf-8")
    return path


@pytest.mark.parametrize("tight", [False, True])
@pytest.mark.parametrize(
    "sections",
    [
        [("Presentation", OPENING + CLOSING), ("q and a", QUESTIONS)],
        [("Presentation", OPENING + CLOSING), ("Question and Answers", QUESTIONS)],
        [("Transcript", OPENING + CLOSING), ("Questions and Answers", QUESTIONS)],
        [("Opening", OPENING + CLOSING), ("Q-and-A", QUESTIONS)],
        # A call in two parts repeats the headings
        [
            ("Presentation", OPENING),
            ("Questions and Answers", QUESTIONS),
            ("Presentation", CLOSING),
            ("Questions and Answers", FOLLOW_UP),
        ],
        # A speaker line that is only "Operator" is a speaker line like any other
        [
            ("Presentation", [("Operator", "Welcome to the call.")] + CLOSING),
            ("Questions and Answers", QUESTIONS),
        ],
    ],
    ids=["q and a", "Question and Answers", "Transcript then Q&A", "Opening", "two parts", "bare Operator"],
)
def test_every_section_is_kept_whatever_its_heading(tmp_path, sections, tight):
    text = spoken_text(_transcript(tmp_path, sections, tight))
    spoken = [speech for _, turns in sections for _, speech in turns]
    assert text.split() == " ".join(spoken).split()
    if tight:  # no blank lines inside a section, so each block is one section
        assert len(text.split("\n\n")) == len(sections)


def test_unreadable_file_gives_empty_document(tmp_path):
    bad = tmp_path / "bad_T.xml"
    bad.write_text("<Event><Body>unclosed", encoding="utf-8")
    assert load_document(bad).text == ""
    with pytest.raises(ParseError):  # spoken_text leaves the error to its caller
        spoken_text(bad)


@pytest.mark.parametrize(
    "raw, expected", [("AAPL.OQ", "AAPL.OQ"), ("  ", None), ("\\n  ", None), (None, None)]
)
def test_clean_ticker(raw, expected):
    assert clean_ticker(raw) == expected


def test_sources_find_files_in_folders_and_zips(transcripts, tmp_path):
    import zipfile

    archive = tmp_path / "2017.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(transcripts / "tight_T.xml", "2017/tight_T.xml")
    index = index_sources([str(transcripts), str(archive)])
    assert {kind for kind, _, _ in index["tight_T.xml"]} == {"dir", "zip"}

    doc = {
        "source_file": "tight_T.xml",
        "year": 2017,
        "event_title": "Q4 2016 Example Industries Inc Earnings Conference Call",
    }
    location, problem = resolve_location(doc, index["tight_T.xml"])
    # both copies carry the same title, so the year folder decides
    assert location[0] == "zip" and problem == ""
    with open_transcript(location) as fh:
        assert load_document(fh).text
