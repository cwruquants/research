import pytest

from research.transcripts.parse import clean_ticker, get_company_metadata, load_document
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


def test_unreadable_file_gives_empty_document(tmp_path):
    bad = tmp_path / "bad_T.xml"
    bad.write_text("<Event><Body>unclosed", encoding="utf-8")
    assert load_document(bad).text == ""


@pytest.mark.parametrize("raw, expected", [("AAPL.OQ", "AAPL.OQ"), ("  ", None), ("\\n  ", None), (None, None)])
def test_clean_ticker(raw, expected):
    assert clean_ticker(raw) == expected


def test_sources_find_files_in_folders_and_zips(transcripts, tmp_path):
    import zipfile

    archive = tmp_path / "2017.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(transcripts / "tight_T.xml", "2017/tight_T.xml")
    index = index_sources([str(transcripts), str(archive)])
    assert {kind for kind, _, _ in index["tight_T.xml"]} == {"dir", "zip"}

    doc = {"source_file": "tight_T.xml", "year": 2017,
           "event_title": "Q4 2016 Example Industries Inc Earnings Conference Call"}
    location, problem = resolve_location(doc, index["tight_T.xml"])
    # both copies carry the same title, so the year folder decides
    assert location[0] == "zip" and problem == ""
    with open_transcript(location) as fh:
        assert load_document(fh).text
