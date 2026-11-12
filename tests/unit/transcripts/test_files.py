import hashlib
import zipfile
from datetime import datetime, timezone

import pytest

from research.transcripts.files import (
    EMPTY,
    HAS_NUL,
    MACOS_JUNK,
    NOT_UTF8,
    NOT_XML,
    Skipped,
    TranscriptFile,
    parse_last_update,
    read_source,
)


def vendor_xml(body="Hello.", last_update="Wednesday, January 9, 2019 at 4:23:16am GMT", action="publish"):
    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        f'<Event Id="9000001" lastUpdate="{last_update}" eventTypeId="1" eventTypeName="Earnings Call">\n'
        "<eventTitle>Q4 2016 Example Industries Inc Earnings Call</eventTitle>\n"
        f'<EventStory Id="9000001.F" expirationDate="x" action="{action}" storyType="transcript" '
        f'version="Final">\n<Body>{body}</Body>\n</EventStory>\n</Event>\n'
    ).encode("utf-8")


DELETE_STUB = (
    b'<Event Id="9000002" lastUpdate="Monday, March 1, 2021 at 12:05:00pm GMT" eventTypeId="1" '
    b'eventTypeName="Earnings Call"><EventStory Id="9000002.F" expirationDate="x" action="delete" '
    b'storyType="transcript" version="Final" /></Event>'
)


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "2019.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("2019/9000001_T.xml", vendor_xml())
        z.writestr("2020/9000001_T.xml", vendor_xml(body="Hello, corrected."))  # a later revision
        z.writestr("2019/9000002_T.xml", DELETE_STUB)
        z.writestr("2019/9000003_T.xml", b"<Event Id=9000003><Body>unclosed")  # not well-formed XML
        z.writestr("2019/9000004_T.xml", b"")
        z.writestr("2019/9000005_T.xml", b"<Event>caf\xe9</Event>")  # Latin-1, not UTF-8
        z.writestr("2019/9000006_T.xml", b"<Event>a\x00b</Event>")
        z.writestr("__MACOSX/2019/._9000001_T.xml", b"\x00\x05\x16\x07")
        z.writestr("2019/notes.txt", b"not a transcript")
    return path


def test_read_source_keeps_every_text_file_and_says_why_it_skips_the_rest(archive):
    items = list(read_source(archive))
    files = {f.member: f for f in items if isinstance(f, TranscriptFile)}
    skipped = {s.member: s.reason for s in items if isinstance(s, Skipped)}

    assert set(files) == {
        "2019/9000001_T.xml",
        "2020/9000001_T.xml",
        "2019/9000002_T.xml",
        "2019/9000003_T.xml",
    }
    assert skipped == {
        "2019/9000004_T.xml": EMPTY,
        "2019/9000005_T.xml": NOT_UTF8,
        "2019/9000006_T.xml": HAS_NUL,
        "__MACOSX/2019/._9000001_T.xml": MACOS_JUNK,
        "2019/notes.txt": NOT_XML,
    }


def test_a_file_carries_its_hash_text_and_revision_fields(archive):
    files = {f.member: f for f in read_source(archive) if isinstance(f, TranscriptFile)}
    first = files["2019/9000001_T.xml"]
    raw = vendor_xml()
    assert first.source_file == "9000001_T.xml" and first.archive == "2019.zip"
    assert first.sha256 == hashlib.sha256(raw).digest() and first.byte_length == len(raw)
    assert first.xml.encode("utf-8") == raw  # the stored text reproduces the file
    assert first.last_update == datetime(2019, 1, 9, 4, 23, 16, tzinfo=timezone.utc)
    assert (first.story_action, first.story_version, first.story_type) == ("publish", "Final", "transcript")
    assert first.folder_year == 2019

    revision = files["2020/9000001_T.xml"]
    assert revision.source_file == first.source_file and revision.sha256 != first.sha256
    assert revision.folder_year == 2020


def test_withdrawn_and_malformed_files_are_kept(archive):
    files = {f.member: f for f in read_source(archive) if isinstance(f, TranscriptFile)}
    stub = files["2019/9000002_T.xml"]
    assert stub.story_action == "delete"
    assert stub.last_update == datetime(2021, 3, 1, 12, 5, 0, tzinfo=timezone.utc)
    malformed = files["2019/9000003_T.xml"]
    assert malformed.last_update is None and malformed.story_action is None


def test_already_loaded_members_are_not_read_again(archive):
    files = [
        f
        for f in read_source(archive, already_loaded={"2019/9000001_T.xml", "2019/9000002_T.xml"})
        if isinstance(f, TranscriptFile)
    ]
    assert {f.member for f in files} == {"2020/9000001_T.xml", "2019/9000003_T.xml"}


def test_a_folder_reads_like_a_zip(tmp_path):
    (tmp_path / "2019").mkdir()
    (tmp_path / "2019" / "9000001_T.xml").write_bytes(vendor_xml())
    [only] = read_source(tmp_path)
    assert (only.archive, only.member, only.folder_year) == (tmp_path.name, "2019/9000001_T.xml", 2019)


@pytest.mark.parametrize(
    "stamp, expected",
    [
        ("Wednesday, January 9, 2019 at 4:23:16am GMT", datetime(2019, 1, 9, 4, 23, 16, tzinfo=timezone.utc)),
        ("Friday, July 31, 2024 at 12:00:01am GMT", datetime(2024, 7, 31, 0, 0, 1, tzinfo=timezone.utc)),
        ("Friday, July 31, 2024 at 12:30:00pm GMT", datetime(2024, 7, 31, 12, 30, 0, tzinfo=timezone.utc)),
        ("Friday, July 31, 2024 at 11:59:59pm GMT", datetime(2024, 7, 31, 23, 59, 59, tzinfo=timezone.utc)),
        ("31-Jul-24 1:30pm GMT", None),
        ("Friday, Smarch 31, 2024 at 1:00:00pm GMT", None),
        ("Friday, February 30, 2024 at 1:00:00pm GMT", None),
        (None, None),
    ],
)
def test_parse_last_update(stamp, expected):
    assert parse_last_update(stamp) == expected
