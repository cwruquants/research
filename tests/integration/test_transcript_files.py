"""Load, reload and verify against a throwaway Postgres (TEST_DATABASE_URL) that has the schema."""

import zipfile
from uuid import uuid4

import psycopg
import pytest

from research.pipelines.load_transcript_files import load_sources
from research.pipelines.verify_transcript_files import verify


def xml(event_id, body, last_update, action="publish", story_type="transcript"):
    return (
        f'<Event Id="{event_id}" lastUpdate="{last_update}" eventTypeId="1" eventTypeName="Earnings Call">'
        f'<EventStory Id="{event_id}.F" expirationDate="x" action="{action}" storyType="{story_type}" '
        f'version="Final"><Body>{body}</Body></EventStory></Event>'
    ).encode("utf-8")


JAN, FEB, MAR = (f"Monday, {month} 4, 2019 at 9:00:00am GMT" for month in ("January", "February", "March"))


@pytest.mark.db
def test_load_is_idempotent_and_the_view_picks_the_latest_published_transcript(test_database_url, tmp_path):
    tag = uuid4().hex[:12]
    names = {key: f"{tag}{n}_T.xml" for n, key in enumerate(("revised", "withdrawn", "republished", "tied"))}
    brief = f"{tag}0_B.xml"
    archive = tmp_path / f"{tag}.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(f"2019/{names['revised']}", xml(1, "first", JAN))
        z.writestr(f"2020/{names['revised']}", xml(1, "corrected, with an accent: café", FEB))
        z.writestr(f"2021/{names['revised']}", xml(1, "first", JAN))  # byte-identical second copy
        z.writestr(f"2019/{names['withdrawn']}", xml(2, "published", JAN))
        z.writestr(f"2020/{names['withdrawn']}", xml(2, "", FEB, action="delete"))
        z.writestr(f"2019/{names['republished']}", xml(3, "published", JAN))
        z.writestr(f"2020/{names['republished']}", xml(3, "", FEB, action="delete"))
        z.writestr(f"2021/{names['republished']}", xml(3, "published again", MAR))
        z.writestr(f"2019/{names['tied']}", xml(4, "copy in the earlier folder", JAN))
        z.writestr(f"2020/{names['tied']}", xml(4, "copy in the later folder", JAN))
        z.writestr(f"2019/{brief}", xml(1, "a brief, not a transcript", MAR, story_type="brief"))
        z.writestr("__MACOSX/2019/._junk_T.xml", b"\x00\x05")
        z.writestr("2019/empty_T.xml", b"")

    # Each step uses its own connection, as the commands do: a load that is not
    # committed must not look loaded to the next step
    first, skipped = load_sources([str(archive)], test_database_url)
    assert first["inserted"] == 10 and first["identical_copy"] == 1
    assert [s.reason for s in skipped] == ["empty file"]

    again, _ = load_sources([str(archive)], test_database_url)
    assert again["inserted"] == 0 and again["already_loaded"] == 10

    counted, _ = load_sources([str(archive)], None)  # a dry run reads and counts only
    assert counted["read"] == 11 and counted["inserted"] == 0

    with psycopg.connect(test_database_url) as conn:
        result = verify(conn, [str(archive)])
        assert result["ok"] and result["distinct_revisions"] == 10

        rows = conn.execute(
            "SELECT source_file, xml, folder_year, withdrawn_at IS NOT NULL "
            "FROM public.transcript_files_latest WHERE source_file LIKE %s",
            [f"{tag}%"],
        ).fetchall()
        latest = {source_file: (text, year, withdrawn) for source_file, text, year, withdrawn in rows}
        assert set(latest) == set(names.values())  # the brief is never a call's transcript
        assert "café" in latest[names["revised"]][0]
        assert latest[names["revised"]][2] is False
        assert "published" in latest[names["withdrawn"]][0] and latest[names["withdrawn"]][2] is True
        assert "published again" in latest[names["republished"]][0]
        assert latest[names["republished"]][2] is False
        assert latest[names["tied"]][1] == 2020  # a tie on the stamp goes to the later year folder
