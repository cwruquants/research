import contextlib

from research import config
from research.matching.direct import MatchingAgent
from research.pipelines.backfill_exposures import process_batch


class FakeConn:
    """Stands in for psycopg: records the run columns and the COPY rows."""

    def __init__(self):
        self.runs, self.matches = None, []

    @contextlib.contextmanager
    def transaction(self):
        yield

    def cursor(self):
        conn = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, columns):
                conn.runs = columns

            def fetchall(self):
                return [(100 + i, d, k) for i, (d, k) in enumerate(zip(conn.runs[0], conn.runs[1]))]

            @contextlib.contextmanager
            def copy(self, sql):
                yield type("Writer", (), {"write_row": lambda _, row: conn.matches.append(row)})()

        return Cursor()


def test_keyword_sets_config_lists_the_db_sets():
    sets = config.keyword_sets()
    assert {"risk", "activeness"} <= sets.keys()
    assert all(ks.file.exists() for ks in sets.values())
    assert sets["risk"].matching_options()["exclude_safe_harbor"] is False
    # the rerun's sets: the approved risk list and both noise filters
    rerun = sets["risk_2026_11"].matching_options()
    assert rerun["keywords_path"].endswith("Risk_v2.csv") and rerun["exclude_safe_harbor"] is True
    assert rerun["exclude_phrases_path"].endswith("Risk_v2_exclusions.csv") and rerun["normalize_phrases"]
    activeness = sets["activeness_2026_11"].matching_options()
    assert activeness["exclude_phrases_path"].endswith("activeness_exclusions.csv")
    assert activeness["keywords_path"].endswith("activeness_v2.csv")


def test_process_batch_writes_one_run_per_set(transcripts, tmp_path):
    words = tmp_path / "words.csv"
    words.write_text("risk\ntrade war\ntaxes\n", encoding="utf-8")
    agents = {"demo": MatchingAgent(keywords_path=str(words))}
    batch = [
        {
            "document_id": 7,
            "source_file": "spaced_T.xml",
            "keyword_sets": [(3, "demo")],
            "location": ("dir", str(transcripts / "spaced_T.xml"), None),
        }
    ]

    conn = FakeConn()
    result = process_batch(batch, "20260101_000000", agents=agents, conn=conn, run_id=12)

    assert result["runs_written"] == 1 and result["failures"] == []
    document_ids, set_ids, methods = conn.runs[0], conn.runs[1], conn.runs[2]
    assert (document_ids, set_ids, methods) == ([7], [3], ["direct"])
    assert (conn.runs[-2], conn.runs[-1]) == ([12], ["20260101_000000"])  # pipeline run, then timestamp
    keywords = [row[1] for row in conn.matches]
    assert keywords.count("trade war") == 2 and "taxes" in keywords and "risk" in keywords
    assert all(row[0] == 100 for row in conn.matches)  # every match points at the new run id
    assert result["matches_written"] == len(conn.matches)


def test_process_batch_reports_empty_transcripts(tmp_path):
    empty = tmp_path / "empty_T.xml"
    empty.write_text("<Event><EventStory><Body></Body></EventStory></Event>", encoding="utf-8")
    batch = [
        {
            "document_id": 1,
            "source_file": "empty_T.xml",
            "keyword_sets": [(1, "x")],
            "location": ("dir", str(empty), None),
        }
    ]
    result = process_batch(batch, "ts", agents={}, conn=FakeConn())
    assert result["runs_written"] == 0 and len(result["failures"]) == 1
