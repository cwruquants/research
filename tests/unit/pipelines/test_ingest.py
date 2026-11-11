import time

import pytest

from research.pipelines import ingest as ingest_module
from research.pipelines.ingest import analyze_document
from research.transcripts.parse import load_document


def test_analyze_document(transcripts):
    out = analyze_document(transcripts / "spaced_T.xml")
    doc, attrs = out["document"], out["document_attr"]
    assert doc["file_name"] == "spaced_T.xml"
    assert doc["company_name"] == "Example Industries Inc"
    assert doc["start_date"] == "02-Feb-17 01:30PM GMT"
    assert attrs["num_words"] > 150 and attrs["num_sentences"] > 10
    assert {"flesch_reading_ease", "gunning_fog", "text_standard"} <= attrs.keys()
    assert "sentiment" not in attrs and "HIV4" not in attrs and out["doc_fit"] is None
    # the dictionary tone counts need no model, so they are always computed
    assert attrs["LM"] is not None and attrs["ML"] is not None


def test_analyze_document_from_bytes_matches_the_file(transcripts):
    path = transcripts / "tight_T.xml"
    from_bytes = analyze_document("tight_T.xml", xml=path.read_bytes())
    from_file = analyze_document(path)
    assert from_bytes["document_attr"] == from_file["document_attr"]
    assert from_bytes["document"]["file_name"] == "tight_T.xml"
    assert from_bytes["document"]["company_name"] == from_file["document"]["company_name"]


class ScoresEverything:
    """Stands in for SentimentSetup: fit_all gives every document the same scores."""

    def fit_all(self, doc):
        doc.sentiment = 0.25
        return doc


def test_a_transcript_with_no_spoken_text_gets_null_scores_not_zeros(tmp_path):
    stub = tmp_path / "stub_T.xml"
    stub.write_text(
        '<Event Id="1"><companyName>Example Industries Inc</companyName>'
        "<EventStory><Body>Example Industries Inc Q4 2016 Earnings Call</Body></EventStory></Event>",
        encoding="utf-8",
    )
    attrs = analyze_document(stub, sentiment=ScoresEverything())["document_attr"]
    assert attrs["num_words"] == 0 and attrs["num_sentences"] == 0
    for key in ("sentiment", "LM", "ML", "flesch_reading_ease", "text_standard", "difficult_words"):
        assert attrs[key] is None, key


def test_sentiment_scores_come_from_the_model_when_there_is_text(transcripts):
    attrs = analyze_document(transcripts / "spaced_T.xml", sentiment=ScoresEverything())["document_attr"]
    assert attrs["sentiment"] == 0.25
    assert "HIV4" not in attrs  # dropped 2026-11-11: stem matching made it unreliable in calls


@pytest.mark.parametrize("name", ["spaced_T.xml", "tight_T.xml", "no_headings_T.xml"])
def test_analyze_document_measures_the_spoken_text(transcripts, name):
    # the same words keyword matching reads: no header, participant lists or speaker lines
    attrs = analyze_document(transcripts / name)["document_attr"]
    assert attrs["num_words"] == len(load_document(transcripts / name).text.split())


def test_tight_and_spaced_layouts_give_the_same_measures(transcripts):
    # cleaning the whole body in one pass kept speaker names and dropped the answers
    # in the tight layout, which cut its word count
    spaced = analyze_document(transcripts / "spaced_T.xml")["document_attr"]
    tight = analyze_document(transcripts / "tight_T.xml")["document_attr"]
    assert tight["num_words"] == spaced["num_words"]
    assert tight["num_sentences"] == spaced["num_sentences"]
    assert tight["flesch_reading_ease"] == pytest.approx(spaced["flesch_reading_ease"])


class FakeUploader:
    instances = []

    def __init__(self, database_url, max_connections=1):
        self.rows = []
        FakeUploader.instances.append(self)

    def initialize_company_cache(self):
        return 0

    def upload_document(self, data, year):
        self.rows.append((data["doc_attrs"]["document"]["file_name"], year))
        return {"status": "success", "year": 2017, "year_fallback": False}

    def update_measures(self, data):
        self.rows.append((data["doc_attrs"]["document"]["file_name"], "measures only"))
        return {"status": "success", "year": 2017, "year_fallback": False}

    def finalize(self):
        pass


def test_measures_only_never_upserts(transcripts, monkeypatch):
    monkeypatch.setattr(ingest_module, "PostgresUploader", FakeUploader)
    summary = ingest_module.ingest(transcripts, 2016, database_url="postgresql://fake", measures_only=True)
    assert summary["num_uploaded"] == 3
    assert {how for _, how in FakeUploader.instances[-1].rows} == {"measures only"}


def test_measure_columns_leave_out_what_was_not_computed():
    from research.db.documents import PostgresUploader

    uploader = object.__new__(PostgresUploader)
    without_sentiment = uploader._measure_columns(
        {"num_words": 100, "num_sentences": 9, "flesch_reading_ease": 55.0}, only_present=True
    )
    assert set(without_sentiment) == {"num_words", "num_sentences", "flesch_reading_ease"}
    full = uploader._measure_columns({"num_words": 100})
    assert full["sentiment"] is None and len(full) == len(PostgresUploader._MEASURES)


@pytest.mark.parametrize("threads", [1, 3])
def test_ingest_uploads_every_file(transcripts, monkeypatch, threads):
    monkeypatch.setattr(ingest_module, "PostgresUploader", FakeUploader)
    summary = ingest_module.ingest(transcripts, 2016, database_url="postgresql://fake", num_threads=threads)
    assert summary["num_files_processed"] == summary["num_uploaded"] == 3
    assert summary["num_errors"] == 0
    assert sorted(FakeUploader.instances[-1].rows) == [
        ("no_headings_T.xml", 2016),
        ("spaced_T.xml", 2016),
        ("tight_T.xml", 2016),
    ]


def test_ingest_refuses_an_empty_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_module, "PostgresUploader", FakeUploader)
    with pytest.raises(FileNotFoundError):
        ingest_module.ingest(tmp_path, 2016, database_url="postgresql://fake")


def test_stop_on_error_drops_queued_files(tmp_path, monkeypatch):
    for i in range(20):
        (tmp_path / f"{i:02d}_T.xml").write_text("<Event/>", encoding="utf-8")

    def analyze(path, sentiment=None):
        if path.name == "00_T.xml":
            raise ValueError("bad file")
        time.sleep(0.05)
        return {"document": {"file_name": path.name}}

    monkeypatch.setattr(ingest_module, "PostgresUploader", FakeUploader)
    monkeypatch.setattr(ingest_module, "analyze_document", analyze)
    with pytest.raises(ValueError):
        ingest_module.ingest(
            tmp_path, 2016, database_url="postgresql://fake", num_threads=2, skip_on_error=False
        )
    assert len(FakeUploader.instances[-1].rows) < 5  # only the files already running
