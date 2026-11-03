import time

import pytest

from research.pipelines import ingest as ingest_module
from research.pipelines.ingest import analyze_document


def test_analyze_document(transcripts):
    out = analyze_document(transcripts / "spaced_T.xml")
    doc, attrs = out["document"], out["document_attr"]
    assert doc["file_name"] == "spaced_T.xml"
    assert doc["company_name"] == "Example Industries Inc"
    assert doc["start_date"] == "02-Feb-17 01:30PM GMT"
    assert attrs["num_words"] > 150 and attrs["num_sentences"] > 10
    assert {"flesch_reading_ease", "gunning_fog", "text_standard"} <= attrs.keys()
    assert "sentiment" not in attrs and out["doc_fit"] is None


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

    def finalize(self):
        pass


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
