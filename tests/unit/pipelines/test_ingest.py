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
    assert {"flesch_reading_ease", "gunning_fog", "difficult_words"} <= attrs.keys()
    assert not {"linsear_write_formula", "text_standard"} & attrs.keys()  # dropped 2026-11-11
    assert "sentiment" not in attrs and "HIV4" not in attrs
    # the dictionary tone counts need no model, so they are always computed
    assert attrs["LM"] is not None and attrs["ML"] is not None
    assert attrs["LM"] == attrs["LM_positive"] - attrs["LM_negative"]
    assert (doc["event_type_id"], doc["event_type"]) == ("1", "Earnings Conference Call")


def test_analyze_document_from_bytes_matches_the_file(transcripts):
    path = transcripts / "tight_T.xml"
    from_bytes = analyze_document("tight_T.xml", xml=path.read_bytes())
    from_file = analyze_document(path)
    assert from_bytes["document_attr"] == from_file["document_attr"]
    assert from_bytes["document"]["file_name"] == "tight_T.xml"
    assert from_bytes["document"]["company_name"] == from_file["document"]["company_name"]


class ScoresEverything:
    """Stands in for SentimentSetup: the first `ones` sentences score 1, the rest 0."""

    def __init__(self, ones=1):
        self.ones, self.calls = ones, []

    def sentence_scores(self, texts):
        self.calls.append(list(texts))
        return [1.0 if i < self.ones else 0.0 for i in range(len(texts))]


def test_a_transcript_with_no_spoken_text_gets_null_scores_not_zeros(tmp_path):
    stub = tmp_path / "stub_T.xml"
    stub.write_text(
        '<Event Id="1"><companyName>Example Industries Inc</companyName>'
        "<EventStory><Body>Example Industries Inc Q4 2016 Earnings Call</Body></EventStory></Event>",
        encoding="utf-8",
    )
    attrs = analyze_document(stub, sentiment=ScoresEverything())["document_attr"]
    assert attrs["num_words"] == 0 and attrs["num_sentences"] == 0
    for key in ("sentiment", "LM", "LM_positive", "ML_negative", "flesch_reading_ease", "gunning_fog"):
        assert attrs[key] is None, key


def test_sentiment_is_the_mean_over_every_sentence(transcripts):
    scorer = ScoresEverything(ones=3)
    out = analyze_document(transcripts / "spaced_T.xml", sentiment=scorer)
    [texts] = scorer.calls  # the whole call in one request
    assert len(texts) == out["document_attr"]["num_sentences"] > 3
    assert out["document_attr"]["sentiment"] == pytest.approx(3 / len(texts))


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
        self.rows, self.measures, self.runs = [], [], []
        FakeUploader.instances.append(self)

    def start_run(self, step, settings):
        self.runs.append((step, settings))
        return 41

    def initialize_company_cache(self):
        return 0

    def upload_document(self, doc, year):
        self.rows.append((doc["document"]["file_name"], year))
        self.measures.append(doc["document_attr"])
        return {"status": "success", "year": 2017, "year_fallback": False}

    def update_measures(self, doc):
        self.rows.append((doc["document"]["file_name"], "measures only"))
        return {"status": "success", "year": 2017, "year_fallback": False}

    def finalize(self):
        pass


def test_every_row_names_the_run_that_made_it(transcripts, monkeypatch):
    monkeypatch.setattr(ingest_module, "PostgresUploader", FakeUploader)
    ingest_module.ingest(transcripts, 2016, database_url="postgresql://fake")
    uploader = FakeUploader.instances[-1]
    [(step, settings)] = uploader.runs
    assert step == "ingest" and settings["sentiment"] is None and not settings["measures_only"]
    lm = settings["lexicon"]["LM"]["negative"]
    assert lm["file"] == "data/word_sets/LM_Negative.csv" and len(lm["sha256"]) == 64
    assert all(m["measures_run_id"] == 41 and "sentiment_run_id" not in m for m in uploader.measures)


def test_measures_only_never_upserts(transcripts, monkeypatch):
    monkeypatch.setattr(ingest_module, "PostgresUploader", FakeUploader)
    summary = ingest_module.ingest(transcripts, 2016, database_url="postgresql://fake", measures_only=True)
    assert summary["num_uploaded"] == 3
    assert {how for _, how in FakeUploader.instances[-1].rows} == {"measures only"}


def test_resume_is_only_for_runs_that_read_the_database(transcripts, monkeypatch):
    monkeypatch.setattr(ingest_module, "PostgresUploader", FakeUploader)
    with pytest.raises(ValueError, match="resume needs from_db"):
        ingest_module.ingest(transcripts, 2016, database_url="postgresql://fake", resume=True)
    with pytest.raises(SystemExit):
        ingest_module.main(["--input-dir", str(transcripts), "--year", "2016", "--resume"])


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


def test_ingest_with_worker_processes(transcripts, monkeypatch):
    # spawned workers analyze, this process uploads (so the fake uploader still sees every row)
    monkeypatch.setattr(ingest_module, "PostgresUploader", FakeUploader)
    summary = ingest_module.ingest(transcripts, 2016, database_url="postgresql://fake", num_workers=2)
    assert summary["num_uploaded"] == 3 and summary["num_errors"] == 0
    assert sorted(name for name, _ in FakeUploader.instances[-1].rows) == [
        "no_headings_T.xml",
        "spaced_T.xml",
        "tight_T.xml",
    ]


def test_gpu_and_cpu_runs_record_different_settings():
    # GPU batches move scores by about 1e-6, so --resume must never mix the two
    pytest.importorskip("torch")
    cpu = ingest_module.run_settings(-1, True, True, None, None, "*.xml")
    gpu = ingest_module.run_settings(0, True, True, None, None, "*.xml")
    assert cpu["sentiment"]["batch_size"] == 32 and "gpu_batches" not in cpu["sentiment"]
    assert gpu["sentiment"]["gpu_batches"] == {"max_tokens": 4096, "pad_multiple": 32}
    assert "batch_size" not in gpu["sentiment"]
    assert ingest_module.run_settings(None, True, True, None, None, "*.xml")["sentiment"] is None


def test_device_needs_sentiment(transcripts):
    with pytest.raises(SystemExit):
        ingest_module.main(["--input-dir", str(transcripts), "--year", "2016", "--device", "0"])


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


class FailsOnWrite(FakeUploader):
    """A FakeUploader whose write of tight_T.xml fails, as a constraint violation would."""

    def upload_document(self, doc, year):
        if doc["document"]["file_name"] == "tight_T.xml":
            raise ValueError("write failed")
        return super().upload_document(doc, year)


@pytest.mark.parametrize("mode", [dict(num_threads=1), dict(num_threads=3), dict(num_workers=2)])
def test_analysis_and_write_errors_are_recorded_in_every_mode(transcripts, tmp_path, monkeypatch, mode):
    for name in ("spaced_T.xml", "tight_T.xml"):
        (tmp_path / name).write_bytes((transcripts / name).read_bytes())
    (tmp_path / "bad_T.xml").write_text("<Event><Body>unclosed", encoding="utf-8")
    monkeypatch.setattr(ingest_module, "PostgresUploader", FailsOnWrite)

    summary = ingest_module.ingest(tmp_path, 2016, database_url="postgresql://fake", **mode)
    assert summary["num_files_processed"] == 3 and summary["num_uploaded"] == 1
    reasons = {e["file"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]: e["reason"] for e in summary["errors"]}
    assert set(reasons) == {"bad_T.xml", "tight_T.xml"}
    assert reasons["tight_T.xml"] == "write failed"

    with pytest.raises(Exception):  # --stop-on-error stops on either kind
        ingest_module.ingest(tmp_path, 2016, database_url="postgresql://fake", skip_on_error=False, **mode)
