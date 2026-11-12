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
    """Stands in for SentimentSetup: the sentences of the longest paragraph score 1, all
    others 0, and fit_all's own document score is the mean of the paragraph means."""

    def fit_all(self, doc):
        longest = max(doc.paragraphs, key=lambda p: len(p.sentences))
        for paragraph in doc.paragraphs:
            for sentence in paragraph.sentences:
                sentence.sentiment = 1.0 if paragraph is longest else 0.0
            paragraph.sentiment = 1.0 if paragraph is longest else 0.0
        doc.sentiment = sum(p.sentiment for p in doc.paragraphs) / len(doc.paragraphs)
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
    for key in ("sentiment", "LM", "ML", "flesch_reading_ease", "gunning_fog", "difficult_words"):
        assert attrs[key] is None, key


def test_sentiment_is_the_mean_over_every_sentence(transcripts):
    out = analyze_document(transcripts / "spaced_T.xml", sentiment=ScoresEverything())
    paragraphs = out["doc_fit"].paragraphs
    longest = max(len(p.sentences) for p in paragraphs)
    total = sum(len(p.sentences) for p in paragraphs)
    assert longest > 1
    assert out["document_attr"]["sentiment"] == pytest.approx(longest / total)
    # not the mean of the paragraph means, which gives each paragraph the same weight
    assert out["doc_fit"].sentiment == pytest.approx(1 / len(paragraphs))


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

    def upload_document(self, data, year):
        self.rows.append((data["doc_attrs"]["document"]["file_name"], year))
        self.measures.append(data["doc_attrs"]["document_attr"])
        return {"status": "success", "year": 2017, "year_fallback": False}

    def update_measures(self, data):
        self.rows.append((data["doc_attrs"]["document"]["file_name"], "measures only"))
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
