from research.db import provenance


def test_code_hash_is_stable_and_changes_with_the_code(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n")
    first = provenance.code_sha256(tmp_path)
    assert first == provenance.code_sha256(tmp_path) and len(first) == 64
    (tmp_path / "sub" / "b.py").write_text("y = 3\n")
    assert provenance.code_sha256(tmp_path) != first


def test_word_list_is_recorded_relative_to_the_repo_with_its_hash():
    from research import config

    entry = provenance.word_list(config.WORD_SETS_DIR / "Risk_v2.csv")
    assert entry["file"] == "data/word_sets/Risk_v2.csv"
    assert entry["sha256"] == provenance.file_sha256(config.WORD_SETS_DIR / "Risk_v2.csv")
    assert provenance.word_list(None) is None


def test_versions_name_python_and_the_packages_that_change_numbers():
    versions = provenance.package_versions()
    assert versions["python"] and versions["nltk"] and versions["textstat"]
    assert "transformers" in versions  # None when the ml extra is not installed


def test_model_revision_needs_no_network():
    assert provenance.model_revision("no-such-org/no-such-model") is None
