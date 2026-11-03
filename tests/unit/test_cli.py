import pytest

from research import cli


def test_help_lists_every_step(capsys):
    cli.main(["--help"])
    out = capsys.readouterr().out
    for step in ("ingest", "backfill", "link route-a", "returns compute", "export", "demo"):
        assert step in out


def test_step_help_reaches_the_module(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["link", "route-b", "--help"])
    assert exit_info.value.code == 0
    assert "--stocknames" in capsys.readouterr().out


def test_unknown_and_incomplete_steps_fail():
    with pytest.raises(SystemExit):
        cli.main(["nope"])
    with pytest.raises(SystemExit, match="route-a"):
        cli.main(["link"])
