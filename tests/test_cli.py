import argparse

import pytest

from qudock.cli import main


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert all(cmd in out for cmd in ("data", "benchmark", "sweep", "noise", "final",
                                      "scaling", "trainable", "confidence", "scorecard",
                                      "hardware-svm"))


def test_command_is_required():
    with pytest.raises(SystemExit):
        main([])


def test_reproduce_runs_every_study_that_makes_a_figure(monkeypatch):
    from qudock import cli

    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv[0]))
    cli.cmd_reproduce(argparse.Namespace(skip="noise,hardware", verbose=False))
    assert calls == ["benchmark", "sweep", "final", "final", "scaling", "trainable",
                     "confidence", "scorecard", "hardware-svm"]
