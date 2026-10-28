import argparse

import pytest

from qudock.cli import main, parse_ints, parse_range, parse_sizes


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert all(cmd in out for cmd in ("data", "benchmark", "sweep", "noise", "final",
                                      "scaling", "trainable", "confidence", "scorecard",
                                      "explain", "hardware-svm"))


def test_command_is_required():
    with pytest.raises(SystemExit):
        main([])


def test_reproduce_runs_every_study_that_makes_a_figure(monkeypatch):
    from qudock import cli

    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv[0]))
    cli.cmd_reproduce(argparse.Namespace(skip="noise,hardware", verbose=False))
    assert calls == ["benchmark", "sweep", "final", "final", "scaling", "trainable",
                     "confidence", "explain", "scorecard", "hardware-svm"]


def test_list_options_are_parsed_and_checked():
    assert parse_sizes("25, 50,full") == [25, 50, None]
    assert parse_ints("256,1024") == [256, 1024]
    assert parse_range("2-4") == range(2, 5)
    for parse, bad in [(parse_sizes, "25,lots"), (parse_sizes, "0"), (parse_ints, "4,x"),
                       (parse_range, "10-2"), (parse_range, "2to10")]:
        with pytest.raises(argparse.ArgumentTypeError):
            parse(bad)


def test_bad_option_value_exits_with_usage(capsys):
    with pytest.raises(SystemExit):
        main(["benchmark", "--sizes", "25,lots"])
    assert "bad training size" in capsys.readouterr().err
