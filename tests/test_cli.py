import argparse

import pytest

from qudock.cli import main, parse_ints, parse_range, parse_sizes


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert all(cmd in out for cmd in ("data", "benchmark", "sweep", "noise", "final",
                                      "scaling", "trainable", "confidence", "scorecard",
                                      "explain", "robustness", "campaign", "cost",
                                      "hardware-svm"))


def test_command_is_required():
    with pytest.raises(SystemExit):
        main([])


def test_reproduce_runs_every_study_that_makes_a_figure(monkeypatch, tmp_path):
    from qudock import cli

    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv[0]))
    (tmp_path / "saved.npz").touch()
    monkeypatch.setitem(cli.SAVED_DEVICE_DATA, "hardware-svm", tmp_path / "saved.npz")
    cli.cmd_reproduce(argparse.Namespace(skip="noise,hardware", verbose=False))
    assert calls == ["benchmark", "sweep", "final", "final", "scaling", "trainable",
                     "confidence", "explain", "robustness", "campaign", "cost",
                     "scorecard", "hardware-svm"]


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


def test_version_flag(capsys):
    from qudock import __version__

    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_data_command_reports_the_chosen_dataset(monkeypatch, capsys):
    import pandas as pd

    from qudock import cli

    toy = pd.DataFrame({"smiles": ["c1ccccc1O", "c1ccccc1N", "C1CCCCC1O", "CCO", "CCN"] * 2,
                        "label": [0, 1] * 5})
    monkeypatch.setitem(cli.DATASETS, "bbbp", lambda: toy)
    main(["data", "--dataset", "bbbp"])
    out = capsys.readouterr().out
    assert "bbbp molecules: 10" in out and "scaffold" in out


def test_reproduce_skips_reanalysis_without_saved_device_data(monkeypatch, tmp_path, capsys):
    from qudock import cli

    calls = []
    monkeypatch.setattr(cli, "main", lambda argv: calls.append(argv[0]))
    monkeypatch.setitem(cli.SAVED_DEVICE_DATA, "hardware-svm", tmp_path / "missing.npz")
    skip = ("benchmark,sweep,noise,final,scaling,trainable,confidence,explain,"
            "robustness,campaign,cost,scorecard,hardware")
    cli.cmd_reproduce(argparse.Namespace(skip=skip, verbose=False))
    assert calls == []
    assert "skipping qudock hardware-svm" in capsys.readouterr().out


def test_reanalyse_without_saved_data_explains_itself(tmp_path):
    with pytest.raises(SystemExit, match="nothing to reanalyse"):
        main(["hardware-svm", "--reanalyse", "--out", str(tmp_path)])
