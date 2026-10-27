import pytest

from qudock.cli import main


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert all(cmd in out for cmd in ("data", "benchmark", "sweep", "noise", "final",
                                       "hardware-svm"))


def test_command_is_required():
    with pytest.raises(SystemExit):
        main([])
