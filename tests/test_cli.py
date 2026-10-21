import pytest

from qudock.cli import main


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "data" in out and "benchmark" in out


def test_command_is_required():
    with pytest.raises(SystemExit):
        main([])
