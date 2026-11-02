"""T-001: command-line entry point behaviour."""

from __future__ import annotations

import subprocess
import sys

import pytest

from retro_ai_lab import __version__
from retro_ai_lab.cli import PROG, build_parser, main


def test_version_flag_prints_version_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert out.strip() == f"{PROG} {__version__}"


def test_help_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "Retro AI Lab" in capsys.readouterr().out


def test_no_arguments_prints_help_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--bogus"], ["--version=1"], ["extra", "positional"]])
def test_unknown_arguments_exit_non_zero_with_message(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) != 0
    assert "usage:" in capsys.readouterr().err


def test_build_parser_is_fresh_each_call() -> None:
    assert build_parser() is not build_parser()


def test_module_is_runnable_as_script() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "retro_ai_lab", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"{PROG} {__version__}"


def test_module_script_propagates_error_code() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "retro_ai_lab", "--nope"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr
