"""T-001: command-line entry point behaviour."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

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


def test_no_arguments_launches_the_app_and_returns_its_exit_code() -> None:
    launched: list[Sequence[str]] = []

    def fake_launch(argv: Sequence[str]) -> int:
        launched.append(tuple(argv))
        return 7

    assert main([], launch=fake_launch) == 7
    assert main(["--exit-after-ms", "5"], launch=fake_launch) == 7
    assert launched == [(), ("--exit-after-ms", "5")]


def test_no_argv_uses_process_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["retro-ai-lab", "--exit-after-ms", "1"])
    launched: list[Sequence[str]] = []
    assert main(launch=lambda argv: launched.append(argv) or 3) == 3
    assert launched == [("--exit-after-ms", "1")]


def test_exit_after_ms_requires_an_integer(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--exit-after-ms", "soon"]) == 2
    assert "invalid int value" in capsys.readouterr().err


def test_version_and_help_do_not_launch_the_app() -> None:
    def never(argv: Sequence[str]) -> int:
        raise AssertionError("must not launch")

    assert main(["--version"], launch=never) == 0
    assert main(["--help"], launch=never) == 0
    assert main(["--bogus"], launch=never) == 2


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
