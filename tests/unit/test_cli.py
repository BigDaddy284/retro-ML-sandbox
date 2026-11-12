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


# ---------------------------------------------------------------- T-995


@pytest.fixture
def _launch_log(tmp_path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """The real launcher with its log in a temp folder and no process-wide watch."""
    import logging

    from retro_ai_lab.core import logbook
    from retro_ai_lab.core.paths import ENV_VAR

    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    monkeypatch.setattr(logbook, "watch_start", lambda *a, **k: None)
    root = logging.getLogger()
    before, level, hook = list(root.handlers), root.level, sys.excepthook
    yield logbook.log_path(tmp_path)
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)
    sys.excepthook = hook


def test_the_launch_logs_its_stages(_launch_log, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    import retro_ai_lab.ui.app as ui_app
    from retro_ai_lab.cli import _default_launch

    monkeypatch.setattr(ui_app, "run_app", lambda argv, exit_after_ms=None: 0)
    assert _default_launch(()) == 0
    text = _launch_log.read_text(encoding="utf-8")
    stages = ["starting", "Loading Qt", "Loading the app's own code", "Starting the app"]
    places = [text.index(stage) for stage in stages]
    assert places == sorted(places)


def test_a_launch_that_fails_is_logged_once_and_raised(  # type: ignore[no-untyped-def]
    _launch_log, monkeypatch: pytest.MonkeyPatch
) -> None:
    import retro_ai_lab.ui.app as ui_app
    from retro_ai_lab.cli import _default_launch

    def broken(argv: Sequence[str], exit_after_ms: int | None = None) -> int:
        raise RuntimeError("the window never came")

    monkeypatch.setattr(ui_app, "run_app", broken)
    # The windowed build's own error box takes the place of the hook: stand in for it.
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    with pytest.raises(RuntimeError, match="the window never came") as caught:
        _default_launch(())
    text = _launch_log.read_text(encoding="utf-8")
    assert "The app could not start" in text and "the window never came" in text
    assert "Traceback" in text
    # Where the hook does run (from source), it does not write the same thing again.
    sys.excepthook(RuntimeError, caught.value, caught.value.__traceback__)
    assert "Uncaught exception" not in _launch_log.read_text(encoding="utf-8")


class _Splash:
    """A stand-in for the build tool's ``pyi_splash`` module (T-1002)."""

    def __init__(self) -> None:
        self.did: list[str] = []
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def update_text(self, text: str) -> None:
        self.did.append(text)

    def close(self) -> None:
        self.did.append("closed")
        self.alive = False


@pytest.fixture
def _splash(monkeypatch: pytest.MonkeyPatch) -> _Splash:
    from retro_ai_lab.core import starting

    splash = _Splash()
    monkeypatch.setattr(starting, "_window", starting._UNKNOWN)
    monkeypatch.setitem(sys.modules, "pyi_splash", splash)
    return splash


def test_the_launch_says_its_steps_on_the_starting_window(  # type: ignore[no-untyped-def]
    _launch_log, _splash: _Splash, monkeypatch: pytest.MonkeyPatch
) -> None:
    import retro_ai_lab.ui.app as ui_app
    from retro_ai_lab.cli import _default_launch

    seen: list[list[str]] = []

    def opened(argv: Sequence[str], exit_after_ms: int | None = None) -> int:
        seen.append(list(_splash.did))
        return 0

    monkeypatch.setattr(ui_app, "run_app", opened)
    assert _default_launch(()) == 0
    # Both loading steps were named before the app itself was started; the rest is the app's.
    assert seen == [["Step 1 of 4: loading the window toolkit", "Step 2 of 4: loading the app"]]
    text = _launch_log.read_text(encoding="utf-8")
    places = [text.index(line) for line in ("Loading Qt", "The starting window is showing")]
    assert places == sorted(places)


def test_a_launch_that_fails_closes_the_starting_window_first(  # type: ignore[no-untyped-def]
    _launch_log, _splash: _Splash, monkeypatch: pytest.MonkeyPatch
) -> None:
    import retro_ai_lab.ui.app as ui_app
    from retro_ai_lab.cli import _default_launch

    def broken(argv: Sequence[str], exit_after_ms: int | None = None) -> int:
        raise RuntimeError("the window never came")

    monkeypatch.setattr(ui_app, "run_app", broken)
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    with pytest.raises(RuntimeError, match="the window never came"):
        _default_launch(())
    assert _splash.did[-1] == "closed" and _splash.did.count("closed") == 1
    text = _launch_log.read_text(encoding="utf-8")
    places = [
        text.index(line) for line in ("The app could not start", "The starting window is closed")
    ]
    assert places == sorted(places)
