"""T-815: the log file every run leaves behind."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from retro_ai_lab import __version__
from retro_ai_lab.core import logbook
from retro_ai_lab.core.logbook import HANDLER_NAME, LOG_FILE, LOG_FOLDER, log_path, setup_logging
from retro_ai_lab.core.paths import ENV_VAR


@pytest.fixture(autouse=True)
def _fresh_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no file handler and the original excepthook."""
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    hook = sys.excepthook
    yield  # type: ignore[misc]
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)
    sys.excepthook = hook


def test_log_path_follows_the_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert log_path(tmp_path) == tmp_path / LOG_FOLDER / LOG_FILE
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "elsewhere"))
    assert log_path() == tmp_path / "elsewhere" / LOG_FOLDER / LOG_FILE


def test_setup_creates_the_file_with_a_header(tmp_path: Path) -> None:
    path = setup_logging(tmp_path)
    assert path == tmp_path / LOG_FOLDER / LOG_FILE and path.exists()
    text = path.read_text(encoding="utf-8")
    assert f"Retro AI Lab {__version__} starting" in text
    assert "source checkout" in text and str(tmp_path) in text
    logging.getLogger("retro_ai_lab.something").info("hello from a module")
    assert "hello from a module" in path.read_text(encoding="utf-8")


def test_setup_twice_keeps_one_handler(tmp_path: Path) -> None:
    first = setup_logging(tmp_path)
    second = setup_logging(tmp_path / "other")
    assert first == second
    named = [h for h in logging.getLogger().handlers if h.get_name() == HANDLER_NAME]
    assert len(named) == 1


def test_an_unwritable_place_logs_nowhere_and_raises_nothing(tmp_path: Path) -> None:
    (tmp_path / LOG_FOLDER).write_text("a file where the folder should be")
    assert setup_logging(tmp_path) is None
    assert not any(h.get_name() == HANDLER_NAME for h in logging.getLogger().handlers)


def test_uncaught_exceptions_reach_the_file_and_the_previous_hook(tmp_path: Path) -> None:
    seen: list[type[BaseException]] = []
    sys.excepthook = lambda kind, value, trace: seen.append(kind)
    path = setup_logging(tmp_path)
    assert path is not None
    try:
        raise RuntimeError("the window never came")
    except RuntimeError as exc:
        sys.excepthook(type(exc), exc, exc.__traceback__)
    text = path.read_text(encoding="utf-8")
    assert "Uncaught exception" in text and "the window never came" in text
    assert "Traceback" in text and seen == [RuntimeError]
    # Setting up again does not wrap the hook a second time.
    hook = sys.excepthook
    logbook._install_excepthook()
    assert sys.excepthook is hook
