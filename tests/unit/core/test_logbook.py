"""T-815: the log file every run leaves behind."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from retro_ai_lab import __version__
from retro_ai_lab.core import logbook
from retro_ai_lab.core.logbook import (
    HANDLER_NAME,
    LOG_FILE,
    LOG_FOLDER,
    TRACE_PREFIX,
    log_path,
    setup_logging,
    watch_start,
)
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


# ---------------------------------------------------------------- T-995
# faulthandler belongs to the whole process (and pytest uses it), so every watch
# is run in a child Python.

PRELUDE = """
import sys, time
from pathlib import Path
from retro_ai_lab.core import logbook
folder = Path(sys.argv[1])
logbook.setup_logging(folder)
"""


def _child(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONFAULTHANDLER", None)
    return subprocess.run(
        [sys.executable, "-c", PRELUDE + body, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )


def _traces(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / LOG_FOLDER).glob(f"{TRACE_PREFIX}*.txt"))


def test_a_start_that_gets_stuck_says_where_and_the_next_start_logs_it(tmp_path: Path) -> None:
    stuck = _child(
        tmp_path,
        """
def stuck_here():
    time.sleep(1.5)
path = logbook.watch_start(folder, stuck_after=0.3)
print(path)
stuck_here()
import os
os._exit(0)  # killed, as a person would: nothing tidies up
""",
    )
    assert stuck.returncode == 0, stuck.stderr
    (trace,) = _traces(tmp_path)
    assert str(trace) == stuck.stdout.strip()
    text = trace.read_text(encoding="utf-8")
    assert "stuck_here" in text and "Timeout" in text
    log_file = tmp_path / LOG_FOLDER / LOG_FILE
    assert "stuck_here" not in log_file.read_text(encoding="utf-8")
    # The next start copies it into the log a person sends, and clears it away.
    after = _child(tmp_path, "logbook.watch_start(folder)\nlogbook.started()\n")
    assert after.returncode == 0, after.stderr
    logged = log_file.read_text(encoding="utf-8")
    assert "An earlier start left a trace" in logged and "this is where it was" in logged
    assert trace.name in logged and "stuck_here" in logged
    assert _traces(tmp_path) == []
    # And only once: a third start has nothing to say about it.
    again = _child(tmp_path, "logbook.watch_start(folder)\nlogbook.started()\n")
    assert again.returncode == 0, again.stderr
    assert log_file.read_text(encoding="utf-8").count("An earlier start left a trace") == 1


def test_a_start_in_time_leaves_nothing(tmp_path: Path) -> None:
    fine = _child(
        tmp_path,
        """
first = logbook.watch_start(folder, stuck_after=0.3)
assert first is not None and first.exists()
assert logbook.watch_start(folder, stuck_after=0.3) == first  # one watch
logbook.started()
time.sleep(0.8)  # well past the limit: the window was up in time, nothing is written
assert first.read_text(encoding="utf-8") == ""
""",
    )
    assert fine.returncode == 0, fine.stderr
    assert _traces(tmp_path) == []  # removed when the run ended with nothing in it
    text = (tmp_path / LOG_FOLDER / LOG_FILE).read_text(encoding="utf-8")
    assert "left a trace" not in text


def test_a_crash_in_a_library_says_where(tmp_path: Path) -> None:
    crashed = _child(
        tmp_path,
        """
import faulthandler
def crash_here():
    faulthandler._sigsegv()
logbook.watch_start(folder)
logbook.started()
crash_here()
""",
    )
    assert crashed.returncode != 0
    (trace,) = _traces(tmp_path)
    assert "crash_here" in trace.read_text(encoding="utf-8")
    after = _child(tmp_path, "logbook.watch_start(folder)\nlogbook.started()\n")
    assert after.returncode == 0, after.stderr
    assert "crash_here" in (tmp_path / LOG_FOLDER / LOG_FILE).read_text(encoding="utf-8")
    assert _traces(tmp_path) == []


def test_an_unwritable_place_watches_nothing(tmp_path: Path) -> None:
    (tmp_path / LOG_FOLDER).write_text("a file where the folder should be")
    assert watch_start(tmp_path) is None  # and nothing was raised or switched on
    logbook.started()  # harmless with no watch
