"""A log file for every run, so a silent start or a crash can be read back.

``setup_logging`` is called once by the entry point before Qt is imported.
It writes to ``<data dir>/logs/retro-ai-lab.log`` and keeps a few rotated
copies; it never raises, because a missing log must not stop the app.

A start that hangs, or dies inside a library, runs no Python that could log
it (T-995). ``watch_start`` hands that to :mod:`faulthandler`, which needs no
Python to be running: if :func:`started` has not been called after
``STUCK_AFTER`` seconds it writes every thread's stack to this run's trace file
(``logs/start-<pid>.txt``), and so does a crash at any time. The next start
copies what earlier runs left there into the log and removes the files, so the
one log a person sends says where the start before it stopped.
"""

from __future__ import annotations

import atexit
import contextlib
import faulthandler
import logging
import os
import platform
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import TextIO

from retro_ai_lab import __version__
from retro_ai_lab.core.paths import default_data_dir

LOG_FOLDER = "logs"
LOG_FILE = "retro-ai-lab.log"
MAX_BYTES = 1_000_000
BACKUPS = 3
HANDLER_NAME = "retro-ai-lab-file"
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
TRACE_PREFIX = "start-"  # logs/start-<pid>.txt: where a start got stuck or crashed (T-995)
STUCK_AFTER = 10.0  # seconds; a start takes about one
TRACE_MOST = 20_000  # characters of an earlier trace copied into the log
LOGGED_MARK = "_retro_ai_lab_logged"  # on an exception already written to the log

log = logging.getLogger("retro_ai_lab")


def log_path(data_dir: Path | None = None) -> Path:
    """Where the log file goes for ``data_dir`` (the default data dir if None)."""
    return (data_dir if data_dir is not None else default_data_dir()) / LOG_FOLDER / LOG_FILE


def _file_handler() -> RotatingFileHandler | None:
    root = logging.getLogger()
    for handler in root.handlers:
        if handler.get_name() == HANDLER_NAME and isinstance(handler, RotatingFileHandler):
            return handler
    return None


def setup_logging(data_dir: Path | None = None, *, level: int = logging.INFO) -> Path | None:
    """Start logging to the file under ``data_dir``. Returns its path, or None if it can't.

    Calling it again returns the path already in use. Uncaught exceptions are
    written to the file too (the previous ``sys.excepthook`` still runs).
    """
    existing = _file_handler()
    if existing is not None:
        return Path(existing.baseFilename)
    path = log_path(data_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8"
        )
    except OSError:
        return None
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(logging.Formatter(FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    _install_excepthook()
    log.info(
        "Retro AI Lab %s starting: Python %s on %s (%s), data dir %s, argv %s",
        __version__,
        platform.python_version(),
        platform.platform(),
        "built app" if getattr(sys, "frozen", False) else "source checkout",
        path.parent.parent,
        sys.argv[1:],
    )
    return path


def _install_excepthook() -> None:
    previous = sys.excepthook
    if getattr(previous, "_retro_ai_lab_hook", False):
        return

    def hook(kind: type[BaseException], value: BaseException, trace: TracebackType | None) -> None:
        if not getattr(value, LOGGED_MARK, False):
            log.critical("Uncaught exception", exc_info=(kind, value, trace))
        previous(kind, value, trace)

    hook._retro_ai_lab_hook = True  # type: ignore[attr-defined]
    sys.excepthook = hook


def log_failure(error: BaseException) -> None:
    """Write an exception that is about to end the launch to the log, once (T-995).

    The windowed build shows its own error box for such an exception and never
    calls ``sys.excepthook``, so the hook above does not see it there; from
    source the hook does run, and leaves an exception marked here alone.
    """
    log.critical("The app could not start", exc_info=(type(error), error, error.__traceback__))
    with contextlib.suppress(Exception):  # an exception that takes no attributes
        setattr(error, LOGGED_MARK, True)


_watch: tuple[Path, TextIO] | None = None  # this run's trace file, open for faulthandler


def _carry_over(folder: Path, mine: Path) -> None:
    """Copy what earlier runs left in their trace files into the log, and remove them."""
    try:
        earlier = sorted(p for p in folder.glob(f"{TRACE_PREFIX}*.txt") if p != mine)
    except OSError:
        return
    for path in earlier:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if text:
            log.warning(
                "An earlier start left a trace (it took over %d seconds to open, got stuck "
                "or crashed); this is where it was (%s):\n%s",
                STUCK_AFTER,
                path.name,
                text[:TRACE_MOST],
            )
        with contextlib.suppress(OSError):  # still open in a run that is still going
            path.unlink()


def watch_start(data_dir: Path | None = None, *, stuck_after: float = STUCK_AFTER) -> Path | None:
    """Watch this start: returns the trace file, or None where it cannot be written.

    Logs the traces earlier runs left, then has ``faulthandler`` write every
    thread's stack to this run's file if :func:`started` is not called within
    ``stuck_after`` seconds, and at a crash inside a library for as long as the
    app runs. Never raises; calling it again returns the file already in use.
    """
    global _watch
    if _watch is not None:
        return _watch[0]
    folder = log_path(data_dir).parent
    path = folder / f"{TRACE_PREFIX}{os.getpid()}.txt"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _carry_over(folder, path)
        handle = open(path, "w", encoding="utf-8")  # noqa: SIM115 (kept open for the run)
    except OSError:
        return None
    try:
        faulthandler.enable(handle, all_threads=True)
        faulthandler.dump_traceback_later(stuck_after, file=handle)
    except (OSError, ValueError, RuntimeError):
        handle.close()
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    _watch = (path, handle)
    atexit.register(_end_watch)
    return path


def started() -> None:
    """The window is up: the start is no longer timed. Crashes are still written."""
    if _watch is not None:
        faulthandler.cancel_dump_traceback_later()


def _end_watch() -> None:
    """The run is ending in good order: stop watching, and leave no empty file."""
    global _watch
    if _watch is None:
        return
    path, handle = _watch
    _watch = None
    with contextlib.suppress(Exception):
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
    with contextlib.suppress(OSError, ValueError):
        handle.close()
    with contextlib.suppress(OSError):
        if path.stat().st_size == 0:
            path.unlink()


__all__ = [
    "BACKUPS",
    "LOG_FILE",
    "LOG_FOLDER",
    "MAX_BYTES",
    "STUCK_AFTER",
    "TRACE_PREFIX",
    "log_failure",
    "log_path",
    "setup_logging",
    "started",
    "watch_start",
]
