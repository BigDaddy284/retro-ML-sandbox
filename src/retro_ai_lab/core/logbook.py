"""A log file for every run, so a silent start or a crash can be read back.

``setup_logging`` is called once by the entry point before Qt is imported.
It writes to ``<data dir>/logs/retro-ai-lab.log`` and keeps a few rotated
copies; it never raises, because a missing log must not stop the app.
"""

from __future__ import annotations

import logging
import platform
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

from retro_ai_lab import __version__
from retro_ai_lab.core.paths import default_data_dir

LOG_FOLDER = "logs"
LOG_FILE = "retro-ai-lab.log"
MAX_BYTES = 1_000_000
BACKUPS = 3
HANDLER_NAME = "retro-ai-lab-file"
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

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
        log.critical("Uncaught exception", exc_info=(kind, value, trace))
        previous(kind, value, trace)

    hook._retro_ai_lab_hook = True  # type: ignore[attr-defined]
    sys.excepthook = hook


__all__ = ["BACKUPS", "LOG_FILE", "LOG_FOLDER", "MAX_BYTES", "log_path", "setup_logging"]
