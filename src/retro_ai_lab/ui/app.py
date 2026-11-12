"""Application bootstrap."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from retro_ai_lab import __version__
from retro_ai_lab.core import logbook, starting
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.main_window import LAYOUT_FILE, MainWindow
from retro_ai_lab.ui.theme import THEME_FILE, apply_theme, install_watcher

APP_NAME = "Retro AI Lab"
ORGANISATION = "RetroAILab"

log = logging.getLogger(__name__)


def create_app(argv: Sequence[str] | None = None) -> QApplication:
    """Return the process-wide ``QApplication``, creating it if needed."""
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([sys.argv[0]])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORGANISATION)
    app.setApplicationVersion(__version__)
    apply_theme(app)  # T-886: dark until the window reads the saved choice
    install_watcher(app)
    return app


def run_app(
    argv: Sequence[str] | None = None,
    *,
    before_exec: Callable[[QApplication, MainWindow], None] | None = None,
    exit_after_ms: int | None = None,
    context: AppContext | None = None,
) -> int:
    """Create the app and main window, show it, and run the event loop.

    ``before_exec`` lets tests (and tooling) hook the live window right
    before the loop starts. ``exit_after_ms`` quits automatically after that
    delay, which is how the launch smoke test runs the real entry point.
    """
    log.info("Creating the Qt application")  # T-995: each stage named before it begins
    starting.step(3)  # T-1002: and shown on a built app's "starting" window
    app = create_app(argv)
    log.info("Qt application created (platform %s)", app.platformName())
    log.info("Building the main window")
    starting.step(4)
    if context is None:
        context = AppContext()
    window = MainWindow(
        context,
        layout_path=context.data_dir / LAYOUT_FILE,
        theme_path=context.data_dir / THEME_FILE,
    )
    window.show()
    log.info("Main window shown")
    starting.close()
    logbook.started()
    if before_exec is not None:
        QTimer.singleShot(0, lambda: before_exec(app, window))
    if exit_after_ms is not None:
        QTimer.singleShot(max(0, exit_after_ms), app.quit)
    code = int(app.exec())
    log.info("Event loop finished with code %d", code)
    return code


__all__ = ["APP_NAME", "ORGANISATION", "create_app", "run_app"]
