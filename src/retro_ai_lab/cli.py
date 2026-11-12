"""Command-line entry point.

Running ``retro-ai-lab`` (or ``python -m retro_ai_lab``) with no arguments
opens the desktop application. ``--version`` and ``--help`` are the only
other options for now.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from retro_ai_lab import __version__

PROG = "retro-ai-lab"

Launcher = Callable[[Sequence[str]], int]


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Retro AI Lab: train an AI to play retro games by setting goals and rewards.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROG} {__version__}",
    )
    # Hidden: used by the launch smoke test and by tooling to open the app
    # and close it again after a delay.
    parser.add_argument("--exit-after-ms", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def _default_launch(argv: Sequence[str]) -> int:
    # The log file first, so a start that never shows a window still leaves a trace;
    # Qt is imported lazily so ``--version`` works without loading it.
    # Each stage is logged before it begins (T-995): the last line of a start that
    # stops is then the stage it stopped in, and the watch writes down where.
    from retro_ai_lab.core import logbook

    logbook.setup_logging()
    logbook.watch_start()
    log = logging.getLogger("retro_ai_lab")
    try:
        log.info("Loading Qt")
        import PySide6.QtWidgets  # noqa: F401

        log.info("Loading the app's own code")
        import retro_ai_lab.ui.app as ui_app

        log.info("Starting the app")
        args = build_parser().parse_args(argv)
        return ui_app.run_app(argv, exit_after_ms=args.exit_after_ms)
    except Exception as error:
        logbook.log_failure(error)
        raise


def main(argv: Sequence[str] | None = None, *, launch: Launcher = _default_launch) -> int:
    """Run the CLI. Returns the process exit code.

    ``argparse`` raises ``SystemExit`` for ``--version``, ``--help`` and bad
    arguments; we convert those into a return code so callers (and tests) get a
    plain integer instead of an exception. ``launch`` is the function that
    starts the desktop app; tests inject a stand-in.
    """
    parser = build_parser()
    try:
        parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    return launch(tuple(argv) if argv is not None else tuple(sys.argv[1:]))
