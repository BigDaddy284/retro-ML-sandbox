"""Command-line entry point.

For now the CLI only reports its version. The desktop application is wired in
by a later task (T-007), which will make a bare ``retro-ai-lab`` launch the UI.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from retro_ai_lab import __version__

PROG = "retro-ai-lab"


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code.

    ``argparse`` raises ``SystemExit`` for ``--version``, ``--help`` and bad
    arguments; we convert those into a return code so callers (and tests) get a
    plain integer instead of an exception.
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
    # No sub-command yet: behave like --help so the user sees what exists.
    parser.print_help()
    return 0
