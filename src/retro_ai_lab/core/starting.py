"""The "starting" window a built app shows while it loads (T-1002).

The first start of a freshly unzipped build can take most of a minute before
any of the app's own code runs: Windows reads every new file once. For all of
that time there was nothing on the screen, and a start that was only slow
looked dead. The built app therefore carries the build tool's splash screen: a
small window its bootloader draws before Python is loaded
(``packaging/retro_ai_lab.spec``). This module is the app's side of it: the
start says which step it is on, and closes the window once the main window is
shown.

The build tool's ``pyi_splash`` module exists only inside a built app, and
talks to the window over a local socket. Run from source, or built where the
window cannot be made, nothing here does anything; and whatever goes wrong
with the window, nothing here raises: it is a courtesy, never a reason for a
start to fail.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: What the start does before its window is up, in the order it does it.
STEPS = (
    "loading the window toolkit",
    "loading the app",
    "starting the window toolkit",
    "building the window",
)

_UNKNOWN: Any = object()
_window: Any = _UNKNOWN  # ``pyi_splash`` once looked for, None where there is none


def _find() -> Any:
    """The build tool's splash module, or None; looked for once."""
    global _window
    if _window is _UNKNOWN:
        try:
            import pyi_splash
        except Exception:  # not a built app, or the window could not be reached
            _window = None
        else:
            _window = pyi_splash
            if showing():
                log.info("The starting window is showing")
    return _window


def showing() -> bool:
    """Whether the starting window is up."""
    window = _find()
    if window is None:
        return False
    try:
        return bool(window.is_alive())
    except Exception:
        return False


def say(text: str) -> None:
    """Show ``text`` as the step the start is on. Plain words: no brackets."""
    if not showing():
        return
    try:
        _window.update_text(text)
    except Exception:
        log.debug("The starting window did not take %r", text, exc_info=True)


def step(number: int) -> None:
    """Say that the start is on step ``number`` (from 1) of :data:`STEPS`."""
    if 1 <= number <= len(STEPS):
        say(f"Step {number} of {len(STEPS)}: {STEPS[number - 1]}")


def close() -> None:
    """Take the starting window down. Safe to call again, and when there is none."""
    if not showing():
        return
    try:
        _window.close()
    except Exception:
        log.warning("The starting window could not be closed", exc_info=True)
        return
    log.info("The starting window is closed")


__all__ = ["STEPS", "close", "say", "showing", "step"]
