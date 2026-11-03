"""What a worker thread does when it has nothing left to report to (T-829).

The pump, the clock and the pad each run a request on a worker and report
back by signal. When the window closes while a request is still waiting on
the emulator (a step that times out takes five seconds), the signal's
QObject can be gone by the time the worker returns; PySide then raises
``RuntimeError("Signal source has been deleted")`` from ``emit``, which the
log recorded as an uncaught exception. There is nobody left to tell, so
the report is dropped quietly instead.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import SignalInstance

log = logging.getLogger(__name__)


def report(signal: SignalInstance, *args: Any) -> bool:
    """Emit ``signal`` with ``args``; False when its owner is already gone."""
    try:
        signal.emit(*args)
    except RuntimeError as exc:
        log.debug("A worker had nobody left to report to: %s", exc)
        return False
    return True


__all__ = ["report"]
