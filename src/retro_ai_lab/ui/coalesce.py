"""Run a refresh at most every so many milliseconds (T-843).

A panel that rebuilds a 1,600-row list on every update is fine for a person
who changes one thing at a time and far too much for a training run that
publishes an update on most steps. :class:`Coalescer` runs the first request
at once (so a single change shows immediately, and tests see it without
waiting) and folds requests that follow within ``delay_ms`` into one run at
the end of that window. A coalescer with a ``name`` times each run in the
app's work log under it (T-906), so a slow refresh is named on the stall line.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, QTimer

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.worklog import WorkLog, app_work

DEFAULT_DELAY_MS = 250


class Coalescer(QObject):
    def __init__(
        self,
        action: Callable[[], None],
        *,
        delay_ms: int = DEFAULT_DELAY_MS,
        name: str | None = None,
        worklog: WorkLog = app_work,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if delay_ms < 1:
            raise ConfigurationError("delay_ms must be at least 1.")
        if name is not None and not name.strip():
            raise ConfigurationError("A coalescer's name cannot be blank.")
        self._action = action
        self._name = name
        self._worklog = worklog
        self._delay_s = delay_ms / 1000
        self._last: float | None = None
        self._runs = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self._fire)

    @property
    def pending(self) -> bool:
        """A request is waiting for the end of the window."""
        return self._timer.isActive()

    @property
    def runs(self) -> int:
        """How many times the action has run."""
        return self._runs

    @property
    def delay_ms(self) -> int:
        return self._timer.interval()

    @property
    def name(self) -> str | None:
        """What the work log calls a run, or None when runs are not timed."""
        return self._name

    def request(self) -> None:
        """Run now if the window is clear, else once when it closes."""
        if self.pending:
            return
        now = time.monotonic()
        if self._last is None or now - self._last >= self._delay_s:
            self._run(now)
            return
        remaining = self._delay_s - (now - self._last)
        self._timer.start(max(1, int(remaining * 1000)))

    def flush(self) -> None:
        """Run a waiting request now."""
        if self.pending:
            self._timer.stop()
            self._fire()

    def cancel(self) -> None:
        self._timer.stop()

    def _fire(self) -> None:
        self._run(time.monotonic())

    def _run(self, now: float) -> None:
        self._last = now
        self._runs += 1
        if self._name is None:
            self._action()
            return
        with self._worklog.timed(self._name):
            self._action()


__all__ = ["DEFAULT_DELAY_MS", "Coalescer"]
