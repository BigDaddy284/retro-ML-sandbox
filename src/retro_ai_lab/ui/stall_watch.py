"""Notice when the GUI thread stalls, and log where the app's time went (T-894).

A timer on the GUI thread fires every few milliseconds; a gap much longer
than its interval means the thread was busy (a long step, a slow panel), which
the person sees as a freeze. Every ``report_every_s`` the watch writes one
line next to the emulator link's: the app's own timings (:data:`app_work`)
and how many stalls over ``stall_s`` there were, with the longest. Since
T-906 the timings name every event the window's thread handled (``on
DiscoveryUpdated``), the panels' coalesced refreshes and the picture's
paint, and the line leaves out work that never took ``quiet_under_s``, so
the stall's cause is on the line and the free deliveries are not.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, QTimer

from retro_ai_lab.core.worklog import WorkLog, app_work

log = logging.getLogger(__name__)

DEFAULT_INTERVAL_MS = 10
DEFAULT_STALL_S = 0.05  # a stall this long shows as a stutter
REPORT_EVERY_S = 5.0
DEFAULT_QUIET_S = 0.001  # work that never took a millisecond is not worth a word
STALL = "GUI stall"


class StallWatch(QObject):
    def __init__(
        self,
        worklog: WorkLog = app_work,
        *,
        interval_ms: int = DEFAULT_INTERVAL_MS,
        stall_s: float = DEFAULT_STALL_S,
        report_every_s: float = REPORT_EVERY_S,
        quiet_under_s: float = DEFAULT_QUIET_S,
        clock: Callable[[], float] = time.perf_counter,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if stall_s <= 0 or report_every_s <= 0:
            raise ValueError("stall_s and report_every_s must be positive")
        if quiet_under_s < 0:
            raise ValueError("quiet_under_s cannot be negative")
        self._worklog = worklog
        self._stall_s = stall_s
        self._report_every_s = report_every_s
        self._quiet_under_s = quiet_under_s
        self._clock = clock
        self._last: float | None = None
        self._stalls = 0
        self._reports = 0
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.beat)
        timer = self._timer
        self.destroyed.connect(lambda *_: timer.stop())

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    @property
    def stalls(self) -> int:
        """Gaps over ``stall_s`` seen since the watch started."""
        return self._stalls

    @property
    def reports(self) -> int:
        """Summary lines written."""
        return self._reports

    def start(self) -> None:
        self._last = None
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def beat(self) -> None:
        now = self._clock()
        if self._last is not None:
            gap = now - self._last - self._timer.interval() / 1000
            if gap > self._stall_s:
                self._stalls += 1
                self._worklog.note(STALL, gap)
        self._last = now
        if self._worklog.window_s >= self._report_every_s:
            self.report()

    def report(self) -> None:
        """Log the window's summary (nothing when the app did nothing worth a word)."""
        window, summary = self._worklog.take_window(quiet_under_s=self._quiet_under_s)
        if summary == "nothing":
            return
        self._reports += 1
        log.info("App work, last %.0f s: %s", window, summary)


__all__ = [
    "DEFAULT_INTERVAL_MS",
    "DEFAULT_QUIET_S",
    "DEFAULT_STALL_S",
    "REPORT_EVERY_S",
    "STALL",
    "StallWatch",
]
