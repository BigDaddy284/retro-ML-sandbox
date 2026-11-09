"""Where the app's own time goes, for the log (T-894).

The emulator link already reports how long every round trip takes. The
user's logs showed the game running at 39 frames a second while each step
cost the emulator 70 ms of a 140 ms cycle: the other half is the app's own
work on the GUI thread, which no line reported. A :class:`WorkLog` keeps
named timings (``learn``, ``ai step``, ``picture``, ``GUI stall``) the same
way the link keeps its calls, and the stall watch writes a summary line next
to the link's. Since T-947 the AI's step is timed stage by stage as well
(``step: choose``, ``step: emulator``, ``step: observe``, ``step: encode``,
``step: learn``) and so is the discovery service's share of the observe stage
(``observe: track``, ``discovery: events``, ``discovery: long look``,
``discovery: relationships``, ``discovery: announce``, ``discovery: review``,
``discovery: experiment``, ``discovery: save``), so a slow step on a user's
PC names its slow stage in the log.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class CallStats:
    """How many times one thing ran in the current window, and how long it took."""

    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += elapsed_s
        self.max_s = max(self.max_s, elapsed_s)

    @property
    def average_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0

    def describe(self, name: str) -> str:
        return (
            f"{name} \u00d7{self.count} avg {self.average_s * 1000:.0f} ms "
            f"max {self.max_s * 1000:.0f} ms"
        )


class WorkLog:
    """Named timings for the current window, slowest first in :meth:`summary`."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._stats: dict[str, CallStats] = {}
        self._window_started = clock()
        # The GUI thread, the AI's thread and the save writer (T-956) all time into
        # the app's log; a note and the window's reset must not interleave.
        self._lock = threading.Lock()

    @property
    def stats(self) -> Mapping[str, CallStats]:
        with self._lock:
            return dict(self._stats)

    @property
    def window_s(self) -> float:
        """Seconds since the window started."""
        return self._clock() - self._window_started

    def note(self, name: str, elapsed_s: float) -> None:
        if not name.strip():
            raise ValueError("A timing needs a name.")
        if elapsed_s < 0:
            raise ValueError("A timing cannot be negative.")
        with self._lock:
            self._stats.setdefault(name, CallStats()).add(elapsed_s)

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        """Time the block under ``name``, whether or not it raises."""
        started = self._clock()
        try:
            yield
        finally:
            self.note(name, self._clock() - started)

    def summary(self, *, quiet_under_s: float = 0.0) -> str:
        """One line, the name with the most total time first; "nothing" when empty.

        A name whose longest run stayed under ``quiet_under_s`` is left out
        (T-906): hundreds of free event deliveries would otherwise bury the
        one handler that cost 100 ms.
        """
        if quiet_under_s < 0:
            raise ValueError("quiet_under_s cannot be negative")
        with self._lock:
            loud = {n: s for n, s in self._stats.items() if s.max_s >= quiet_under_s}
        if not loud:
            return "nothing"
        ordered = sorted(loud.items(), key=lambda item: item[1].total_s, reverse=True)
        return "; ".join(stats.describe(name) for name, stats in ordered)

    def take_window(self, *, quiet_under_s: float = 0.0) -> tuple[float, str]:
        """The window's length and summary, then start a new window."""
        now = self._clock()
        window = now - self._window_started
        summary = self.summary(quiet_under_s=quiet_under_s)
        with self._lock:
            self._stats = {}
            self._window_started = now
        return window, summary


app_work = WorkLog()
"""The app's own work on the GUI thread; the stall watch logs it every few seconds."""


__all__ = ["CallStats", "WorkLog", "app_work"]
