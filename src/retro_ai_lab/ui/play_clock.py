"""Advance the emulator one frame per tick while playing.

Time in Retro AI Lab moves when the app says so: every adapter that supports
``frame_advance`` is stepped by this clock (the fake console only moves this
way; real emulators are driven the same way through their scripts). The
child can pause and resume; training modes drive the same clock.

The timer's steps run on a worker thread (T-818): a real emulator takes at
least one frame to answer and the memory snapshot after it is another round
trip, and the GUI thread must never wait for either. ``tick(n)`` stays
synchronous for tests and tooling that drive time by hand.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from retro_ai_lab.core import EmulatorDisconnectedError, EmulatorError
from retro_ai_lab.services import ConnectionService, EmulatorConnected, EmulatorDisconnected
from retro_ai_lab.ui.context import AppContext

DEFAULT_FPS = 60

log = logging.getLogger(__name__)


class _StepSignals(QObject):
    done = Signal(int, object)  # the step's generation, then the frame count or the exception


class _StepTask(QRunnable):
    """One ``advance_frames`` plus the memory snapshot after it, off the GUI thread."""

    def __init__(
        self, connection: ConnectionService, observation: object, n: int, generation: int
    ) -> None:
        super().__init__()
        self.signals = _StepSignals()
        self._connection = connection
        self._observation = observation
        self._n = n
        self._generation = generation

    def run(self) -> None:
        n = self._n
        try:
            self._connection.call(lambda a: a.advance_frames(n))
            self._observation.advance(n)  # type: ignore[attr-defined]
        except Exception as exc:
            self.signals.done.emit(self._generation, exc)
        else:
            self.signals.done.emit(self._generation, n)


class PlayClock(QObject):
    state_changed = Signal(bool)  # running?

    def __init__(self, context: AppContext, fps: int = DEFAULT_FPS, parent: QObject | None = None):
        super().__init__(parent)
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._connection: ConnectionService = context.connection
        self._observation = context.observation
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, 1000 // fps))
        self._timer.timeout.connect(self.pulse)
        self._available = False
        self._wanted = True  # the child's Play/Pause choice
        self._frames = 0
        self._errors = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._in_flight = False
        self._generation = 0  # bumped by _stop(): results from before it are dropped
        self._skipped = 0
        unsubscribe = (
            context.bridge.subscribe(EmulatorConnected, self._on_connected),
            context.bridge.subscribe(EmulatorDisconnected, self._on_disconnected),
        )
        timer = self._timer

        def release(*_: object) -> None:
            timer.stop()
            for off in unsubscribe:
                off()

        self.destroyed.connect(release)

    @property
    def available(self) -> bool:
        """True while a connected emulator can step frames."""
        return self._available

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    @property
    def playing(self) -> bool:
        return self._wanted

    @property
    def frames(self) -> int:
        """Frames advanced since the last connect."""
        return self._frames

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def interval_ms(self) -> int:
        return self._timer.interval()

    @property
    def in_flight(self) -> bool:
        """Is a step running on the worker right now?"""
        return self._in_flight

    @property
    def skipped(self) -> int:
        """Timer pulses that found the previous step still running."""
        return self._skipped

    def play(self) -> None:
        self._wanted = True
        self._apply()

    def pause(self) -> None:
        self._wanted = False
        self._apply()

    def toggle(self) -> bool:
        self.pause() if self._wanted else self.play()
        return self._wanted

    def pulse(self) -> bool:
        """Advance one frame on the worker. Returns True if a step started.

        Nothing starts, and the clock stops, when the emulator cannot step or
        is gone. Nothing starts either while the previous step is still
        running: that pulse is skipped, so the game runs as fast as the link
        allows instead of queueing up frames.
        """
        if not self._available or not self._connection.is_connected:
            self._stop()
            return False
        if self._in_flight:
            self._skipped += 1
            return False
        self._in_flight = True
        task = _StepTask(self._connection, self._observation, 1, self._generation)
        task.signals.done.connect(self._on_step_done)
        self._pool.start(task)
        return True

    def _on_step_done(self, generation: int, result: object) -> None:
        """Back on the GUI thread with the frame count, or with what went wrong."""
        self._in_flight = False
        if generation != self._generation:
            return  # paused or disconnected while this step was running
        if isinstance(result, EmulatorDisconnectedError):
            self._stop()
            return
        if isinstance(result, EmulatorError):
            self._errors += 1
            return
        if isinstance(result, Exception):
            self._errors += 1
            log.warning("Frame step failed: %s", result)
            return
        if isinstance(result, int):
            self._frames += result

    def tick(self, n: int = 1) -> bool:
        """Advance ``n`` frames now, on this thread. Returns True if frames were advanced.

        The timer uses :meth:`pulse` instead; this is for tests and tooling
        that drive time by hand and want the result before going on.
        """
        if not self._available or not self._connection.is_connected:
            self._stop()
            return False
        try:
            self._connection.call(lambda a: a.advance_frames(n))
        except EmulatorDisconnectedError:
            self._stop()
            return False
        except EmulatorError:
            self._errors += 1
            return False
        self._frames += n
        self._observation.advance(n)
        return True

    # ------------------------------------------------------------ internal

    def _apply(self) -> None:
        want = self._wanted and self._available
        if want and not self._timer.isActive():
            self._timer.start()
            self.state_changed.emit(True)
        elif not want and self._timer.isActive():
            self._stop()

    def _stop(self) -> None:
        self._generation += 1
        if self._timer.isActive():
            self._timer.stop()
            self.state_changed.emit(False)

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._available = event.capabilities.frame_advance
        self._frames = 0
        self._apply()

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._available = False
        self._stop()


__all__ = ["DEFAULT_FPS", "PlayClock"]
