"""Advance the emulator one frame per tick while playing.

Time in Retro AI Lab moves when the app says so: every adapter that supports
``frame_advance`` is stepped by this clock (the fake console only moves this
way; real emulators are driven the same way through their scripts). The
child can pause and resume; training modes drive the same clock.

The timer's steps run on a worker thread (T-818): a real emulator takes at
least one frame to answer, and the GUI thread must never wait for it.
``tick(n)`` stays synchronous for tests and tooling that drive time by hand.

Each step is one request (T-823): the adapter's bundled ``step`` carries the
pad's held buttons, the memory reads the observation service wants, and,
every ``screenshot_every``-th pulse, a screenshot, which comes out of
:attr:`PlayClock.frame_captured`. A real emulator therefore answers one
message per frame instead of three.

The clock keeps real time (T-888): each timer pulse sends the frames the
wall clock owes since play started, several in one request when the link is
slower than a frame, ``MAX_FRAMES_PER_STEP`` at most, with any longer
backlog forgiven rather than raced through. When the on-screen pad holds
nothing, the step watches the emulator's own controller (T-887) and records
what the person held, so a real controller plays and teaches the AI.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

from retro_ai_lab.core import Button, Capabilities, EmulatorDisconnectedError, EmulatorError
from retro_ai_lab.services import ConnectionService, EmulatorConnected, EmulatorDisconnected
from retro_ai_lab.services.observer import ObservationService
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.worker import report

DEFAULT_FPS = 60
MAX_FRAMES_PER_STEP = 6  # a tenth of a second at 60 fps: the most one request catches up

log = logging.getLogger(__name__)

InputsProvider = Callable[[], "frozenset[Button] | None"]
Reads = tuple[tuple[str, int, int], ...]


class _StepSignals(QObject):
    done = Signal(int, object)  # the step's generation, then a (frames, Frame|None) or exception


class _StepTask(QRunnable):
    """One bundled ``step`` (buttons, frames, memory, screenshot), off the GUI thread."""

    def __init__(
        self,
        connection: ConnectionService,
        observation: ObservationService,
        n: int,
        generation: int,
        *,
        pressed: frozenset[Button] | None,
        screenshot: bool,
    ) -> None:
        super().__init__()
        self.signals = _StepSignals()
        self._connection = connection
        self._observation = observation
        self._n = n
        self._generation = generation
        self._pressed = pressed
        self._screenshot = screenshot

    def run(self) -> None:
        try:
            frame = _step(
                self._connection,
                self._observation,
                self._n,
                pressed=self._pressed,
                screenshot=self._screenshot,
            )
        except Exception as exc:
            report(self.signals.done, self._generation, exc)
        else:
            report(self.signals.done, self._generation, (self._n, frame))


def _step(
    connection: ConnectionService,
    observation: ObservationService,
    n: int,
    *,
    pressed: frozenset[Button] | None,
    screenshot: bool,
) -> object:
    """Advance ``n`` frames in one request and feed the observation service.

    Returns the screenshot (a ``Frame``) when one was asked for, else None.
    """
    reads: Reads = observation.snapshot_reads()
    watch = pressed is not None and not pressed  # an empty pad leaves the controller to the person
    result = connection.call(
        lambda a: a.step(
            n, pressed=None if watch else pressed, reads=reads, screenshot=screenshot, watch=watch
        )
    )
    if result.pressed is not None:
        # What the person held during these frames, logged at the frame they started on.
        observation.record_input(result.pressed, "human")
    observation.advance(n, snapshot=result.reads if reads else None)
    return result.frame


class PlayClock(QObject):
    state_changed = Signal(bool)  # running?
    frame_captured = Signal(object)  # the Frame a step brought back (T-823)
    step_failed = Signal(str)  # a step went wrong: the message, once per new message (T-828)

    def __init__(
        self,
        context: AppContext,
        fps: int = DEFAULT_FPS,
        parent: QObject | None = None,
        *,
        screenshot_every: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ):
        super().__init__(parent)
        if fps <= 0:
            raise ValueError("fps must be positive")
        if screenshot_every <= 0:
            raise ValueError("screenshot_every must be positive")
        self._fps = fps
        self._clock = clock
        self._origin = clock()  # when play started, for the frames owed (T-888)
        self._sent = 0  # frames sent since the origin
        self._forgiven = 0
        self._early = 0
        self._connection: ConnectionService = context.connection
        self._observation = context.observation
        self._capabilities = Capabilities()
        self._inputs: InputsProvider | None = None
        self._screenshot_every = screenshot_every
        self._pulses = 0
        self._frames_captured = 0
        self._last_failure: str | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, 1000 // fps))
        self._timer.timeout.connect(self._on_timer)
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

    @property
    def steps(self) -> int:
        """Steps (requests) started since connect; each carries one or more frames (T-888)."""
        return self._pulses

    @property
    def forgiven(self) -> int:
        """Frames the wall clock owed beyond a step's cap and the clock let go (T-888)."""
        return self._forgiven

    @property
    def early(self) -> int:
        """Timer pulses that came before a whole frame was owed (T-888)."""
        return self._early

    @property
    def screenshot_every(self) -> int:
        """A screenshot rides along on every n-th pulse (the first included)."""
        return self._screenshot_every

    @property
    def frames_captured(self) -> int:
        """Screenshots delivered through :attr:`frame_captured` since connect."""
        return self._frames_captured

    @property
    def last_failure(self) -> str | None:
        """What the latest step said went wrong, or None after a good step."""
        return self._last_failure

    def set_inputs(self, provider: InputsProvider | None) -> None:
        """Name who knows the held buttons (the pad); each step carries them.

        The provider is called on the GUI thread as a step starts; None from
        it, or no provider, sends a step without buttons.
        """
        self._inputs = provider

    def play(self) -> None:
        self._wanted = True
        self._apply()

    def pause(self) -> None:
        self._wanted = False
        self._apply()

    def toggle(self) -> bool:
        self.pause() if self._wanted else self.play()
        return self._wanted

    def _owed(self) -> int:
        """Frames the wall clock owes since play started, capped; the rest is forgiven.

        A stall (a slow link, a sleeping laptop) would otherwise be followed
        by a burst of catching up; the game simply goes on from where it is.
        """
        # A hair of slack keeps 0.05 s at 100 fps from rounding down to 4 frames.
        should = int((self._clock() - self._origin) * self._fps + 1e-6)
        owed = should - self._sent
        if owed > MAX_FRAMES_PER_STEP:
            self._forgiven += owed - MAX_FRAMES_PER_STEP
            self._sent = should - MAX_FRAMES_PER_STEP
            owed = MAX_FRAMES_PER_STEP
        return owed

    def _on_timer(self) -> None:
        """Each timer pulse sends the frames owed, if any and if the worker is free."""
        if self._in_flight:
            self._skipped += 1
            return
        owed = self._owed()
        if owed < 1:
            self._early += 1
            return
        self.pulse(owed)

    def pulse(self, frames: int = 1) -> bool:
        """Advance ``frames`` frames on the worker. Returns True if a step started.

        Nothing starts, and the clock stops, when the emulator cannot step or
        is gone. Nothing starts either while the previous step is still
        running: that pulse is skipped, so the game runs as fast as the link
        allows instead of queueing up frames. The timer calls this with the
        frames the wall clock owes (T-888); a direct call sends one.
        """
        if frames < 1:
            raise ValueError("frames must be positive")
        if not self._available or not self._connection.is_connected:
            self._stop()
            return False
        if self._in_flight:
            self._skipped += 1
            return False
        self._in_flight = True
        self._sent += frames
        screenshot = self._capabilities.screen_capture and (
            self._pulses % self._screenshot_every == 0
        )
        self._pulses += 1
        task = _StepTask(
            self._connection,
            self._observation,
            frames,
            self._generation,
            pressed=self._pressed(),
            screenshot=screenshot,
        )
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
        if isinstance(result, Exception):
            self._failed(result)
            return
        if isinstance(result, tuple):
            frames, frame = result
            self._frames += frames
            self._last_failure = None
            if frame is not None:
                self._frames_captured += 1
                self.frame_captured.emit(frame)

    def tick(self, n: int = 1) -> bool:
        """Advance ``n`` frames now, on this thread. Returns True if frames were advanced.

        The timer uses :meth:`pulse` instead; this is for tests and tooling
        that drive time by hand and want the result before going on.
        """
        if not self._available or not self._connection.is_connected:
            self._stop()
            return False
        try:
            _step(self._connection, self._observation, n, pressed=self._pressed(), screenshot=False)
        except EmulatorDisconnectedError:
            self._stop()
            return False
        except EmulatorError as exc:
            self._failed(exc)
            return False
        self._frames += n
        self._last_failure = None
        return True

    def _failed(self, exc: Exception) -> None:
        """Count a failed step and say so once, until the message changes (T-828).

        A step that fails every frame would otherwise fill the log and the
        window 60 times a second, and a step that fails silently leaves the
        user with a blank screen and no clue.
        """
        self._errors += 1
        message = str(exc).strip() or exc.__class__.__name__
        if message == self._last_failure:
            return
        self._last_failure = message
        log.warning("Frame step failed: %s", message)
        self.step_failed.emit(message)

    # ------------------------------------------------------------ internal

    def _apply(self) -> None:
        want = self._wanted and self._available
        if want and not self._timer.isActive():
            self._origin = self._clock()
            self._sent = 0
            self._timer.start()
            self.state_changed.emit(True)
        elif not want and self._timer.isActive():
            self._stop()

    def _stop(self) -> None:
        self._generation += 1
        if self._timer.isActive():
            self._timer.stop()
            self.state_changed.emit(False)

    def _pressed(self) -> frozenset[Button] | None:
        """The pad's held set for this step, or None when there is nothing to send."""
        if self._inputs is None or not self._capabilities.input:
            return None
        return self._inputs()

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._available = event.capabilities.frame_advance
        self._capabilities = event.capabilities
        self._frames = 0
        self._pulses = 0
        self._frames_captured = 0
        self._apply()

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._available = False
        self._stop()


__all__ = ["DEFAULT_FPS", "MAX_FRAMES_PER_STEP", "PlayClock"]
