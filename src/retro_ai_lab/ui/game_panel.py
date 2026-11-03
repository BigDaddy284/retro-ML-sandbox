"""The Game panel: live emulator screen plus a one-line status.

:class:`FramePump` asks the connection service for frames on a timer while
an emulator that can capture its screen is connected. Each capture runs on a
worker thread, because a real emulator answers over a socket in tens of
milliseconds and the GUI thread must never wait for it (T-817); the frame
comes back to the GUI thread by signal. The pump is driven by service
events, so the panel never has to know how a connection was made.

While the play clock runs it brings a screenshot back with every k-th frame
step (T-823), so the pump stays stopped then and only serves a paused game,
training, and emulators that cannot step frames.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, SignalInstance
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from retro_ai_lab.core import EmulatorDisconnectedError, Frame
from retro_ai_lab.services import ConnectionService, EmulatorConnected, EmulatorDisconnected
from retro_ai_lab.training import TrainingStateChanged
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.controller_pad import ControllerPad
from retro_ai_lab.ui.frame_view import FrameView
from retro_ai_lab.ui.play_clock import PlayClock
from retro_ai_lab.ui.worker import report

DEFAULT_FPS = 30
CLOCK_FPS = 60  # the play clock's frame rate (console speed); pictures ride along at DEFAULT_FPS
NOT_CONNECTED = "Not connected. Connect to a game to see it here."
NO_SCREEN = "This emulator can't show its screen, but the AI can still play."
STEP_FAILED = "The emulator did not finish a frame step:"
CAPTURE_FAILED = "The emulator did not send a picture:"

log = logging.getLogger(__name__)


class _CaptureSignals(QObject):
    done = Signal(int, object)  # the capture's generation, then the Frame or the exception
    failed = Signal(str)  # a capture went wrong: the message, once per new message (T-828)


class _CaptureTask(QRunnable):
    """One ``capture_frame`` call, run off the GUI thread."""

    def __init__(self, connection: ConnectionService, generation: int) -> None:
        super().__init__()
        self.signals = _CaptureSignals()
        self._connection = connection
        self._generation = generation

    def run(self) -> None:
        try:
            frame = self._connection.call(lambda a: a.capture_frame())
        except Exception as exc:
            report(self.signals.done, self._generation, exc)
        else:
            report(self.signals.done, self._generation, frame)


class FramePump(QObject):
    """Fetch frames from the live adapter at a steady rate."""

    def __init__(
        self,
        connection: ConnectionService,
        fps: int = DEFAULT_FPS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._connection = connection
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, 1000 // fps))
        self._timer.timeout.connect(self.tick)
        self._on_frame: list[object] = []
        self._latest: Frame | None = None
        self._errors = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._in_flight = False
        self._generation = 0  # bumped by stop(): results from before it are dropped
        self._skipped = 0
        self._captured = 0
        self._last_failure: str | None = None
        self._signals = _CaptureSignals()

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    @property
    def interval_ms(self) -> int:
        return self._timer.interval()

    @property
    def latest(self) -> Frame | None:
        return self._latest

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def in_flight(self) -> bool:
        """Is a capture running on the worker right now?"""
        return self._in_flight

    @property
    def skipped(self) -> int:
        """Ticks that found the previous capture still running."""
        return self._skipped

    @property
    def captured(self) -> int:
        """Frames delivered since the pump was made."""
        return self._captured

    @property
    def failed(self) -> SignalInstance:
        """Emits the message of a failed capture, once per new message (T-828)."""
        return self._signals.failed

    @property
    def last_failure(self) -> str | None:
        return self._last_failure

    def _failed(self, message: str) -> None:
        self._errors += 1
        if message == self._last_failure:
            return
        self._last_failure = message
        log.warning("Frame capture failed: %s", message)
        self._signals.failed.emit(message)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._latest = None
        self._generation += 1

    def subscribe(self, callback: object) -> None:
        self._on_frame.append(callback)

    def tick(self) -> bool:
        """Ask for a frame now, on the worker. Returns True if a capture started.

        Nothing starts, and the pump stops, when no emulator is connected.
        Nothing starts either while the previous capture is still running:
        that tick is skipped, so the real frame rate settles to whatever the
        link can deliver instead of piling up requests.
        """
        if not self._connection.is_connected:
            self.stop()
            return False
        if self._in_flight:
            self._skipped += 1
            return False
        self._in_flight = True
        task = _CaptureTask(self._connection, self._generation)
        task.signals.done.connect(self._on_done)
        self._pool.start(task)
        return True

    def _on_done(self, generation: int, result: object) -> None:
        """Back on the GUI thread with the frame, or with what went wrong.

        A dropped link is reported by the connection service itself (it
        publishes ``EmulatorDisconnected``), so here we only stop quietly.
        """
        self._in_flight = False
        if generation != self._generation:
            return  # stopped (or disconnected) while this capture was running
        if isinstance(result, EmulatorDisconnectedError):
            self.stop()
            return
        if isinstance(result, Exception):
            self._failed(str(result).strip() or result.__class__.__name__)
            return
        if not isinstance(result, Frame):
            self._failed(f"The capture returned {type(result).__name__!r}, not a frame.")
            return
        self._latest = result
        self._captured += 1
        self._last_failure = None
        for cb in self._on_frame:
            cb(result)  # type: ignore[operator]


class GamePanel(QWidget):
    def __init__(
        self,
        context: AppContext,
        parent: QWidget | None = None,
        fps: int = DEFAULT_FPS,
        clock_fps: int = CLOCK_FPS,
    ):
        super().__init__(parent)
        self.setObjectName("game_panel")
        self._context = context
        self._view = FrameView(self)
        self._view.set_placeholder(NOT_CONNECTED)
        self._status = QLabel(NOT_CONNECTED, self)
        self._status.setObjectName("game_status")
        self._status.setWordWrap(True)
        self._clock = PlayClock(
            context, fps=clock_fps, parent=self, screenshot_every=max(1, clock_fps // fps)
        )
        self._play_button = QPushButton("Pause", self)
        self._play_button.setObjectName("play_button")
        self._play_button.setEnabled(False)
        self._play_button.clicked.connect(self._on_play_clicked)
        self._clock.state_changed.connect(self._on_clock_state)
        self._pad = ControllerPad(context, self)
        self._clock.set_inputs(lambda: self._pad.held)
        self._clock.frame_captured.connect(self._on_clock_frame)
        self._clock.step_failed.connect(self._on_step_failed)

        controls = QHBoxLayout()
        controls.addWidget(self._play_button)
        controls.addWidget(self._pad, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view, 1)
        layout.addLayout(controls)
        layout.addWidget(self._status)

        self._pump = FramePump(context.connection, fps=fps, parent=self)
        self._pump.subscribe(context.observation.observe_frame)
        self._pump.subscribe(self._on_frame)
        self._pump.failed.connect(self._on_capture_failed)
        self._ai_had_controls = False
        self._resume_clock = False
        unsubscribe = (
            context.bridge.subscribe(EmulatorConnected, self._on_connected),
            context.bridge.subscribe(EmulatorDisconnected, self._on_disconnected),
            context.bridge.subscribe(TrainingStateChanged, self._on_training_state),
        )
        # ``destroyed`` fires before children are deleted, so the pump is still
        # alive here. Capture what we need rather than ``self``, which is gone.
        pump = self._pump
        self.destroyed.connect(lambda *_: _release(pump, unsubscribe))

    # --------------------------------------------------------------- API

    @property
    def view(self) -> FrameView:
        return self._view

    @property
    def pump(self) -> FramePump:
        return self._pump

    @property
    def clock(self) -> PlayClock:
        return self._clock

    @property
    def pad(self) -> ControllerPad:
        return self._pad

    @property
    def play_button(self) -> QPushButton:
        return self._play_button

    @property
    def status_text(self) -> str:
        return self._status.text()

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    # ------------------------------------------------------------ events

    def _on_play_clicked(self) -> None:
        self._clock.toggle()
        self._refresh_play_button()

    def _on_clock_state(self, running: bool) -> None:
        self._refresh_play_button()
        # One request per frame: the clock's steps carry the screenshots while
        # it runs, and the pump takes over again when the game is paused.
        if running:
            self._pump.stop()
        elif self._can_show_screen():
            self._pump.start()
            self._pump.tick()

    def _can_show_screen(self) -> bool:
        session = self._context.connection.session
        return session is not None and session.capabilities.screen_capture

    def _refresh_play_button(self) -> None:
        self._play_button.setEnabled(self._clock.available and not self._ai_had_controls)
        self._play_button.setText("Pause" if self._clock.playing else "Play")

    def _on_training_state(self, event: TrainingStateChanged) -> None:
        """The game's own controls yield to the AI while it plays, and come back after."""
        ai = self._context.training.ai_in_control
        if ai and not self._ai_had_controls:
            self._resume_clock = self._clock.playing
            self._clock.pause()
            self._pad.set_locked(True)
        elif not ai and self._ai_had_controls:
            self._pad.set_locked(False)
            if self._resume_clock and self._clock.available:
                self._clock.play()
        self._ai_had_controls = ai
        self._refresh_play_button()

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._refresh_play_button()
        if not event.capabilities.screen_capture:
            self._view.clear()
            self._view.set_placeholder(NO_SCREEN)
            self.set_status(f"Connected to {event.game_info.title}. {NO_SCREEN}")
            return
        self.set_status(f"Watching {event.game_info.title}.")
        if not self._clock.running:
            self._pump.start()
            self._pump.tick()  # show something straight away

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._ai_had_controls = False
        self._refresh_play_button()
        self._pump.stop()
        self._view.clear()
        self._view.set_placeholder(NOT_CONNECTED)
        if event.expected:
            self.set_status(NOT_CONNECTED)
        else:
            self.set_status(f"Lost the game: {event.reason} {NOT_CONNECTED}")

    def _on_step_failed(self, message: str) -> None:
        """A frame step went wrong: say so where the user looks (T-828)."""
        self.set_status(f"{self._status_prefix()} {STEP_FAILED} {message}")

    def _on_capture_failed(self, message: str) -> None:
        self.set_status(f"{self._status_prefix()} {CAPTURE_FAILED} {message}")

    def _on_clock_frame(self, frame: Frame) -> None:
        """A screenshot that rode along on a frame step (T-823)."""
        self._context.observation.observe_frame(frame)
        self._on_frame(frame)

    def _on_frame(self, frame: Frame) -> None:
        self._view.set_frame(frame)
        self._status.setText(f"{self._status_prefix()} Frame {frame.number}.")

    def _status_prefix(self) -> str:
        session = self._context.connection.session
        return f"Watching {session.game_info.title}." if session else ""


def _release(pump: FramePump, unsubscribe: tuple[Callable[[], None], ...]) -> None:
    pump.stop()
    for off in unsubscribe:
        off()


__all__ = [
    "CAPTURE_FAILED",
    "CLOCK_FPS",
    "DEFAULT_FPS",
    "NOT_CONNECTED",
    "NO_SCREEN",
    "STEP_FAILED",
    "FramePump",
    "GamePanel",
]
