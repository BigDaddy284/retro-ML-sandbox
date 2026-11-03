"""The Game panel: live emulator screen plus a one-line status.

:class:`FramePump` polls the connection service for frames on a timer while
an emulator that can capture its screen is connected. It is driven by
service events, so the panel never has to know how a connection was made.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from retro_ai_lab.core import EmulatorDisconnectedError, EmulatorError, Frame
from retro_ai_lab.services import ConnectionService, EmulatorConnected, EmulatorDisconnected
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.controller_pad import ControllerPad
from retro_ai_lab.ui.frame_view import FrameView
from retro_ai_lab.ui.play_clock import PlayClock

DEFAULT_FPS = 30
NOT_CONNECTED = "Not connected. Connect to a game to see it here."
NO_SCREEN = "This emulator can't show its screen, but the AI can still play."


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

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._latest = None

    def subscribe(self, callback: object) -> None:
        self._on_frame.append(callback)

    def tick(self) -> Frame | None:
        """Capture one frame now. Returns it, or None if nothing was captured.

        A dropped link is reported by the connection service itself (it
        publishes ``EmulatorDisconnected``), so here we only stop quietly.
        """
        if not self._connection.is_connected:
            self.stop()
            return None
        try:
            frame = self._connection.call(lambda a: a.capture_frame())
        except EmulatorDisconnectedError:
            self.stop()
            return None
        except EmulatorError:
            self._errors += 1
            return None
        self._latest = frame
        for cb in self._on_frame:
            cb(frame)  # type: ignore[operator]
        return frame


class GamePanel(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None, fps: int = DEFAULT_FPS):
        super().__init__(parent)
        self.setObjectName("game_panel")
        self._context = context
        self._view = FrameView(self)
        self._view.set_placeholder(NOT_CONNECTED)
        self._status = QLabel(NOT_CONNECTED, self)
        self._status.setObjectName("game_status")
        self._status.setWordWrap(True)
        self._clock = PlayClock(context, fps=fps, parent=self)
        self._play_button = QPushButton("Pause", self)
        self._play_button.setObjectName("play_button")
        self._play_button.setEnabled(False)
        self._play_button.clicked.connect(self._on_play_clicked)
        self._clock.state_changed.connect(self._on_clock_state)
        self._pad = ControllerPad(context, self)

        controls = QHBoxLayout()
        controls.addWidget(self._play_button)
        controls.addWidget(self._pad, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view, 1)
        layout.addLayout(controls)
        layout.addWidget(self._status)

        self._pump = FramePump(context.connection, fps=fps, parent=self)
        self._pump.subscribe(self._on_frame)
        unsubscribe = (
            context.bridge.subscribe(EmulatorConnected, self._on_connected),
            context.bridge.subscribe(EmulatorDisconnected, self._on_disconnected),
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

    def _refresh_play_button(self) -> None:
        self._play_button.setEnabled(self._clock.available)
        self._play_button.setText("Pause" if self._clock.playing else "Play")

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._refresh_play_button()
        if not event.capabilities.screen_capture:
            self._view.clear()
            self._view.set_placeholder(NO_SCREEN)
            self.set_status(f"Connected to {event.game_info.title}. {NO_SCREEN}")
            return
        self.set_status(f"Watching {event.game_info.title}.")
        self._pump.start()
        self._pump.tick()  # show something straight away

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._refresh_play_button()
        self._pump.stop()
        self._view.clear()
        self._view.set_placeholder(NOT_CONNECTED)
        if event.expected:
            self.set_status(NOT_CONNECTED)
        else:
            self.set_status(f"Lost the game: {event.reason} {NOT_CONNECTED}")

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


__all__ = ["DEFAULT_FPS", "NOT_CONNECTED", "NO_SCREEN", "FramePump", "GamePanel"]
