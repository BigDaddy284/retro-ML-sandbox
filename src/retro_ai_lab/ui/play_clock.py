"""Advance the emulator one frame per tick while playing.

Time in Retro AI Lab moves when the app says so: every adapter that supports
``frame_advance`` is stepped by this clock (the fake console only moves this
way; real emulators will be driven the same way through their scripts). The
child can pause and resume; later, training modes drive the same clock.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

from retro_ai_lab.core import EmulatorDisconnectedError, EmulatorError
from retro_ai_lab.services import ConnectionService, EmulatorConnected, EmulatorDisconnected
from retro_ai_lab.ui.context import AppContext

DEFAULT_FPS = 60


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
        self._timer.timeout.connect(self.tick)
        self._available = False
        self._wanted = True  # the child's Play/Pause choice
        self._frames = 0
        self._errors = 0
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

    def play(self) -> None:
        self._wanted = True
        self._apply()

    def pause(self) -> None:
        self._wanted = False
        self._apply()

    def toggle(self) -> bool:
        self.pause() if self._wanted else self.play()
        return self._wanted

    def tick(self, n: int = 1) -> bool:
        """Advance ``n`` frames now. Returns True if frames were advanced."""
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
