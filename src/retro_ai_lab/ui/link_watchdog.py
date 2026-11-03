"""Notice a link that dies while nothing is being sent.

The connection service only learns about a dropped link when an operation
fails. Between operations (paused, or an emulator that drives its own clock)
nothing would be sent, so this watchdog asks the service to check the link on
a timer. The service reports the loss exactly once, as usual.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer

from retro_ai_lab.services import ConnectionService

DEFAULT_INTERVAL_MS = 500


class LinkWatchdog(QObject):
    def __init__(
        self,
        connection: ConnectionService,
        interval_ms: int = DEFAULT_INTERVAL_MS,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self._connection = connection
        self._checks = 0
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.check)
        timer = self._timer
        self.destroyed.connect(lambda *_: timer.stop())

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    @property
    def checks(self) -> int:
        return self._checks

    @property
    def interval_ms(self) -> int:
        return self._timer.interval()

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def check(self) -> bool:
        self._checks += 1
        return self._connection.check()


__all__ = ["DEFAULT_INTERVAL_MS", "LinkWatchdog"]
