"""Deliver :class:`EventBus` events on the GUI thread.

Services may publish from worker threads (connecting, training). Qt widgets
must only be touched from the GUI thread, so UI code subscribes through this
bridge instead of the bus directly. Events published on the GUI thread are
delivered immediately; events from other threads are queued onto it.

Each delivery is timed in the app's work log under ``on <EventName>``
(T-906), so the stall watch's line can name the handler behind a stall.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from PySide6.QtCore import QObject, Signal
from shiboken6 import isValid

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.core.worklog import WorkLog, app_work

E = TypeVar("E")


class EventBridge(QObject):
    _received = Signal(object)

    def __init__(
        self, bus: EventBus, parent: QObject | None = None, *, worklog: WorkLog = app_work
    ) -> None:
        super().__init__(parent)
        self._bus = bus
        self._local = EventBus()
        self._worklog = worklog
        self._received.connect(self._deliver)  # AutoConnection: queued across threads
        self._unsubscribe = bus.subscribe(object, self._forward)
        # Qt may delete the C++ side while the bus still holds the handler
        # (a context outliving its window in tests); drop forwarding then.
        self.destroyed.connect(self._unsubscribe)

    def _forward(self, event: Any) -> None:
        if isValid(self):
            self._received.emit(event)

    def _deliver(self, event: Any) -> None:
        """Hand the event to the GUI subscribers, timed by its name."""
        with self._worklog.timed(f"on {type(event).__name__}"):
            self._local.publish(event)

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> Callable[[], None]:
        return self._local.subscribe(event_type, handler)

    def unsubscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        self._local.unsubscribe(event_type, handler)

    def handler_count(self, event_type: type | None = None) -> int:
        return self._local.handler_count(event_type)

    def detach(self) -> None:
        """Stop forwarding from the bus (used at shutdown)."""
        self._unsubscribe()

    def publish_local(self, event: Any) -> int:
        """Deliver an event to GUI subscribers only (tests and UI-only events)."""
        with self._worklog.timed(f"on {type(event).__name__}"):
            return self._local.publish(event)


__all__ = ["EventBridge"]
