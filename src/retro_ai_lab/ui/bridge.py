"""Deliver :class:`EventBus` events on the GUI thread.

Services may publish from worker threads (connecting, training). Qt widgets
must only be touched from the GUI thread, so UI code subscribes through this
bridge instead of the bus directly. Events published on the GUI thread are
delivered immediately; events from other threads are queued onto it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from PySide6.QtCore import QObject, Signal

from retro_ai_lab.core.events import EventBus

E = TypeVar("E")


class EventBridge(QObject):
    _received = Signal(object)

    def __init__(self, bus: EventBus, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bus = bus
        self._local = EventBus()
        self._received.connect(self._local.publish)  # AutoConnection: queued across threads
        self._unsubscribe = bus.subscribe(object, self._received.emit)

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
        return self._local.publish(event)


__all__ = ["EventBridge"]
