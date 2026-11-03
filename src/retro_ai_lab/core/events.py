"""A small synchronous event bus.

Lower layers publish plain frozen-dataclass events; the UI and services
subscribe. Handlers run in subscription order on the publisher's thread. A
handler that raises does not stop the others: the error is passed to the
bus's ``on_error`` callback (default: logged) so one bad listener cannot take
down training.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

log = logging.getLogger(__name__)

E = TypeVar("E")
Handler = Callable[[Any], None]
ErrorHook = Callable[[BaseException, Any, Handler], None]


def _log_error(exc: BaseException, event: Any, handler: Handler) -> None:
    log.exception("Event handler %r failed on %r: %s", handler, event, exc)


class EventBus:
    def __init__(self, on_error: ErrorHook | None = None) -> None:
        self._handlers: dict[type, list[Handler]] = defaultdict(list)
        self._on_error: ErrorHook = on_error or _log_error

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> Callable[[], None]:
        """Register ``handler`` for ``event_type`` (and subclasses).

        Returns a function that unsubscribes it again.
        """
        if not isinstance(event_type, type):
            raise TypeError("event_type must be a class.")
        if not callable(handler):
            raise TypeError("handler must be callable.")
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            self.unsubscribe(event_type, handler)

        return unsubscribe

    def unsubscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        """Remove ``handler``; removing one that is not registered is a no-op."""
        handlers = self._handlers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)

    def publish(self, event: object) -> int:
        """Deliver ``event`` to matching handlers. Returns how many ran."""
        delivered = 0
        for event_type, handlers in list(self._handlers.items()):
            if not isinstance(event, event_type):
                continue
            for handler in list(handlers):
                try:
                    handler(event)
                except Exception as exc:
                    self._on_error(exc, event, handler)
                delivered += 1
        return delivered

    def handler_count(self, event_type: type | None = None) -> int:
        if event_type is None:
            return sum(len(h) for h in self._handlers.values())
        return len(self._handlers.get(event_type, []))


__all__ = ["EventBus"]
