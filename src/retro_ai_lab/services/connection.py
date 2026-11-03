"""Connection service: owns the one live emulator adapter and its lifecycle.

State machine::

    DISCONNECTED --connect()--> CONNECTING --ok--> CONNECTED
         ^                          |                  |
         |----------fail------------+                  |
         +------------------disconnect()/link drop-----+

The service publishes events on an :class:`EventBus` so the UI never has to
poll. Every operation on the live adapter should go through :meth:`call`, which
turns an unexpected ``EmulatorDisconnectedError`` into a state change plus an
event before re-raising, so the rest of the app learns about a dropped link
exactly once.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from retro_ai_lab.core import (
    Button,
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
    GameInfo,
)
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.base import DEFAULT_CONNECT_TIMEOUT_S, EmulatorAdapter
from retro_ai_lab.emulator.registry import AdapterRegistry

T = TypeVar("T")
log = logging.getLogger(__name__)

LINK_LOST = "The emulator went away."


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"


# ---------------------------------------------------------------- events


@dataclass(frozen=True, slots=True)
class ConnectionStateChanged:
    old: ConnectionState
    new: ConnectionState
    adapter_name: str | None


@dataclass(frozen=True, slots=True)
class EmulatorConnected:
    adapter_name: str
    game_info: GameInfo
    capabilities: Capabilities
    buttons: tuple[Button, ...]


@dataclass(frozen=True, slots=True)
class EmulatorDisconnected:
    adapter_name: str
    expected: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ConnectionFailed:
    adapter_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class Session:
    """What is known about the live connection."""

    adapter_name: str
    display_name: str
    game_info: GameInfo
    capabilities: Capabilities
    buttons: tuple[Button, ...]


# --------------------------------------------------------------- service


class ConnectionService:
    def __init__(self, registry: AdapterRegistry, bus: EventBus) -> None:
        self._registry = registry
        self._bus = bus
        self._state = ConnectionState.DISCONNECTED
        self._adapter: EmulatorAdapter | None = None
        self._session: Session | None = None
        # Workers (frame pump, play clock) may find a dropped link at the same
        # moment; the loss is still recorded and announced exactly once.
        self._lock = threading.RLock()

    # ------------------------------------------------------------ queries

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state is ConnectionState.CONNECTED

    @property
    def adapter(self) -> EmulatorAdapter | None:
        return self._adapter if self.is_connected else None

    @property
    def session(self) -> Session | None:
        return self._session if self.is_connected else None

    def available_adapters(self) -> Sequence[tuple[str, str]]:
        return tuple((i.name, i.display_name) for i in self._registry.describe())

    # ---------------------------------------------------------- lifecycle

    def connect(
        self,
        adapter_name: str,
        *,
        timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        **options: Any,
    ) -> Session:
        if self._state is not ConnectionState.DISCONNECTED:
            raise ConfigurationError(
                "Already connected to an emulator. Disconnect before connecting again."
            )
        adapter = self._registry.create(adapter_name, **options)  # validates the name
        self._set_state(ConnectionState.CONNECTING, adapter_name)
        try:
            adapter.connect(timeout_s)
            session = Session(
                adapter_name=adapter.name,
                display_name=adapter.display_name,
                game_info=adapter.game_info(),
                capabilities=adapter.capabilities(),
                buttons=tuple(adapter.buttons()),
            )
        except Exception as exc:
            with contextlib.suppress(Exception):
                adapter.disconnect()
            self._set_state(ConnectionState.DISCONNECTED, None)
            self._bus.publish(ConnectionFailed(adapter_name, _message(exc)))
            raise
        self._adapter = adapter
        self._session = session
        self._set_state(ConnectionState.CONNECTED, adapter_name)
        self._bus.publish(
            EmulatorConnected(
                adapter.name, session.game_info, session.capabilities, session.buttons
            )
        )
        return session

    def disconnect(self) -> str:
        """Close the connection. A no-op when not connected.

        Returns the reason recorded in the ``EmulatorDisconnected`` event. A
        backend that fails while closing is still treated as disconnected
        (the failure is logged and mentioned in the reason, never raised),
        because from the child's point of view the game is gone either way.
        """
        # Held throughout (T-820): a worker whose call fails while the adapter
        # is closing must find the disconnect already recorded, not record an
        # unexpected loss of its own.
        with self._lock:
            if self._state is not ConnectionState.CONNECTED or self._adapter is None:
                return ""
            adapter = self._adapter
            reason = "Disconnected."
            try:
                adapter.disconnect()
            except EmulatorError as exc:
                log.warning("Adapter %s failed while disconnecting: %s", adapter.name, exc)
                reason = f"Disconnected, although the emulator complained: {_message(exc)}"
            finally:
                self._finish_disconnect(expected=True, reason=reason)
            return reason

    def check(self) -> bool:
        """Verify the live link is still up; report a silent drop exactly once.

        Returns True if connected after the check. Use it from a watchdog so a
        link that dies while nothing is being sent is noticed promptly.
        """
        if self._state is not ConnectionState.CONNECTED or self._adapter is None:
            return False
        if self._adapter.check_link():
            return True
        self._finish_disconnect(expected=False, reason=LINK_LOST)
        return False

    def call(self, operation: Callable[[EmulatorAdapter], T]) -> T:
        """Run ``operation`` on the live adapter.

        Raises ``EmulatorDisconnectedError`` if there is no connection, and if
        the adapter reports a dropped link mid-operation the service records
        the disconnect (state + event) before re-raising.
        """
        adapter = self.adapter
        if adapter is None:
            raise EmulatorDisconnectedError()
        try:
            return operation(adapter)
        except EmulatorDisconnectedError as exc:
            reason = _message(exc)
            if reason == str(EmulatorDisconnectedError()):
                reason = LINK_LOST  # the generic message; say something a child can read
            self._finish_disconnect(expected=False, reason=reason)
            raise

    # ------------------------------------------------------------ helpers

    def _finish_disconnect(self, *, expected: bool, reason: str) -> None:
        with self._lock:
            if self._state is not ConnectionState.CONNECTED:
                return  # another thread already recorded this loss
            name = self._session.adapter_name if self._session else "unknown"
            self._adapter = None
            self._session = None
            self._set_state(ConnectionState.DISCONNECTED, None)
            self._bus.publish(EmulatorDisconnected(name, expected, reason))

    def _set_state(self, new: ConnectionState, adapter_name: str | None) -> None:
        old = self._state
        if old is new:
            return
        self._state = new
        self._bus.publish(ConnectionStateChanged(old, new, adapter_name))


def _message(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


__all__ = [
    "LINK_LOST",
    "ConnectionFailed",
    "ConnectionService",
    "ConnectionState",
    "ConnectionStateChanged",
    "EmulatorConnected",
    "EmulatorDisconnected",
    "Session",
]
