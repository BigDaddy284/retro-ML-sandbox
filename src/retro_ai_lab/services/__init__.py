"""Session orchestration: connect, observe, train. No Qt below this line."""

from retro_ai_lab.services.connection import (
    ConnectionFailed,
    ConnectionService,
    ConnectionState,
    ConnectionStateChanged,
    EmulatorConnected,
    EmulatorDisconnected,
    Session,
)

__all__ = [
    "ConnectionFailed",
    "ConnectionService",
    "ConnectionState",
    "ConnectionStateChanged",
    "EmulatorConnected",
    "EmulatorDisconnected",
    "Session",
]
