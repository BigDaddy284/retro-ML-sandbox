"""Error hierarchy shared by every layer.

Every error raised by Retro AI Lab derives from :class:`RetroAILabError` so
callers can catch "anything the app did on purpose" separately from bugs.
Messages are written so the service layer can show them to a child after light
rewording; keep them short and free of jargon where possible.
"""

from __future__ import annotations


class RetroAILabError(Exception):
    """Base class for all application errors."""


class ConfigurationError(RetroAILabError):
    """A setting, file or argument is invalid."""


class EmulatorError(RetroAILabError):
    """Base class for anything that goes wrong talking to an emulator."""


class EmulatorDisconnectedError(EmulatorError):
    """The emulator is not connected (never connected, or the link dropped)."""

    def __init__(self, message: str = "The emulator is not connected.") -> None:
        super().__init__(message)


class UnsupportedCapabilityError(EmulatorError):
    """The connected emulator cannot do what was asked.

    ``capability`` names the missing feature (see ``Capabilities`` field
    names) so the UI can explain exactly what is unavailable.
    """

    def __init__(self, capability: str, message: str | None = None) -> None:
        self.capability = capability
        super().__init__(message or f"This emulator does not support '{capability}'.")


class ProtocolError(EmulatorError):
    """The emulator sent something we could not understand."""


__all__ = [
    "ConfigurationError",
    "EmulatorDisconnectedError",
    "EmulatorError",
    "ProtocolError",
    "RetroAILabError",
    "UnsupportedCapabilityError",
]
