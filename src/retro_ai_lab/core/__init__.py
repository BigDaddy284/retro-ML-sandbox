"""Shared domain types, errors and configuration used by every layer."""

from retro_ai_lab.core.errors import (
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
    ProtocolError,
    RetroAILabError,
    UnsupportedCapabilityError,
)
from retro_ai_lab.core.types import (
    Button,
    Capabilities,
    Frame,
    GameInfo,
    MemoryRegion,
    StepResult,
)

__all__ = [
    "Button",
    "Capabilities",
    "ConfigurationError",
    "EmulatorDisconnectedError",
    "EmulatorError",
    "Frame",
    "GameInfo",
    "MemoryRegion",
    "ProtocolError",
    "RetroAILabError",
    "StepResult",
    "UnsupportedCapabilityError",
]
