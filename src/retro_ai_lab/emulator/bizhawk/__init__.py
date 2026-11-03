"""The BizHawk (EmuHawk) adapter and its bridge script (ARCHITECTURE.md §5)."""

from retro_ai_lab.emulator.bizhawk.adapter import (
    DEFAULT_CALL_TIMEOUT_S,
    SCRIPT_PATH,
    BizHawkAdapter,
)
from retro_ai_lab.emulator.bizhawk.protocol import DEFAULT_HOST, DEFAULT_PORT, PROTOCOL_VERSION
from retro_ai_lab.emulator.bizhawk.simulator import BizHawkSimulator

__all__ = [
    "DEFAULT_CALL_TIMEOUT_S",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "PROTOCOL_VERSION",
    "SCRIPT_PATH",
    "BizHawkAdapter",
    "BizHawkSimulator",
]
