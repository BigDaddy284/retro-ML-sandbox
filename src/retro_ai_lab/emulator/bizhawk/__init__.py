"""The BizHawk (EmuHawk) adapter and its bridge script (ARCHITECTURE.md §5)."""

from pathlib import Path

from retro_ai_lab.emulator.bizhawk.adapter import DEFAULT_CALL_TIMEOUT_S, BizHawkAdapter
from retro_ai_lab.emulator.bizhawk.protocol import DEFAULT_HOST, DEFAULT_PORT, PROTOCOL_VERSION
from retro_ai_lab.emulator.bizhawk.simulator import BizHawkSimulator

SCRIPT_PATH = Path(__file__).with_name("retro_ai_lab.lua")
"""Where the shipped bridge script lives (open it in EmuHawk's Lua Console)."""

__all__ = [
    "DEFAULT_CALL_TIMEOUT_S",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "PROTOCOL_VERSION",
    "SCRIPT_PATH",
    "BizHawkAdapter",
    "BizHawkSimulator",
]
