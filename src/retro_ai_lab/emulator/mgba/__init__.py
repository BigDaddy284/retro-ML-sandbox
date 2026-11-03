"""The mGBA bridge script (ARCHITECTURE.md §6, ADR-010); the adapter is T-811."""

from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("retro_ai_lab.lua")  # the bridge the app ships

__all__ = ["SCRIPT_PATH"]
