"""``MgbaAdapter``: mGBA through the T-810 script (ARCHITECTURE.md §6, ADR-010).

The script speaks the same protocol as the BizHawk one, so this adapter is the
BizHawk adapter under another name, with its own script and advice. What
differs is what the handshake says: mGBA can't hold the game still between
requests (``hold: false``) and has no speed control, which the base adapter
already reads from ``hello`` and from the command list.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from retro_ai_lab.emulator.bizhawk.adapter import BizHawkAdapter

SCRIPT_PATH = Path(__file__).with_name("retro_ai_lab.lua")  # the bridge the app ships
MGBA_HINT = (
    "Load retro_ai_lab.lua in mGBA (Tools > Scripting... > File > Load script) and try again."
)


class MgbaAdapter(BizHawkAdapter):
    name = "mgba"
    display_name = "mGBA"
    emulator_name: ClassVar[str] = "mGBA"
    script_path: ClassVar[Path] = SCRIPT_PATH
    open_hint: ClassVar[str] = MGBA_HINT


__all__ = ["MGBA_HINT", "SCRIPT_PATH", "MgbaAdapter"]
