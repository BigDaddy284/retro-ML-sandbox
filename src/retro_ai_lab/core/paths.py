"""Where the app keeps its own files (knowledge, archives, settings).

``RETRO_AI_LAB_DATA_DIR`` overrides everything (tests and portable installs);
otherwise the per-user application data folder of the platform is used.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_VAR = "RETRO_AI_LAB_DATA_DIR"
APP_FOLDER = "RetroAILab"


def default_data_dir() -> Path:
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / APP_FOLDER
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_FOLDER
    xdg = os.environ.get("XDG_DATA_HOME")
    root = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return root / APP_FOLDER.lower()


__all__ = ["APP_FOLDER", "ENV_VAR", "default_data_dir"]
