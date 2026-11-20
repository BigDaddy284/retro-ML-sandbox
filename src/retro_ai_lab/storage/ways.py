"""Save and load the way the AI remembers, as a JSON file per game (T-1064).

A way is the moves that begin each go (``training/route.py``). It is only good
together with the save state it starts from, which stays with the emulator on
this computer, so it is kept beside the brain and not in it: a brain is also
what "Save this AI" copies, and a copy has no use for a way.

The file holds the plain dictionary the training service makes; this module
needs nothing from above it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.storage.jsonfiles import (
    ensure_folder,
    game_ids_in,
    read_json,
    safe_file_name,
    write_json,
)


class WayStore:
    def __init__(self, folder: str | Path) -> None:
        self._folder = ensure_folder(folder, "ways")

    @property
    def folder(self) -> Path:
        return self._folder

    def path_for(self, game_id: str) -> Path:
        return self._folder / safe_file_name(game_id)

    def games(self) -> tuple[str, ...]:
        """Game ids with a way written down, sorted."""
        return game_ids_in(self._folder)

    def exists(self, game_id: str) -> bool:
        return self.path_for(game_id).exists()

    def save(self, data: dict[str, Any]) -> Path:
        """Write a way's dictionary (it names its own game)."""
        game_id = data.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            raise ConfigurationError("A way to save must name its game.")
        return write_json(self.path_for(game_id), data, "way")

    def load(self, game_id: str) -> dict[str, Any]:
        path = self.path_for(game_id)
        if not path.exists():
            raise ConfigurationError(f"No way written down for {game_id!r}.")
        return read_json(path, "way")

    def delete(self, game_id: str) -> bool:
        path = self.path_for(game_id)
        if not path.exists():
            return False
        path.unlink()
        return True


__all__ = ["WayStore"]
