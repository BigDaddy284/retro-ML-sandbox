"""Save and load the AI's brain as a JSON file per game.

The file holds the plain dictionary :meth:`Brain.to_dict` produces; the
training layer turns it back into a :class:`Brain`, so this module needs
nothing from above it.
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


class BrainStore:
    def __init__(self, folder: str | Path) -> None:
        self._folder = ensure_folder(folder, "brains")

    @property
    def folder(self) -> Path:
        return self._folder

    def path_for(self, game_id: str) -> Path:
        return self._folder / safe_file_name(game_id)

    def games(self) -> tuple[str, ...]:
        """Game ids with a saved brain, sorted."""
        return game_ids_in(self._folder)

    def exists(self, game_id: str) -> bool:
        return self.path_for(game_id).exists()

    def save(self, data: dict[str, Any]) -> Path:
        """Write a brain's dictionary (it names its own game)."""
        game_id = data.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            raise ConfigurationError("A brain to save must name its game.")
        return export_brain(data, self.path_for(game_id))

    def load(self, game_id: str) -> dict[str, Any]:
        path = self.path_for(game_id)
        if not path.exists():
            raise ConfigurationError(f"No saved brain for {game_id!r}.")
        return load_brain(path)

    def delete(self, game_id: str) -> bool:
        path = self.path_for(game_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def export_brain(data: dict[str, Any], path: str | Path) -> Path:
    return write_json(path, data, "brain")


def load_brain(path: str | Path) -> dict[str, Any]:
    return read_json(path, "brain")


__all__ = ["BrainStore", "export_brain", "load_brain"]
