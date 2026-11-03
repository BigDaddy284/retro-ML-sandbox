"""Saved AIs on disk: a folder per game, a JSON file per saved AI.

Like the brain store this works on the plain dictionary a ``SavedAi``
produces, so the storage layer needs nothing from the layers above it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.storage.jsonfiles import ensure_folder, read_json, safe_file_name, write_json


class SavedAiStore:
    def __init__(self, folder: str | Path) -> None:
        self._folder = ensure_folder(folder, "saved AIs")

    @property
    def folder(self) -> Path:
        return self._folder

    def folder_for(self, game_id: str) -> Path:
        return self._folder / safe_file_name(game_id, "")

    def path_for(self, game_id: str, name: str) -> Path:
        return self.folder_for(game_id) / safe_file_name(name)

    def games(self) -> tuple[str, ...]:
        """Game ids with at least one saved AI, sorted."""
        found = set()
        for path in self._folder.glob("*/*.json"):
            data = _peek(path)
            if data is not None:
                found.add(str(data["game_id"]))
        return tuple(sorted(found))

    def names(self, game_id: str) -> tuple[str, ...]:
        """The saved AIs of a game, alphabetically (case-insensitive)."""
        folder = self.folder_for(game_id)
        if not folder.is_dir():
            return ()
        names = []
        for path in folder.glob("*.json"):
            data = _peek(path)
            if data is not None and data["game_id"] == game_id:
                names.append(str(data["name"]))
        return tuple(sorted(names, key=str.casefold))

    def exists(self, game_id: str, name: str) -> bool:
        data = _peek(self.path_for(game_id, name))
        return data is not None and data["name"] == name

    def save(self, data: dict[str, Any]) -> Path:
        """Write a saved AI's dictionary (it names itself and its game)."""
        if _header(data) is None:
            raise ConfigurationError("Only a saved AI's own data can be stored here.")
        name, game_id = str(data["name"]), str(data["game_id"])
        path = self.path_for(game_id, name)
        current = _peek(path)
        if current is not None and current["name"] != name:
            raise ConfigurationError(
                f"The name {name!r} is too close to the saved AI {current['name']!r}. "
                "Pick a name that differs in more than punctuation."
            )
        ensure_folder(path.parent, "saved AIs")
        return write_json(path, data, "saved AI")

    def load(self, game_id: str, name: str) -> dict[str, Any]:
        path = self.path_for(game_id, name)
        if not self.exists(game_id, name):
            raise ConfigurationError(f"No saved AI called {name!r} for {game_id!r}.")
        return read_json(path, "saved AI")

    def delete(self, game_id: str, name: str) -> bool:
        if not self.exists(game_id, name):
            return False
        self.path_for(game_id, name).unlink()
        return True


def _peek(path: Path) -> dict[str, Any] | None:
    """The file's data if it is one of ours, else None (foreign files are left alone)."""
    try:
        return _header(json.loads(path.read_text("utf-8")))
    except (OSError, ValueError):
        return None


def _header(data: object) -> dict[str, Any] | None:
    if (
        isinstance(data, dict)
        and data.get("kind") == "saved_ai"
        and isinstance(data.get("name"), str)
        and isinstance(data.get("game_id"), str)
    ):
        return data
    return None


__all__ = ["SavedAiStore"]
