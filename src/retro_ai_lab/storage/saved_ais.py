"""Saved AIs on disk: a folder per game, a JSON file per saved AI."""

from __future__ import annotations

import json
from pathlib import Path

from retro_ai_lab.comparison.saved_ai import SavedAi
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

    def save(self, saved: SavedAi) -> Path:
        path = self.path_for(saved.game_id, saved.name)
        current = _peek(path)
        if current is not None and current["name"] != saved.name:
            raise ConfigurationError(
                f"The name {saved.name!r} is too close to the saved AI {current['name']!r}. "
                "Pick a name that differs in more than punctuation."
            )
        ensure_folder(path.parent, "saved AIs")
        return write_json(path, saved.to_dict(), "saved AI")

    def load(self, game_id: str, name: str) -> SavedAi:
        path = self.path_for(game_id, name)
        if not self.exists(game_id, name):
            raise ConfigurationError(f"No saved AI called {name!r} for {game_id!r}.")
        saved = SavedAi.from_dict(read_json(path, "saved AI"))
        if saved.game_id != game_id:
            raise ConfigurationError(f"{path} holds an AI for another game.")
        return saved

    def delete(self, game_id: str, name: str) -> bool:
        if not self.exists(game_id, name):
            return False
        self.path_for(game_id, name).unlink()
        return True


def _peek(path: Path) -> dict[str, object] | None:
    """The file's header if it is one of ours, else None (foreign files are left alone)."""
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if (
        isinstance(data, dict)
        and data.get("kind") == "saved_ai"
        and isinstance(data.get("name"), str)
        and isinstance(data.get("game_id"), str)
    ):
        return data
    return None


__all__ = ["SavedAiStore"]
