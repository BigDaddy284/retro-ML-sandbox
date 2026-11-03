"""Save, load and export :class:`Knowledge` as JSON files.

One file per game, named after the game id, in a folder the app owns. The
format is the plain ``to_dict`` output so a curious person can open it in a
text editor, and ``export`` writes the same thing anywhere for sharing.
"""

from __future__ import annotations

from pathlib import Path

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.storage.jsonfiles import (
    ensure_folder,
    game_ids_in,
    read_json,
    safe_file_name,
    write_json,
)
from retro_ai_lab.understanding.knowledge import Knowledge


class KnowledgeStore:
    def __init__(self, folder: str | Path) -> None:
        self._folder = ensure_folder(folder, "knowledge")

    @property
    def folder(self) -> Path:
        return self._folder

    def path_for(self, game_id: str) -> Path:
        return self._folder / safe_file_name(game_id)

    def games(self) -> tuple[str, ...]:
        """Game ids with saved knowledge, sorted."""
        return game_ids_in(self._folder)

    def exists(self, game_id: str) -> bool:
        return self.path_for(game_id).exists()

    def save(self, knowledge: Knowledge) -> Path:
        path = self.path_for(knowledge.game_id)
        export(knowledge, path)
        return path

    def load(self, game_id: str) -> Knowledge:
        path = self.path_for(game_id)
        if not path.exists():
            raise ConfigurationError(f"No saved knowledge for {game_id!r}.")
        return load(path)

    def load_or_new(self, game_id: str, *, title: str = "") -> Knowledge:
        if self.exists(game_id):
            return self.load(game_id)
        return Knowledge(game_id, title=title)

    def delete(self, game_id: str) -> bool:
        path = self.path_for(game_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def export(knowledge: Knowledge, path: str | Path) -> Path:
    """Write knowledge as readable JSON (atomically) to any path."""
    return write_json(path, knowledge.to_dict(), "knowledge")


def load(path: str | Path) -> Knowledge:
    return Knowledge.from_dict(read_json(path, "knowledge"))


__all__ = ["KnowledgeStore", "export", "load"]
