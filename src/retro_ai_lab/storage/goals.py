"""Save, load and export :class:`Goal` as JSON files, one per game."""

from __future__ import annotations

from pathlib import Path

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards.goals import Goal
from retro_ai_lab.storage.jsonfiles import (
    ensure_folder,
    game_ids_in,
    read_json,
    safe_file_name,
    write_json,
)


class GoalStore:
    def __init__(self, folder: str | Path) -> None:
        self._folder = ensure_folder(folder, "goals")

    @property
    def folder(self) -> Path:
        return self._folder

    def path_for(self, game_id: str) -> Path:
        return self._folder / safe_file_name(game_id)

    def games(self) -> tuple[str, ...]:
        return game_ids_in(self._folder)

    def exists(self, game_id: str) -> bool:
        return self.path_for(game_id).exists()

    def save(self, goal: Goal) -> Path:
        return export_goal(goal, self.path_for(goal.game_id))

    def load(self, game_id: str) -> Goal:
        path = self.path_for(game_id)
        if not path.exists():
            raise ConfigurationError(f"No saved goal for {game_id!r}.")
        return load_goal(path)

    def load_or_new(self, game_id: str, *, title: str = "") -> Goal:
        if self.exists(game_id):
            return self.load(game_id)
        return Goal(game_id, title=title)

    def delete(self, game_id: str) -> bool:
        path = self.path_for(game_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def export_goal(goal: Goal, path: str | Path) -> Path:
    return write_json(path, goal.to_dict(), "goal")


def load_goal(path: str | Path) -> Goal:
    return Goal.from_dict(read_json(path, "goal"))


__all__ = ["GoalStore", "export_goal", "load_goal"]
