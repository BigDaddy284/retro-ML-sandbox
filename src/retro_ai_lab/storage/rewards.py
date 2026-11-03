"""Save, load and export :class:`RewardProfile` as JSON files, one per game."""

from __future__ import annotations

from pathlib import Path

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards.model import RewardProfile
from retro_ai_lab.storage.jsonfiles import (
    ensure_folder,
    game_ids_in,
    read_json,
    safe_file_name,
    write_json,
)


class RewardStore:
    def __init__(self, folder: str | Path) -> None:
        self._folder = ensure_folder(folder, "rewards")

    @property
    def folder(self) -> Path:
        return self._folder

    def path_for(self, game_id: str) -> Path:
        return self._folder / safe_file_name(game_id)

    def games(self) -> tuple[str, ...]:
        """Game ids with a saved reward profile, sorted."""
        return game_ids_in(self._folder)

    def exists(self, game_id: str) -> bool:
        return self.path_for(game_id).exists()

    def save(self, profile: RewardProfile) -> Path:
        return export_profile(profile, self.path_for(profile.game_id))

    def load(self, game_id: str) -> RewardProfile:
        path = self.path_for(game_id)
        if not path.exists():
            raise ConfigurationError(f"No saved rewards for {game_id!r}.")
        return load_profile(path)

    def load_or_new(self, game_id: str) -> RewardProfile:
        if self.exists(game_id):
            return self.load(game_id)
        return RewardProfile(game_id)

    def delete(self, game_id: str) -> bool:
        path = self.path_for(game_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def export_profile(profile: RewardProfile, path: str | Path) -> Path:
    return write_json(path, profile.to_dict(), "rewards")


def load_profile(path: str | Path) -> RewardProfile:
    return RewardProfile.from_dict(read_json(path, "rewards"))


__all__ = ["RewardStore", "export_profile", "load_profile"]
