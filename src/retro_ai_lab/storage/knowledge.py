"""Save, load and export :class:`Knowledge` as JSON files.

One file per game, named after the game id, in a folder the app owns. The
format is the plain ``to_dict`` output so a curious person can open it in a
text editor, and ``export`` writes the same thing anywhere for sharing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.understanding.knowledge import Knowledge

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _file_name(game_id: str) -> str:
    cleaned = _SAFE.sub("_", game_id).strip("._")
    if not cleaned:
        raise ConfigurationError(f"Game id {game_id!r} can't be turned into a file name.")
    return cleaned + ".json"


class KnowledgeStore:
    def __init__(self, folder: str | Path) -> None:
        self._folder = Path(folder)
        if self._folder.exists() and not self._folder.is_dir():
            raise ConfigurationError(f"{self._folder} is not a folder.")
        try:
            self._folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigurationError(
                f"Can't create knowledge folder {self._folder}: {exc}"
            ) from exc

    @property
    def folder(self) -> Path:
        return self._folder

    def path_for(self, game_id: str) -> Path:
        return self._folder / _file_name(game_id)

    def games(self) -> tuple[str, ...]:
        """Game ids with saved knowledge, sorted."""
        ids = []
        for p in sorted(self._folder.glob("*.json")):
            try:
                ids.append(str(json.loads(p.read_text("utf-8"))["game_id"]))
            except (OSError, ValueError, KeyError, TypeError):
                continue  # not one of ours; leave it alone
        return tuple(ids)

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
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(knowledge.to_dict(), indent=2, sort_keys=True), "utf-8")
        tmp.replace(target)
    except OSError as exc:
        raise ConfigurationError(f"Can't write knowledge to {target}: {exc}") from exc
    return target


def load(path: str | Path) -> Knowledge:
    source = Path(path)
    try:
        data = json.loads(source.read_text("utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Can't read knowledge from {source}: {exc}") from exc
    except ValueError as exc:
        raise ConfigurationError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{source} does not hold a knowledge object.")
    return Knowledge.from_dict(data)


__all__ = ["KnowledgeStore", "export", "load"]
