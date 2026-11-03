"""Small helpers shared by the JSON file stores."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from retro_ai_lab.core import ConfigurationError

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_file_name(game_id: str, suffix: str = ".json") -> str:
    cleaned = _SAFE.sub("_", game_id).strip("._")
    if not cleaned:
        raise ConfigurationError(f"Game id {game_id!r} can't be turned into a file name.")
    return cleaned + suffix


def ensure_folder(folder: str | Path, what: str) -> Path:
    path = Path(folder)
    if path.exists() and not path.is_dir():
        raise ConfigurationError(f"{path} is not a folder.")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(f"Can't create {what} folder {path}: {exc}") from exc
    return path


def write_json(path: str | Path, data: dict[str, Any], what: str) -> Path:
    """Write readable JSON atomically (a crash mid-write never leaves a half file)."""
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), "utf-8")
        tmp.replace(target)
    except OSError as exc:
        raise ConfigurationError(f"Can't write {what} to {target}: {exc}") from exc
    return target


def read_json(path: str | Path, what: str) -> dict[str, Any]:
    source = Path(path)
    try:
        data = json.loads(source.read_text("utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Can't read {what} from {source}: {exc}") from exc
    except ValueError as exc:
        raise ConfigurationError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{source} does not hold a {what} object.")
    return data


def game_ids_in(folder: Path) -> tuple[str, ...]:
    """Game ids of the JSON files in a folder, sorted; foreign files are skipped."""
    ids = []
    for p in sorted(folder.glob("*.json")):
        try:
            ids.append(str(json.loads(p.read_text("utf-8"))["game_id"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue  # not one of ours; leave it alone
    return tuple(ids)


__all__ = ["ensure_folder", "game_ids_in", "read_json", "safe_file_name", "write_json"]
