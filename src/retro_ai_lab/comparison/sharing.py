"""Saved AIs as files a child can hand to a friend (DESIGN.md §13).

A shared file is data, never code, and it is checked at every layer before
anything keeps it: size, strict JSON, the saved-AI envelope, and every part
rebuilt through its own ``from_dict`` so a damaged brain or reward profile
is caught at import time rather than when the AI is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retro_ai_lab import __version__
from retro_ai_lab.comparison.saved_ai import SavedAi
from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards.goals import Goal
from retro_ai_lab.rewards.model import RewardProfile
from retro_ai_lab.training.brain import Brain

MAX_SHARE_BYTES = 32 * 1024 * 1024  # a big network brain is a few MB; this is generous
MAX_DEPTH = 32  # saved AIs nest about six deep; anything deeper is not one of ours
SUFFIX = ".json"


def export_saved_ai(saved: SavedAi, path: str | Path) -> Path:
    """Write ``saved`` to ``path`` as a shareable JSON file."""
    target = Path(path)
    data = {**saved.to_dict(), "app_version": __version__}
    try:
        target.write_text(json.dumps(data, indent=2, sort_keys=True), "utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Can't write the shared AI to {target}: {exc}") from exc
    return target


def import_saved_ai(path: str | Path, *, max_bytes: int = MAX_SHARE_BYTES) -> SavedAi:
    """Read a shared file back, refusing anything that is not a sound saved AI."""
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ConfigurationError(f"Can't read the shared AI from {source}: {exc}") from exc
    if size > max_bytes:
        raise ConfigurationError(
            f"{source.name} is too big to be a shared AI ({size // (1024 * 1024)} MB)."
        )
    try:
        text = source.read_text("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigurationError(f"Can't read the shared AI from {source}: {exc}") from exc
    try:
        data = json.loads(text, parse_constant=_refuse_constant)
    except ValueError as exc:
        raise ConfigurationError(f"{source.name} is not a shared AI file: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{source.name} is not a shared AI file.")
    if _depth(data) > MAX_DEPTH:
        raise ConfigurationError(f"{source.name} is nested far deeper than a shared AI.")
    if data.get("kind") != "saved_ai":
        raise ConfigurationError(f"{source.name} is not a shared AI file.")
    saved = SavedAi.from_dict(data)
    check_parts(saved)
    return saved


def check_parts(saved: SavedAi) -> None:
    """Rebuild every part so a damaged one is refused now, in plain words."""
    try:
        RewardProfile.from_dict(saved.rewards)
    except ConfigurationError as exc:
        raise ConfigurationError(f"The shared AI's rewards can't be used: {exc}") from exc
    try:
        Goal.from_dict(saved.goal)
    except ConfigurationError as exc:
        raise ConfigurationError(f"The shared AI's goal can't be used: {exc}") from exc
    try:
        Brain.from_dict(saved.brain)
    except ConfigurationError as exc:
        raise ConfigurationError(f"The shared AI's brain can't be used: {exc}") from exc


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed in a shared AI")


def _depth(value: object, level: int = 1) -> int:
    if level > MAX_DEPTH:
        return level
    if isinstance(value, dict):
        return max((_depth(v, level + 1) for v in value.values()), default=level)
    if isinstance(value, list):
        return max((_depth(v, level + 1) for v in value), default=level)
    return level


__all__ = [
    "MAX_DEPTH",
    "MAX_SHARE_BYTES",
    "SUFFIX",
    "check_parts",
    "export_saved_ai",
    "import_saved_ai",
]
