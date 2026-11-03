"""A saved AI: a name, and the rewards, goal and brain that make it what it is.

DESIGN.md §11 calls this an AI configuration ("AI A prioritises exploration,
AI B prioritises combat"). The three parts are kept as the plain dictionaries
their own ``to_dict`` methods produce, so this module depends on nothing above
the storage layer and a file can be checked without rebuilding a brain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from retro_ai_lab.core import ConfigurationError

SCHEMA_VERSION = 1
MAX_NAME_LENGTH = 40
PARTS = ("rewards", "goal", "brain")


def check_name(name: object) -> str:
    """The name trimmed, or a ConfigurationError saying what is wrong with it."""
    if not isinstance(name, str):
        raise ConfigurationError("An AI's name must be text.")
    cleaned = name.strip()
    if not cleaned:
        raise ConfigurationError("Give the AI a name.")
    if len(cleaned) > MAX_NAME_LENGTH:
        raise ConfigurationError(f"Keep the AI's name to {MAX_NAME_LENGTH} characters.")
    if not cleaned.isprintable():
        raise ConfigurationError("The AI's name has characters that can't be shown.")
    return cleaned


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SavedAi:
    name: str
    game_id: str
    rewards: dict[str, Any]
    goal: dict[str, Any]
    brain: dict[str, Any]
    created: str = field(default_factory=now_iso)
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", check_name(self.name))
        if not isinstance(self.game_id, str) or not self.game_id:
            raise ConfigurationError("A saved AI must name its game.")
        if not isinstance(self.created, str) or not self.created:
            raise ConfigurationError("A saved AI needs the time it was saved.")
        if not isinstance(self.notes, str):
            raise ConfigurationError("Notes must be text.")
        for part in PARTS:
            data = getattr(self, part)
            if not isinstance(data, dict):
                raise ConfigurationError(f"The saved AI's {part} must be an object.")
            owner = data.get("game_id")
            if owner is not None and owner != self.game_id:
                raise ConfigurationError(
                    f"The saved AI's {part} belong to {owner!r}, not {self.game_id!r}."
                )

    # ----------------------------------------------------------- summaries

    @property
    def episodes(self) -> int:
        return _int_in(self.brain, "episodes")

    @property
    def total_steps(self) -> int:
        return _int_in(self.brain, "total_steps")

    @property
    def algorithm(self) -> str:
        agent = self.brain.get("agent")
        if isinstance(agent, dict) and isinstance(agent.get("algorithm"), str):
            return str(agent["algorithm"])
        return ""

    @property
    def personality(self) -> str:
        value = self.rewards.get("personality")
        return value if isinstance(value, str) else ""

    # --------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "kind": "saved_ai",
            "name": self.name,
            "game_id": self.game_id,
            "created": self.created,
            "notes": self.notes,
            "rewards": self.rewards,
            "goal": self.goal,
            "brain": self.brain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SavedAi:
        if not isinstance(data, dict):
            raise ConfigurationError("A saved AI must be an object.")
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("The saved AI's schema number is not a number.") from exc
        if schema > SCHEMA_VERSION:
            raise ConfigurationError(f"This saved AI is from a newer version (schema {schema}).")
        missing = [part for part in PARTS if part not in data]
        if missing:
            raise ConfigurationError(f"The saved AI is missing its {', '.join(missing)}.")
        if "game_id" not in data or "name" not in data:
            raise ConfigurationError("The saved AI needs a name and a game.")
        return cls(
            data["name"],
            data["game_id"],
            data["rewards"],
            data["goal"],
            data["brain"],
            created=data.get("created", now_iso()),
            notes=data.get("notes", ""),
        )


def _int_in(data: dict[str, Any], key: str) -> int:
    value = data.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


__all__ = ["MAX_NAME_LENGTH", "PARTS", "SCHEMA_VERSION", "SavedAi", "check_name", "now_iso"]
