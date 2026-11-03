"""The brain: everything the AI has learned about one game, ready to save.

A brain is the agent (its Q-table, settings, exploration rate and random
state) plus the story of its goes, so that the next session carries on where
this one stopped (DESIGN.md Phase 5 "training history").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.qlearning import QLearningAgent
from retro_ai_lab.training.records import EpisodeRecord

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Brain:
    game_id: str
    agent: QLearningAgent
    goes: tuple[EpisodeRecord, ...] = ()
    episodes: int = 0  # all goes ever, even those no longer in ``goes``
    total_steps: int = 0

    def __post_init__(self) -> None:
        if not self.game_id or not isinstance(self.game_id, str):
            raise ConfigurationError("A brain needs a game id.")
        if self.episodes < len(self.goes) or self.total_steps < 0:
            raise ConfigurationError("A brain's counts can't be smaller than its goes.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "game_id": self.game_id,
            "agent": self.agent.to_dict(),
            "goes": [g.to_dict() for g in self.goes],
            "episodes": self.episodes,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, space: ActionSpace | None = None) -> Brain:
        """Rebuild a brain; ``space`` (the connected game's controls) must match."""
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(f"This brain is from a newer version (schema {schema}).")
            agent = QLearningAgent.from_dict(data["agent"], space=space)
            goes = tuple(EpisodeRecord.from_dict(g) for g in data.get("goes", []))
            return cls(
                str(data["game_id"]),
                agent,
                goes,
                int(data.get("episodes", len(goes))),
                int(data.get("total_steps", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad brain data: {exc}") from exc


__all__ = ["SCHEMA_VERSION", "Brain"]
