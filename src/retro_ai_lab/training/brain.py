"""The brain: everything the AI has learned about one game, ready to save.

A brain is the learner (whatever it has learned, its settings, exploration
rate and random state) plus the story of its goes, so that the next session carries on where
this one stopped (DESIGN.md Phase 5 "training history").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.demonstrations import Demonstrations
from retro_ai_lab.learning.learner import Learner, learner_from_dict
from retro_ai_lab.training.records import EpisodeRecord

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Brain:
    game_id: str
    agent: Learner
    goes: tuple[EpisodeRecord, ...] = ()
    episodes: int = 0  # all goes ever, even those no longer in ``goes``
    total_steps: int = 0
    demonstrations: Demonstrations | None = None  # the child's moves, if any were shown
    # The go settings a person chose for this game (T-873); None means the app's default.
    go_length: int | None = None  # steps a go lasts at most
    give_up_after: int | None = None  # steps with nothing changing before a go gives up
    closer: tuple[tuple[str, int], ...] = ()  # sights read in fine steps (T-981)

    def __post_init__(self) -> None:
        if not self.game_id or not isinstance(self.game_id, str):
            raise ConfigurationError("A brain needs a game id.")
        if self.episodes < len(self.goes) or self.total_steps < 0:
            raise ConfigurationError("A brain's counts can't be smaller than its goes.")
        for name in ("go_length", "give_up_after"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ConfigurationError(f"A brain's {name} must be a whole number of steps.")

    @property
    def algorithm(self) -> str:
        return self.agent.algorithm

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema": SCHEMA_VERSION,
            "game_id": self.game_id,
            "agent": self.agent.to_dict(),
            "goes": [g.to_dict() for g in self.goes],
            "episodes": self.episodes,
            "total_steps": self.total_steps,
        }
        if self.demonstrations is not None:
            data["demonstrations"] = self.demonstrations.to_dict()
        if self.go_length is not None:
            data["go_length"] = self.go_length
        if self.give_up_after is not None:
            data["give_up_after"] = self.give_up_after
        if self.closer:
            data["closer"] = [[vid, b] for vid, b in self.closer]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, space: ActionSpace | None = None) -> Brain:
        """Rebuild a brain; ``space`` (the connected game's controls) must match."""
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(f"This brain is from a newer version (schema {schema}).")
            agent = learner_from_dict(data["agent"], space=space)
            goes = tuple(EpisodeRecord.from_dict(g) for g in data.get("goes", []))
            demos = data.get("demonstrations")
            return cls(
                str(data["game_id"]),
                agent,
                goes,
                int(data.get("episodes", len(goes))),
                int(data.get("total_steps", 0)),
                Demonstrations.from_dict(demos) if demos is not None else None,
                _steps_or_none(data.get("go_length")),
                _steps_or_none(data.get("give_up_after")),
                tuple((str(vid), int(b)) for vid, b in data.get("closer", [])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad brain data: {exc}") from exc


def _steps_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = ["SCHEMA_VERSION", "Brain"]
