"""AI personalities: starting reward configurations (DESIGN.md §5.3).

A personality is not a different AI. It is a set of weights, one per
category, that scale the suggested values: an Explorer cares more about
finding new places, a Survivor minds losing lives more. The weights live
on the profile so later suggestions keep the personality in effect, and a
child can still change every single value afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards.model import Category, RewardProfile


def _weights(**overrides: float) -> dict[Category, float]:
    out = {c: 1.0 for c in Category}
    for name, value in overrides.items():
        out[Category[name.upper()]] = value
    return out


@dataclass(frozen=True, slots=True)
class Personality:
    id: str
    name: str
    description: str  # one child-friendly sentence
    weights: dict[Category, float] = field(default_factory=lambda: _weights())

    def __post_init__(self) -> None:
        if not self.id or not self.name:
            raise ConfigurationError("Personality needs an id and a name.")
        object.__setattr__(self, "weights", RewardProfile.check_weights(self.weights))


PERSONALITIES: tuple[Personality, ...] = (
    Personality(
        "explorer",
        "Explorer",
        "Loves finding new places and trying new things.",
        _weights(exploration=2.0, progression=1.25, collection=0.75),
    ),
    Personality(
        "collector",
        "Collector",
        "Wants every coin, item and point there is.",
        _weights(collection=2.0, completion=1.25, efficiency=0.75),
    ),
    Personality(
        "completionist",
        "Completionist",
        "Will not rest until everything is done and seen.",
        _weights(completion=2.0, quests=1.5, collection=1.5, exploration=1.25, efficiency=0.5),
    ),
    Personality(
        "speedrunner",
        "Speedrunner",
        "Gets to the end as fast as it can and skips the rest.",
        _weights(efficiency=2.0, progression=1.5, collection=0.5, exploration=0.5),
    ),
    Personality(
        "battler",
        "Battler",
        "Looks for a fight and fights to win.",
        _weights(combat=2.0, survival=1.25, exploration=0.75),
    ),
    Personality(
        "survivor",
        "Survivor",
        "Stays alive above all else and takes no silly risks.",
        _weights(survival=2.0, combat=0.75, efficiency=0.75),
    ),
    Personality(
        "balanced",
        "Balanced Player",
        "A bit of everything, nothing more than the rest.",
        _weights(),
    ),
)

_BY_ID = {p.id: p for p in PERSONALITIES}


def personality(personality_id: str) -> Personality:
    try:
        return _BY_ID[personality_id.strip().lower()]
    except (KeyError, AttributeError):
        names = ", ".join(_BY_ID)
        raise ConfigurationError(
            f"No personality {personality_id!r}. Choose one of: {names}."
        ) from None


def apply_personality(profile: RewardProfile, which: str | Personality) -> Personality:
    """Switch the profile to a personality's weights; values a person set are kept."""
    chosen = which if isinstance(which, Personality) else personality(which)
    profile.set_weights(chosen.weights)
    profile.personality = chosen.id
    return chosen


__all__ = ["PERSONALITIES", "Personality", "apply_personality", "personality"]
