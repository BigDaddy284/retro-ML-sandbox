"""Saved AIs racing under one set of conditions (DESIGN.md §13)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training.episode import (
    DEFAULT_FRAMES_PER_STEP,
    DEFAULT_MAX_STEPS,
    EpisodeResult,
)

DEFAULT_GOES = 3
MAX_GOES = 50


@dataclass(frozen=True, slots=True)
class Challenge:
    """The conditions every entrant races under."""

    goes: int = DEFAULT_GOES
    max_steps: int = DEFAULT_MAX_STEPS
    frames_per_step: int = DEFAULT_FRAMES_PER_STEP

    def __post_init__(self) -> None:
        for name, value, top in (
            ("goes", self.goes, MAX_GOES),
            ("max_steps", self.max_steps, 100_000),
            ("frames_per_step", self.frames_per_step, 600),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= top:
                raise ConfigurationError(f"{name} must be a whole number from 1 to {top}.")

    def describe(self) -> str:
        goes = "go" if self.goes == 1 else "goes"
        return (
            f"{self.goes} {goes} each from the same starting point, "
            f"up to {self.max_steps} steps a go, with the rewards and goal set now."
        )


@dataclass(frozen=True, slots=True)
class Entry:
    name: str
    results: tuple[EpisodeResult, ...]

    @property
    def goes(self) -> int:
        return len(self.results)

    @property
    def total_reward(self) -> float:
        return sum(r.reward for r in self.results)

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.goes if self.goes else 0.0

    @property
    def objectives(self) -> int:
        return sum(r.objectives for r in self.results)

    @property
    def average_steps(self) -> float:
        return sum(r.steps for r in self.results) / self.goes if self.goes else 0.0

    @property
    def successes(self) -> int:
        return sum(1 for r in self.results if r.succeeded)

    def sort_key(self) -> tuple[float, int, float]:
        """Higher is better: points a go, then goal steps, then fewer steps."""
        return (self.average_reward, self.objectives, -self.average_steps)


class Scoreboard:
    def __init__(self, challenge: Challenge, entries: Sequence[Entry]) -> None:
        names = [e.name for e in entries]
        if len(set(names)) != len(names):
            raise ConfigurationError("Each entrant needs its own name.")
        self._challenge = challenge
        self._entries = tuple(entries)

    @property
    def challenge(self) -> Challenge:
        return self._challenge

    @property
    def entries(self) -> tuple[Entry, ...]:
        return self._entries

    def ranked(self) -> tuple[Entry, ...]:
        return tuple(sorted(self._entries, key=lambda e: e.sort_key(), reverse=True))

    def winner(self) -> Entry | None:
        """The clear winner, or None on a tie for first (or no entrants)."""
        ranked = self.ranked()
        if not ranked:
            return None
        if len(ranked) > 1 and ranked[0].sort_key() == ranked[1].sort_key():
            return None
        return ranked[0]

    def describe(self) -> str:
        if not self._entries:
            return "Nobody raced."
        places = []
        for i, e in enumerate(self.ranked(), start=1):
            steps = "step" if e.objectives == 1 else "steps"
            places.append(
                f"{i}. {e.name}: {e.average_reward:+.1f} points a go, "
                f"{e.objectives} goal {steps}, {e.average_steps:.0f} steps a go."
            )
        winner = self.winner()
        verdict = f"{winner.name} wins!" if winner is not None else "It's a tie at the top!"
        return f"{verdict} " + " ".join(places) + f" ({self._challenge.describe()})"


@dataclass(frozen=True, slots=True)
class CompetitionFinished:
    game_id: str
    scoreboard: Scoreboard


__all__ = [
    "DEFAULT_GOES",
    "MAX_GOES",
    "Challenge",
    "CompetitionFinished",
    "Entry",
    "Scoreboard",
]
