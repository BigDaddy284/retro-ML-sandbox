"""Two or more saved AIs side by side (DESIGN.md §11).

A scorecard is read straight out of a saved AI's goes; behaviour differences
come from rebuilding the two learners and asking each for its best move in
every situation both of them know. Everything here is a pure function, so
the panel only has to show the results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from retro_ai_lab.comparison.saved_ai import SavedAi
from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.learner import Learner, learner_from_dict
from retro_ai_lab.learning.state import State
from retro_ai_lab.training.curve import LearningCurve
from retro_ai_lab.training.records import EpisodeRecord
from retro_ai_lab.understanding.knowledge import Knowledge


@dataclass(frozen=True, slots=True)
class Measure:
    key: str
    label: str  # how the panel heads the column
    higher_is_better: bool
    sentence: str  # "scored more points a go", used by describe()


MEASURES: tuple[Measure, ...] = (
    Measure("average_reward", "Points a go", True, "scored more points a go"),
    Measure("best_reward", "Best go", True, "had the best single go"),
    Measure("objectives", "Goals reached", True, "completed more goal steps"),
    Measure("successes", "Winning goes", True, "reached a goal in more goes"),
    Measure("average_steps", "Steps a go", False, "finished its goes in fewer steps"),
    Measure("attempts", "Goes", True, "has practised more"),
)


@dataclass(frozen=True, slots=True)
class Scorecard:
    name: str
    personality: str
    algorithm: str
    attempts: int
    average_reward: float | None
    best_reward: float | None
    objectives: int
    successes: int
    success_rate: float
    average_steps: float | None
    average_frames: float | None
    situations: int

    @classmethod
    def from_saved(cls, saved: SavedAi) -> Scorecard:
        goes = _goes_of(saved)
        curve = LearningCurve(goes)
        best = curve.best
        n = len(goes)
        return cls(
            saved.name,
            saved.personality,
            saved.algorithm,
            curve.attempts,
            (sum(curve.rewards) / n) if n else None,
            best.reward if best is not None else None,
            sum(g.result.objectives for g in goes),
            curve.successes,
            curve.success_rate,
            (sum(g.result.steps for g in goes) / n) if n else None,
            (sum(g.result.frames for g in goes) / n) if n else None,
            _situations(saved),
        )

    def value(self, measure: str) -> float | None:
        if measure not in {m.key for m in MEASURES}:
            raise ConfigurationError(f"There is no measure called {measure!r}.")
        raw = getattr(self, measure)
        return float(raw) if raw is not None else None


@dataclass(frozen=True, slots=True)
class BehaviourDifference:
    state_key: str
    a_move: int
    b_move: int


class AiComparison:
    def __init__(self, cards: Sequence[Scorecard]) -> None:
        names = [c.name for c in cards]
        if len(set(names)) != len(names):
            raise ConfigurationError("Each AI in a comparison needs its own name.")
        self._cards = tuple(cards)

    @classmethod
    def of(cls, saved: Sequence[SavedAi]) -> AiComparison:
        return cls([Scorecard.from_saved(s) for s in saved])

    @property
    def cards(self) -> tuple[Scorecard, ...]:
        return self._cards

    @staticmethod
    def measures() -> tuple[Measure, ...]:
        return MEASURES

    def leader(self, measure: str) -> Scorecard | None:
        """The card that does best on ``measure``; None on a tie or with one card."""
        scored: list[tuple[float, Scorecard]] = []
        for card in self._cards:
            value = card.value(measure)
            if value is not None:
                scored.append((value, card))
        if len(scored) < 2:
            return None
        higher = next(m.higher_is_better for m in MEASURES if m.key == measure)
        best = max(v for v, _ in scored) if higher else min(v for v, _ in scored)
        winners = [c for v, c in scored if v == best]
        return winners[0] if len(winners) == 1 else None

    def leaders(self) -> dict[str, Scorecard | None]:
        return {m.key: self.leader(m.key) for m in MEASURES}


def learner_of(saved: SavedAi) -> Learner:
    agent = saved.brain.get("agent")
    if not isinstance(agent, dict):
        raise ConfigurationError(f"The saved AI {saved.name!r} has no brain to compare.")
    return learner_from_dict(agent)


def behaviour_differences(a: SavedAi, b: SavedAi) -> tuple[BehaviourDifference, ...]:
    """Situations both AIs know where they would press different buttons."""
    first, second = learner_of(a), learner_of(b)
    if first.space != second.space:
        raise ConfigurationError(
            f"{a.name} and {b.name} learned with different controls, so their moves "
            "can't be compared."
        )
    mine, theirs = first.policy(), second.policy()
    return tuple(
        BehaviourDifference(key, mine[key], theirs[key])
        for key in sorted(mine)
        if key in theirs and mine[key] != theirs[key]
    )


def describe(
    comparison: AiComparison,
    *,
    differences: Sequence[BehaviourDifference] = (),
    space: ActionSpace | None = None,
    knowledge: Knowledge | None = None,
) -> str:
    """The comparison in a child's words."""
    cards = comparison.cards
    if not cards:
        return "Save two AIs for this game and I'll compare them."
    if len(cards) == 1:
        c = cards[0]
        return f"Only {c.name} is saved so far. Save another AI to compare them."
    played = [c for c in cards if c.attempts]
    if not played:
        return _names(cards) + " haven't had any goes yet, so there is nothing to compare."
    sentences: list[str] = []
    rested = [c for c in cards if not c.attempts]
    if rested:
        sentences.append(_names(rested) + " " + _have(rested) + " not had a go yet.")
    for measure in MEASURES:
        if measure.key == "attempts":
            continue
        leader = comparison.leader(measure.key)
        if leader is None:
            continue
        others = [c for c in played if c.name != leader.name]
        theirs = ", ".join(f"{c.name}'s {_fmt(measure.key, c)}" for c in others)
        sentences.append(
            f"{leader.name} {measure.sentence} ({_fmt(measure.key, leader)} against {theirs})."
        )
    if len(sentences) == (1 if rested else 0):
        sentences.append(_names(played) + " did about as well as each other.")
    if differences and len(cards) == 2:
        n = len(differences)
        a, b = cards
        example = differences[0]
        where = State.from_key(example.state_key).describe(knowledge)
        moves = (
            f"{a.name} presses {_move(space, example.a_move)} and "
            f"{b.name} presses {_move(space, example.b_move)}"
        )
        sentences.append(
            f"In {n} situation{'s' if n != 1 else ''} they would press different buttons: "
            f"where {where} {moves}."
        )
    elif len(cards) == 2 and all(c.situations for c in cards):
        sentences.append("In every situation they both know, they would press the same buttons.")
    return " ".join(sentences)


# ------------------------------------------------------------------ helpers


def _goes_of(saved: SavedAi) -> tuple[EpisodeRecord, ...]:
    raw = saved.brain.get("goes", [])
    if not isinstance(raw, list):
        raise ConfigurationError(f"The saved AI {saved.name!r} has damaged goes.")
    return tuple(EpisodeRecord.from_dict(g) for g in raw)


def _situations(saved: SavedAi) -> int:
    agent = saved.brain.get("agent")
    if not isinstance(agent, dict):
        return 0
    try:
        return len(learner_from_dict(agent).policy())
    except ConfigurationError:
        return 0


def _names(cards: Sequence[Scorecard]) -> str:
    names = [c.name for c in cards]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _have(cards: Sequence[Any]) -> str:
    return "has" if len(cards) == 1 else "have"


def _fmt(measure: str, card: Scorecard) -> str:
    value = card.value(measure)
    if value is None:
        return "no goes"
    if measure in ("average_reward", "best_reward"):
        return f"{value:+.1f}"
    if measure == "successes":
        return f"{card.successes} of {card.attempts}"
    if measure == "average_steps":
        return f"{value:.0f}"
    return f"{value:g}"


def _move(space: ActionSpace | None, index: int) -> str:
    if space is None:
        return f"move {index}"
    try:
        return space[index].label
    except (IndexError, ConfigurationError):
        return f"move {index}"


__all__ = [
    "MEASURES",
    "AiComparison",
    "BehaviourDifference",
    "Measure",
    "Scorecard",
    "behaviour_differences",
    "describe",
    "learner_of",
]
