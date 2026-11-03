"""The contract every learning algorithm meets (DESIGN.md §6: "modular").

The training service, the saved brain, the insights and the panels talk to a
:class:`Learner` and never to a particular algorithm. A learner is registered
under a short ``algorithm`` name, which is what a saved brain records so the
right class rebuilds it next session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.state import State

Q_TABLE = "q_table"
AUTO = "auto"  # not a learner: the training service picks one and moves up as the game grows

_LABELS: dict[str, tuple[str, str]] = {
    AUTO: (
        "Choose for me",
        "Starts with the simple table and moves to a neural network when the game grows.",
    ),
    Q_TABLE: (
        "Simple table",
        "Keeps a score for every move in every situation it has seen. Best for small games.",
    ),
    "dqn": (
        "Neural network",
        "Guesses scores for situations it has never seen from ones like them. "
        "Better once the game gets big.",
    ),
    "skills": (
        "Skills per goal step",
        "Learns a separate skill for each step of the goal and uses the one for the step "
        "it is working on.",
    ),
}


@dataclass(frozen=True, slots=True)
class Decision:
    """How the last move was picked (for the UI to explain)."""

    action: int
    explored: bool  # True when it tried something at random
    value: float  # what the learner thought the move was worth


@runtime_checkable
class Learner(Protocol):
    """What the rest of the app needs from a learning algorithm."""

    algorithm: ClassVar[str]

    @property
    def space(self) -> ActionSpace: ...

    @property
    def epsilon(self) -> float:
        """How often it tries something at random right now (0..1)."""
        ...

    @property
    def updates(self) -> int:
        """Steps learned from."""
        ...

    @property
    def episodes(self) -> int:
        """Goes finished."""
        ...

    @property
    def known(self) -> int:
        """Situations it has an opinion about."""
        ...

    @property
    def last_decision(self) -> Decision | None: ...

    def stats(self) -> str: ...

    def choose(self, state: State) -> int: ...

    def greedy(self, state: State) -> int: ...

    def values(self, state: State) -> tuple[float, ...]:
        """What it thinks every move in the space is worth in a situation."""
        ...

    def configure(self, settings: Any) -> None:
        """Learn with these :class:`Settings` from now on (exploration restarts from them)."""
        ...

    def learn(
        self, state: State, action: int, reward: float, next_state: State, done: bool = False
    ) -> float: ...

    def end_episode(self) -> float: ...

    def reset(self) -> None: ...

    def policy(self) -> dict[str, int]:
        """The best known move for every situation it knows (state key -> action)."""
        ...

    def to_dict(self) -> dict[str, Any]: ...


_REGISTRY: dict[str, type[Any]] = {}


def register_learner(cls: type[Any]) -> type[Any]:
    """Class decorator: make ``cls`` buildable by its ``algorithm`` name.

    The class takes ``(space, settings=None, *, seed=None)`` and has a
    ``from_dict(data, *, space=None)`` classmethod; both return a :class:`Learner`.
    """
    name = getattr(cls, "algorithm", "")
    if not isinstance(name, str) or not name:
        raise ConfigurationError("A learner needs a non-empty algorithm name.")
    _REGISTRY[name] = cls
    return cls


def algorithms() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def algorithm_choices() -> tuple[str, ...]:
    """What the child can pick: "choose for me", the table, then the other learners."""
    rest = tuple(name for name in algorithms() if name != Q_TABLE)
    return (AUTO, Q_TABLE, *rest) if Q_TABLE in _REGISTRY else (AUTO, *rest)


def describe_algorithm(name: str) -> tuple[str, str]:
    """(label, one plain sentence) for a choice."""
    if name not in algorithm_choices():
        raise ConfigurationError(
            f"There is no learning algorithm called {name!r} "
            f"(choices: {', '.join(algorithm_choices())})."
        )
    return _LABELS.get(name, (name.replace("_", " ").capitalize(), "An experimental learner."))


def _factory(name: object) -> type[Any]:
    if not isinstance(name, str) or name not in _REGISTRY:
        choices = ", ".join(algorithms()) or "none registered"
        raise ConfigurationError(
            f"There is no learning algorithm called {name!r} (choices: {choices})."
        )
    return _REGISTRY[name]


def make_learner(
    algorithm: str, space: ActionSpace, settings: Any = None, *, seed: int | None = None
) -> Learner:
    learner: Learner = _factory(algorithm)(space, settings, seed=seed)
    return learner


def learner_from_dict(data: dict[str, Any], *, space: ActionSpace | None = None) -> Learner:
    """Rebuild whichever learner saved ``data``; data without a name is a Q-table."""
    if not isinstance(data, dict):
        raise ConfigurationError("Bad brain data: the learner is not a mapping.")
    learner: Learner = _factory(data.get("algorithm", Q_TABLE)).from_dict(data, space=space)
    return learner


__all__ = [
    "AUTO",
    "Q_TABLE",
    "Decision",
    "Learner",
    "algorithm_choices",
    "algorithms",
    "describe_algorithm",
    "learner_from_dict",
    "make_learner",
    "register_learner",
]
