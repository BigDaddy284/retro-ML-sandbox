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
    "Q_TABLE",
    "Decision",
    "Learner",
    "algorithms",
    "learner_from_dict",
    "make_learner",
    "register_learner",
]
