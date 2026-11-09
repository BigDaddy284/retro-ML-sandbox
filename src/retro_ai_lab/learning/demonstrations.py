"""The child's play, kept for the AI to learn from again (DESIGN.md §6.3, §9).

In manual mode the AI learns from every step the child takes as it happens.
Keeping those steps lets the AI rehearse them later: after it has forgotten
a trick, after it has changed how it learns, or simply because a few
hundred good moves are worth more than once through. Demonstrating is
optional; an empty record costs nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.replay import ReplayBuffer, Transition
from retro_ai_lab.learning.state import State

SCHEMA_VERSION = 1
DEFAULT_DEMO_CAPACITY = 5_000
CHILD = "child"


class _Learns(Protocol):
    def learn(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool = False,
        *,
        steps: int = 1,
    ) -> float: ...


class Demonstrations:
    def __init__(self, capacity: int = DEFAULT_DEMO_CAPACITY, *, seed: int | None = None) -> None:
        self._buffer = ReplayBuffer(capacity, seed=seed)

    @property
    def capacity(self) -> int:
        return self._buffer.capacity

    @property
    def count(self) -> int:
        """Demonstrations kept."""
        return len(self._buffer)

    @property
    def shown(self) -> int:
        """Moves the child has shown in all, kept or not."""
        return self._buffer.added

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._buffer)

    def record(
        self, state: State, action: int, reward: float, next_state: State, done: bool = False
    ) -> Transition:
        transition = Transition(state, action, reward, next_state, done, CHILD)
        self._buffer.append(transition)
        return transition

    def add(self, transition: Transition) -> None:
        """Keep a transition someone else built; it is marked as the child's."""
        if transition.source != CHILD:
            transition = Transition(
                transition.state,
                transition.action,
                transition.reward,
                transition.next_state,
                transition.done,
                CHILD,
            )
        self._buffer.append(transition)

    def rehearse(self, learner: _Learns, *, passes: int = 1) -> int:
        """Feed every demonstration to the learner again, oldest first; returns the steps."""
        if passes < 0:
            raise ConfigurationError("passes can't be negative.")
        steps = 0
        for _ in range(passes):
            for t in self._buffer:
                learner.learn(t.state, t.action, t.reward, t.next_state, t.done, steps=t.steps)
                steps += 1
        return steps

    def clear(self) -> None:
        """Forget every move shown (the count starts again too)."""
        self._buffer = ReplayBuffer(self._buffer.capacity)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA_VERSION, "buffer": self._buffer.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Demonstrations:
        if not isinstance(data, dict):
            raise ConfigurationError("Bad demonstration data: not a mapping.")
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(
                    f"These demonstrations are from a newer version (schema {schema})."
                )
            demos = cls()
            if "buffer" in data:
                demos._buffer = ReplayBuffer.from_dict(data["buffer"])
            return demos
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad demonstration data: {exc}") from exc


__all__ = ["CHILD", "DEFAULT_DEMO_CAPACITY", "SCHEMA_VERSION", "Demonstrations"]
