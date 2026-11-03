"""Experience replay: a bounded memory of past steps (DESIGN.md §6.1).

A learner that trains on every step as it happens only ever sees the most
recent few moments. Keeping the last ``capacity`` steps and learning from a
random handful of them at a time lets a network see the same moment many
times and in a mixed order, which is what keeps it from forgetting.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from random import Random
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.state import State

SCHEMA_VERSION = 1
DEFAULT_CAPACITY = 10_000


@dataclass(frozen=True, slots=True)
class Transition:
    """One step: where it was, what it did, what it got, where it ended up."""

    state: State
    action: int
    reward: float
    next_state: State
    done: bool
    source: str = "ai"  # who drove the step: "ai" or "child"

    def __post_init__(self) -> None:
        if not isinstance(self.state, State) or not isinstance(self.next_state, State):
            raise ConfigurationError("A transition needs a State before and after.")
        if not isinstance(self.action, int) or self.action < 0:
            raise ConfigurationError("A transition's action must be a non-negative index.")
        if not isinstance(self.reward, (int, float)) or not math.isfinite(self.reward):
            raise ConfigurationError("A transition's reward must be a finite number.")
        if not isinstance(self.source, str) or not self.source:
            raise ConfigurationError("A transition's source must be a non-empty name.")
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "done", bool(self.done))

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.key,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.key,
            "done": self.done,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transition:
        try:
            return cls(
                State.from_key(str(data["state"])),
                int(data["action"]),
                float(data["reward"]),
                State.from_key(str(data["next_state"])),
                bool(data["done"]),
                str(data.get("source", "ai")),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ConfigurationError(f"Bad transition data: {exc}") from exc


class ReplayBuffer:
    def __init__(self, capacity: int = DEFAULT_CAPACITY, *, seed: int | None = None) -> None:
        if not isinstance(capacity, int) or capacity < 1:
            raise ConfigurationError("A replay buffer needs a capacity of at least one.")
        self._capacity = capacity
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._added = 0
        self._rng = Random(seed)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def added(self) -> int:
        """Transitions ever stored, including those since dropped."""
        return self._added

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Transition]:
        return iter(tuple(self._items))

    def append(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise ConfigurationError("Only transitions can be remembered.")
        self._items.append(transition)
        self._added += 1

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.append(transition)

    def latest(self, n: int) -> tuple[Transition, ...]:
        """The newest ``n`` (or fewer), oldest first."""
        if n < 1:
            return ()
        return tuple(self._items)[-n:]

    def sample(self, n: int) -> tuple[Transition, ...]:
        """``n`` distinct transitions in random order."""
        if n < 1:
            raise ConfigurationError("Sample at least one transition.")
        if not self._items:
            raise ConfigurationError("The replay buffer is empty.")
        if n > len(self._items):
            raise ConfigurationError(
                f"The replay buffer only holds {len(self._items)} transitions, not {n}."
            )
        return tuple(self._rng.sample(tuple(self._items), n))

    def clear(self) -> None:
        self._items.clear()

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "capacity": self._capacity,
            "added": self._added,
            "rng": self._rng.getstate(),
            "transitions": [t.to_dict() for t in self._items],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayBuffer:
        if not isinstance(data, dict):
            raise ConfigurationError("Bad replay data: not a mapping.")
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(f"This replay memory is from a newer version ({schema}).")
            buffer = cls(int(data["capacity"]))
            transitions = data.get("transitions", [])
            if not isinstance(transitions, list):
                raise ConfigurationError("Bad replay data: transitions must be a list.")
            buffer.extend(Transition.from_dict(t) for t in transitions)
            buffer._added = max(int(data.get("added", len(transitions))), len(buffer))
            if "rng" in data:
                version, internal, gauss = data["rng"]
                buffer._rng.setstate((int(version), tuple(int(x) for x in internal), gauss))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad replay data: {exc}") from exc
        return buffer


__all__ = ["DEFAULT_CAPACITY", "SCHEMA_VERSION", "ReplayBuffer", "Transition"]
