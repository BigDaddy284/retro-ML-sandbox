"""Tabular Q-learning: the beginner learner (DESIGN.md §6.1).

For every situation it has seen (a :class:`State`) the agent keeps a score
per action: how much reward it expects if it does that now and plays well
afterwards. It mostly picks the best-scoring action and sometimes tries
something at random (``epsilon``), trying less as episodes go by. Each
step nudges the score towards what actually happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Any, ClassVar

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.exploring import ResultsWatch
from retro_ai_lab.learning.learner import Q_TABLE, Decision, Learner, register_learner
from retro_ai_lab.learning.state import State

SCHEMA_VERSION = 1


def _check(value: float, what: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{what} must be a number.") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ConfigurationError(f"{what} must be between {low:g} and {high:g}.")
    return number


def _whole(value: object, what: str) -> int:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{what} must be a whole number.") from exc
    if not math.isfinite(number) or number != int(number) or number < 1:
        raise ConfigurationError(f"{what} must be a whole number of 1 or more.")
    return int(number)


@dataclass(frozen=True, slots=True)
class Settings:
    alpha: float = 0.1  # learning rate: how far to move towards what happened
    gamma: float = 0.95  # how much the future counts
    epsilon: float = 1.0  # starting chance of trying something random
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995  # multiplied in at the end of every episode
    # Learned steps over which exploring falls in a straight line to the floor (T-875):
    # a 40-second go is 600 steps, so the per-go decay alone took hundreds of goes.
    explore_steps: int = 6000
    # Exploring goes back up to this when the last ten goes are worse than the ten before
    # (T-932), and eases off again once they recover; 0 leaves the schedule alone.
    rebound: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _check(self.alpha, "alpha", 0.0, 1.0))
        object.__setattr__(self, "gamma", _check(self.gamma, "gamma", 0.0, 1.0))
        object.__setattr__(self, "epsilon", _check(self.epsilon, "epsilon", 0.0, 1.0))
        object.__setattr__(self, "epsilon_min", _check(self.epsilon_min, "epsilon_min", 0.0, 1.0))
        object.__setattr__(
            self, "epsilon_decay", _check(self.epsilon_decay, "epsilon_decay", 0.0, 1.0)
        )
        object.__setattr__(self, "explore_steps", _whole(self.explore_steps, "explore_steps"))
        object.__setattr__(self, "rebound", _check(self.rebound, "rebound", 0.0, 1.0))
        if self.epsilon_min > self.epsilon:
            raise ConfigurationError("epsilon_min can't be above epsilon.")
        if self.alpha == 0.0:
            raise ConfigurationError("alpha must be above zero or nothing is ever learned.")

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "explore_steps": self.explore_steps,
            "rebound": self.rebound,
        }

    def exploring_after(self, updates: int) -> float:
        """Where the step schedule puts exploring after this many learned steps."""
        if updates >= self.explore_steps:
            return self.epsilon_min
        return self.epsilon - (self.epsilon - self.epsilon_min) * updates / self.explore_steps


class QTable:
    def __init__(self, actions: int) -> None:
        if actions < 1:
            raise ConfigurationError("A Q-table needs at least one action.")
        self._actions = actions
        self._rows: dict[str, list[float]] = {}

    @property
    def actions(self) -> int:
        return self._actions

    def __len__(self) -> int:
        return len(self._rows)

    def known_states(self) -> tuple[str, ...]:
        return tuple(self._rows)

    def row(self, state: State) -> list[float]:
        return self._rows.setdefault(state.key, [0.0] * self._actions)

    def values(self, state: State) -> tuple[float, ...]:
        return tuple(self._rows.get(state.key, [0.0] * self._actions))

    def value(self, state: State, action: int) -> float:
        return self.values(state)[action]

    def best(self, state: State, rng: Random | None = None) -> int:
        values = self.values(state)
        top = max(values)
        ties = [i for i, v in enumerate(values) if v == top]
        return ties[0] if rng is None or len(ties) == 1 else rng.choice(ties)

    def to_dict(self) -> dict[str, Any]:
        return {"actions": self._actions, "rows": {k: list(v) for k, v in self._rows.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QTable:
        try:
            table = cls(int(data["actions"]))
            for key, values in data.get("rows", {}).items():
                row = [float(v) for v in values]
                if len(row) != table._actions:
                    raise ConfigurationError(f"Row {key!r} has {len(row)} values.")
                table._rows[str(key)] = row
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad Q-table: {exc}") from exc
        return table


@register_learner
class QLearningAgent:
    algorithm: ClassVar[str] = Q_TABLE

    def __init__(
        self,
        space: ActionSpace,
        settings: Settings | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self._space = space
        self._settings = settings or Settings()
        self._epsilon = self._settings.epsilon
        self._rng = Random(seed)
        self._table = QTable(len(space))
        self._last_decision: Decision | None = None
        self._updates = 0
        self._episodes = 0
        self._results = ResultsWatch(self._settings.rebound)

    # ------------------------------------------------------------ queries

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def table(self) -> QTable:
        return self._table

    @property
    def known(self) -> int:
        return len(self._table)

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def episodes(self) -> int:
        return self._episodes

    def stats(self) -> str:
        n = len(self._table)
        return (
            f"Knows {n} situation{'s' if n != 1 else ''}, learned from {self._updates} "
            f"step{'s' if self._updates != 1 else ''} over {self._episodes} "
            f"go{'es' if self._episodes != 1 else ''}; tries something new "
            f"{self._epsilon:.0%} of the time."
        )

    # ------------------------------------------------------------ acting

    @property
    def last_decision(self) -> Decision | None:
        return self._last_decision

    def choose(self, state: State) -> int:
        if self._rng.random() < self._epsilon:
            action = self._space.sample(self._rng)
            explored = True
        else:
            action = self._table.best(state, self._rng)
            explored = False
        self._last_decision = Decision(action, explored, self._table.value(state, action))
        return action

    def greedy(self, state: State) -> int:
        return self._table.best(state)

    def values(self, state: State) -> tuple[float, ...]:
        """The table's score for every action in a situation (zeros when unseen)."""
        return self._table.values(state)

    def policy(self) -> dict[str, int]:
        return {key: self._table.best(State.from_key(key)) for key in self._table.known_states()}

    # ------------------------------------------------------------ learning

    def learn(
        self, state: State, action: int, reward: float, next_state: State, done: bool = False
    ) -> float:
        """One Q update; returns the new value."""
        self._space[action]  # validates the index
        if not math.isfinite(reward):
            raise ConfigurationError("reward must be a finite number.")
        row = self._table.row(state)
        future = 0.0 if done else max(self._table.values(next_state))
        target = reward + self._settings.gamma * future
        row[action] += self._settings.alpha * (target - row[action])
        self._updates += 1
        self._epsilon = self._scheduled(self._epsilon)
        return row[action]

    def _scheduled(self, epsilon: float) -> float:
        """Exploring capped by the step schedule, unless results hold it up (T-932)."""
        return min(epsilon, max(self._results.lift, self._settings.exploring_after(self._updates)))

    def end_episode(self, reward: float | None = None) -> float:
        """Count an episode and explore a little less from now on.

        With the go's ``reward``, exploring goes back up when the goes get worse (T-932).
        """
        self._episodes += 1
        if reward is not None:
            self._results.note(reward, goes=self._episodes)
        decayed = self._epsilon * self._settings.epsilon_decay
        self._epsilon = max(self._settings.epsilon_min, self._results.lift, decayed)
        self._epsilon = self._scheduled(self._epsilon)
        return self._epsilon

    @property
    def lift(self) -> float:
        """The floor results hold exploring above right now (0 when the schedule rules)."""
        return self._results.lift

    @property
    def recent_rewards(self) -> tuple[float, ...]:
        """The last goes' rewards the lift is judged on (T-932)."""
        return self._results.rewards

    def resume_from(self, other: Learner) -> None:
        """Carry on another learner's exploration and counts (when a brain changes kind)."""
        self._epsilon = _check(other.epsilon, "epsilon", 0.0, 1.0)
        self._updates = other.updates
        self._episodes = other.episodes

    def configure(self, settings: Settings) -> None:
        if not isinstance(settings, Settings):
            raise ConfigurationError("Learning settings must be a Settings.")
        self._settings = settings
        self._epsilon = settings.epsilon
        self._results.set_rebound(settings.rebound)

    def reset(self) -> None:
        self._last_decision = None
        self._table = QTable(len(self._space))
        self._epsilon = self._settings.epsilon
        self._updates = 0
        self._episodes = 0
        self._results = ResultsWatch(self._settings.rebound)

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "algorithm": self.algorithm,
            "space": self._space.to_dict(),
            "settings": self._settings.to_dict(),
            "epsilon": self._epsilon,
            "updates": self._updates,
            "episodes": self._episodes,
            "results": self._results.to_dict(),
            "rng": self._rng.getstate(),
            "table": self._table.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, space: ActionSpace | None = None) -> QLearningAgent:
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(f"This brain is from a newer version (schema {schema}).")
            if data.get("algorithm", cls.algorithm) != cls.algorithm:
                raise ConfigurationError("This brain was not learned with a Q-table.")
            saved = ActionSpace.from_dict(data["space"])
            if space is not None and space != saved:
                raise ConfigurationError(
                    "This brain was trained with different controls than this game has."
                )
            agent = cls(saved, Settings(**data.get("settings", {})))
            agent._epsilon = _check(data.get("epsilon", agent._epsilon), "epsilon", 0.0, 1.0)
            agent._updates = int(data.get("updates", 0))
            agent._episodes = int(data.get("episodes", 0))
            agent._results.load(data.get("results"))
            agent._table = QTable.from_dict(data["table"])
            if agent._table.actions != len(saved):
                raise ConfigurationError("The Q-table does not match the action space.")
            if "rng" in data:
                agent._rng.setstate(_rng_state(data["rng"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad brain data: {exc}") from exc
        return agent


def _rng_state(saved: Any) -> tuple[int, tuple[int, ...], None]:
    """random.getstate() comes back from JSON as lists; rebuild the tuple."""
    version, internal, gauss = saved
    return int(version), tuple(int(x) for x in internal), None if gauss is None else gauss


__all__ = ["SCHEMA_VERSION", "Decision", "QLearningAgent", "QTable", "Settings"]
