"""Deep Q-Network: the intermediate learner (DESIGN.md §6.1).

Where the Q-table keeps one row per situation it has seen, the DQN keeps a
small network that maps a situation's feature vector to a score per action,
so situations it has never seen exactly still get a sensible guess from the
ones nearby. It learns from a replay memory in random batches and chases a
slowly-updated copy of itself (the target network), which is what keeps the
training stable. Everything else, the exploration schedule, the contract
with the training service and the saved brain, matches the Q-table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Any, ClassVar

import numpy as np

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.features import DEFAULT_WIDTH, FeatureMap
from retro_ai_lab.learning.learner import Decision, register_learner
from retro_ai_lab.learning.network import DEFAULT_LEARNING_RATE, Network
from retro_ai_lab.learning.qlearning import Settings, _check, _rng_state
from retro_ai_lab.learning.replay import DEFAULT_CAPACITY, ReplayBuffer, Transition
from retro_ai_lab.learning.state import State

SCHEMA_VERSION = 1
DQN = "dqn"


def _positive(value: Any, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigurationError(f"{what} must be a whole number of at least 1.")
    return value


@dataclass(frozen=True, slots=True)
class DQNConfig:
    hidden: tuple[int, ...] = (64, 64)  # hidden layer sizes
    batch_size: int = 32  # transitions per training step
    warmup: int = 32  # transitions in memory before training starts
    target_sync: int = 250  # training steps between target-network copies
    learning_rate: float = DEFAULT_LEARNING_RATE
    replay_capacity: int = DEFAULT_CAPACITY
    width: int = DEFAULT_WIDTH  # believed variables the feature vector has room for

    def __post_init__(self) -> None:
        hidden = tuple(self.hidden)
        if not hidden:
            raise ConfigurationError("A DQN needs at least one hidden layer.")
        for n in hidden:
            _positive(n, "Every hidden layer size")
        object.__setattr__(self, "hidden", hidden)
        for name in ("batch_size", "warmup", "target_sync", "replay_capacity", "width"):
            _positive(getattr(self, name), name)
        if self.warmup < self.batch_size:
            raise ConfigurationError("warmup can't be below batch_size.")
        if self.replay_capacity < self.batch_size:
            raise ConfigurationError("replay_capacity can't be below batch_size.")
        if not (isinstance(self.learning_rate, int | float) and 0 < self.learning_rate < 1):
            raise ConfigurationError("learning_rate must be between 0 and 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden": list(self.hidden),
            "batch_size": self.batch_size,
            "warmup": self.warmup,
            "target_sync": self.target_sync,
            "learning_rate": self.learning_rate,
            "replay_capacity": self.replay_capacity,
            "width": self.width,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DQNConfig:
        d = cls()
        try:
            return cls(
                tuple(int(n) for n in data.get("hidden", d.hidden)),
                int(data.get("batch_size", d.batch_size)),
                int(data.get("warmup", d.warmup)),
                int(data.get("target_sync", d.target_sync)),
                float(data.get("learning_rate", d.learning_rate)),
                int(data.get("replay_capacity", d.replay_capacity)),
                int(data.get("width", d.width)),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad DQN config: {exc}") from exc


@register_learner
class DQNLearner:
    algorithm: ClassVar[str] = DQN

    def __init__(
        self,
        space: ActionSpace,
        settings: Settings | None = None,
        *,
        seed: int | None = None,
        config: DQNConfig | None = None,
    ) -> None:
        self._space = space
        self._settings = settings or Settings()
        self._config = config or DQNConfig()
        self._rng = Random(seed)
        self._epsilon = self._settings.epsilon
        self._last_decision: Decision | None = None
        self._updates = 0
        self._trained = 0
        self._episodes = 0
        self._build()

    def _build(self) -> None:
        c = self._config
        self._features = FeatureMap(c.width)
        sizes = (self._features.size, *c.hidden, len(self._space))
        self._net = Network(sizes, seed=self._rng.getrandbits(32), learning_rate=c.learning_rate)
        self._target = self._net.copy()
        self._replay = ReplayBuffer(c.replay_capacity, seed=self._rng.getrandbits(32))
        self._seen: dict[str, State] = {}

    # ------------------------------------------------------------ queries

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def config(self) -> DQNConfig:
        return self._config

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def trained(self) -> int:
        """Training steps taken (each on a batch from memory)."""
        return self._trained

    @property
    def episodes(self) -> int:
        return self._episodes

    @property
    def known(self) -> int:
        return len(self._seen)

    @property
    def network(self) -> Network:
        return self._net

    @property
    def target(self) -> Network:
        return self._target

    @property
    def replay(self) -> ReplayBuffer:
        return self._replay

    @property
    def features(self) -> FeatureMap:
        return self._features

    @property
    def last_decision(self) -> Decision | None:
        return self._last_decision

    def stats(self) -> str:
        n = len(self._seen)
        return (
            f"Knows {n} situation{'s' if n != 1 else ''}, learned from {self._updates} "
            f"step{'s' if self._updates != 1 else ''} over {self._episodes} "
            f"go{'es' if self._episodes != 1 else ''}; network trained {self._trained} "
            f"time{'s' if self._trained != 1 else ''}; tries something new "
            f"{self._epsilon:.0%} of the time."
        )

    # ------------------------------------------------------------ acting

    def _q(self, state: State) -> list[float]:
        row = self._net.forward(self._features.vector(state)[None, :])[0]
        return [float(v) for v in row]

    def value(self, state: State, action: int) -> float:
        self._space[action]
        return self._q(state)[action]

    def _best(self, values: list[float], rng: Random | None = None) -> int:
        top = max(values)
        ties = [i for i, v in enumerate(values) if v == top]
        return ties[0] if rng is None or len(ties) == 1 else rng.choice(ties)

    def choose(self, state: State) -> int:
        values = self._q(state)
        if self._rng.random() < self._epsilon:
            action = self._space.sample(self._rng)
            explored = True
        else:
            action = self._best(values, self._rng)
            explored = False
        self._last_decision = Decision(action, explored, values[action])
        return action

    def greedy(self, state: State) -> int:
        return self._best(self._q(state))

    def policy(self) -> dict[str, int]:
        if not self._seen:
            return {}
        rows = self._net.forward(self._features.batch(self._seen.values()))
        return {key: int(np.argmax(row)) for key, row in zip(self._seen, rows, strict=True)}

    # ------------------------------------------------------------ learning

    def _remember(self, *states: State) -> None:
        for state in states:
            self._seen.pop(state.key, None)  # keep the newest at the end
            self._seen[state.key] = state
        while len(self._seen) > self._config.replay_capacity:
            del self._seen[next(iter(self._seen))]

    def learn(
        self, state: State, action: int, reward: float, next_state: State, done: bool = False
    ) -> float:
        """Store the step and train on a batch from memory; returns the new value."""
        self._space[action]  # validates the index
        if not isinstance(reward, int | float) or not math.isfinite(reward):
            raise ConfigurationError("reward must be a finite number.")
        self._replay.append(Transition(state, action, float(reward), next_state, bool(done)))
        self._remember(state, next_state)
        self._updates += 1
        if len(self._replay) >= self._config.warmup:
            self._train_batch()
        return self.value(state, action)

    def _train_batch(self) -> None:
        batch = self._replay.sample(self._config.batch_size)
        x = self._features.batch(t.state for t in batch)
        x_next = self._features.batch(t.next_state for t in batch)
        future = self._target.forward(x_next).max(axis=1)
        targets = np.zeros((len(batch), len(self._space)))
        mask = np.zeros_like(targets)
        gamma = self._settings.gamma
        for i, t in enumerate(batch):
            targets[i, t.action] = t.reward + (0.0 if t.done else gamma * future[i])
            mask[i, t.action] = 1.0
        self._net.train_step(x, targets, mask)
        self._trained += 1
        if self._trained % self._config.target_sync == 0:
            self._target.load_from(self._net)

    def end_episode(self) -> float:
        self._episodes += 1
        self._epsilon = max(
            self._settings.epsilon_min, self._epsilon * self._settings.epsilon_decay
        )
        return self._epsilon

    def reset(self) -> None:
        self._last_decision = None
        self._epsilon = self._settings.epsilon
        self._updates = 0
        self._trained = 0
        self._episodes = 0
        self._build()

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "algorithm": self.algorithm,
            "space": self._space.to_dict(),
            "settings": self._settings.to_dict(),
            "config": self._config.to_dict(),
            "epsilon": self._epsilon,
            "updates": self._updates,
            "trained": self._trained,
            "episodes": self._episodes,
            "rng": self._rng.getstate(),
            "features": self._features.to_dict(),
            "network": self._net.to_dict(),
            "target": self._target.to_dict(),
            "replay": self._replay.to_dict(),
            "seen": [s.key for s in self._seen.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, space: ActionSpace | None = None) -> DQNLearner:
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(f"This brain is from a newer version (schema {schema}).")
            if data.get("algorithm", cls.algorithm) != cls.algorithm:
                raise ConfigurationError("This brain was not learned with a DQN.")
            saved = ActionSpace.from_dict(data["space"])
            if space is not None and space != saved:
                raise ConfigurationError(
                    "This brain was trained with different controls than this game has."
                )
            agent = cls(
                saved,
                Settings(**data.get("settings", {})),
                config=DQNConfig.from_dict(data.get("config", {})),
            )
            agent._epsilon = _check(data.get("epsilon", agent._epsilon), "epsilon", 0.0, 1.0)
            agent._updates = int(data.get("updates", 0))
            agent._trained = int(data.get("trained", 0))
            agent._episodes = int(data.get("episodes", 0))
            if "features" in data:
                agent._features = FeatureMap.from_dict(data["features"])
            agent._net = Network.from_dict(data["network"])
            agent._target = (
                Network.from_dict(data["target"]) if "target" in data else agent._net.copy()
            )
            expected = (agent._features.size, *agent._config.hidden, len(saved))
            if agent._net.sizes != expected or agent._target.sizes != expected:
                raise ConfigurationError("The network does not match the controls and features.")
            if "replay" in data:
                agent._replay = ReplayBuffer.from_dict(data["replay"])
            agent._seen = {}
            agent._remember(*(State.from_key(str(k)) for k in data.get("seen", [])))
            if "rng" in data:
                agent._rng.setstate(_rng_state(data["rng"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad brain data: {exc}") from exc
        return agent


__all__ = ["DQN", "SCHEMA_VERSION", "DQNConfig", "DQNLearner"]
