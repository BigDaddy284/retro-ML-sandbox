"""The screen network: a learner that watches the screen (T-1055).

The table and the first network are fed a handful of numbers guessed from the
game's memory, and can learn nothing those numbers do not say: where an obstacle
is on a road, where the floor ends. This learner is fed what every game shows:
the screen, shrunk to a small grey picture (:mod:`retro_ai_lab.learning.picture`),
what changed in it since the step before, and the move it last made. From those
it scores every move, and it learns the scores from the rewards the app works
out, as the first network does: a memory of steps replayed in random batches, and
a slowly updated copy of itself to aim at. The next move is chosen with the
learning network and valued with the copy (double Q-learning), which keeps a
lucky guess from feeding on itself.

Nothing in it knows a game. A situation that shows no picture cannot be learnt
from, and is counted.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from random import Random
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.exploring import ResultsWatch, Runs
from retro_ai_lab.learning.learner import Decision, Learner, register_learner
from retro_ai_lab.learning.network import Network
from retro_ai_lab.learning.picture import NO_MOVE, PICTURE_SIZE, Sight
from retro_ai_lab.learning.qlearning import (
    Settings,
    _check,
    _rng_state,
    settings_from_saved,
    tries_words,
)
from retro_ai_lab.learning.state import State

SCHEMA_VERSION = 1
SCREEN = "screen"

# One step in memory: what it saw, the move, the reward (scaled), what it saw after,
# whether the game started over there, and the steps the reward spans.
_Step = tuple[Sight, int, float, Sight, bool, int]


def _positive(value: Any, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigurationError(f"{what} must be a whole number of at least 1.")
    return value


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    hidden: tuple[int, ...] = (128, 64)  # hidden layer sizes
    batch_size: int = 32  # steps per training step
    warmup: int = 500  # steps in memory before training starts
    target_sync: int = 500  # training steps between copies to the target network
    learning_rate: float = 5e-4
    memory: int = 30_000  # steps remembered (a picture is 896 bytes, shared by two steps)
    # The app's rewards are tens of points; a network learns best from numbers near one.
    reward_scale: float = 0.1

    def __post_init__(self) -> None:
        hidden = tuple(self.hidden)
        if not hidden:
            raise ConfigurationError("The screen network needs at least one hidden layer.")
        for n in hidden:
            _positive(n, "Every hidden layer size")
        object.__setattr__(self, "hidden", hidden)
        for name in ("batch_size", "warmup", "target_sync", "memory"):
            _positive(getattr(self, name), name)
        if self.warmup < self.batch_size:
            raise ConfigurationError("warmup can't be below batch_size.")
        if self.memory < self.warmup:
            raise ConfigurationError("memory can't be below warmup.")
        if not (isinstance(self.learning_rate, int | float) and 0 < self.learning_rate < 1):
            raise ConfigurationError("learning_rate must be between 0 and 1.")
        scale = self.reward_scale
        if isinstance(scale, bool) or not isinstance(scale, int | float) or not 0 < scale <= 1:
            raise ConfigurationError("reward_scale must be above 0 and at most 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden": list(self.hidden),
            "batch_size": self.batch_size,
            "warmup": self.warmup,
            "target_sync": self.target_sync,
            "learning_rate": self.learning_rate,
            "memory": self.memory,
            "reward_scale": self.reward_scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScreenConfig:
        d = cls()
        try:
            return cls(
                tuple(int(n) for n in data.get("hidden", d.hidden)),
                int(data.get("batch_size", d.batch_size)),
                int(data.get("warmup", d.warmup)),
                int(data.get("target_sync", d.target_sync)),
                float(data.get("learning_rate", d.learning_rate)),
                int(data.get("memory", d.memory)),
                float(data.get("reward_scale", d.reward_scale)),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad screen network config: {exc}") from exc


def inputs(sights: list[Sight], moves: int) -> NDArray[np.float64]:
    """One row a sight: the picture, what changed in it since the step before, and
    the move last made as one slot in ``moves`` (none set at a go's start)."""
    n = len(sights)
    seen = np.frombuffer(b"".join(s.now for s in sights), dtype=np.uint8)
    earlier = np.frombuffer(b"".join(s.before for s in sights), dtype=np.uint8)
    now = seen.reshape(n, PICTURE_SIZE) / 255.0
    rows = np.zeros((n, 2 * PICTURE_SIZE + moves))
    rows[:, :PICTURE_SIZE] = now
    rows[:, PICTURE_SIZE : 2 * PICTURE_SIZE] = now - earlier.reshape(n, PICTURE_SIZE) / 255.0
    for i, sight in enumerate(sights):
        if NO_MOVE < sight.held < moves:
            rows[i, 2 * PICTURE_SIZE + sight.held] = 1.0
    return rows


@register_learner
class ScreenLearner:
    algorithm: ClassVar[str] = SCREEN
    wants_picture: ClassVar[bool] = True  # the runner hands every state its sight

    def __init__(
        self,
        space: ActionSpace,
        settings: Settings | None = None,
        *,
        seed: int | None = None,
        config: ScreenConfig | None = None,
    ) -> None:
        self._space = space
        self._settings = settings or Settings()
        self._config = config or ScreenConfig()
        self._rng = Random(seed)
        self._epsilon = self._settings.epsilon
        self._last_decision: Decision | None = None
        self._updates = 0
        self._trained = 0
        self._episodes = 0
        self._unseen = 0
        self._results = ResultsWatch(self._settings.rebound)
        self._runs = Runs(self._settings.explore_run)
        self._build()

    def _build(self) -> None:
        c = self._config
        sizes = (2 * PICTURE_SIZE + len(self._space), *c.hidden, len(self._space))
        self._net = Network(sizes, seed=self._rng.getrandbits(32), learning_rate=c.learning_rate)
        self._target = self._net.copy()
        self._memory: list[_Step] = []
        self._at = 0  # where the next step goes once the memory is full

    # ------------------------------------------------------------ queries

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def config(self) -> ScreenConfig:
        return self._config

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def updates(self) -> int:
        return self._updates

    @property
    def trained(self) -> int:
        """Training steps the network has taken."""
        return self._trained

    @property
    def episodes(self) -> int:
        return self._episodes

    @property
    def known(self) -> int:
        """Steps in memory."""
        return len(self._memory)

    @property
    def unseen(self) -> int:
        """Steps it was asked to learn from that showed no picture."""
        return self._unseen

    @property
    def network(self) -> Network:
        return self._net

    @property
    def target(self) -> Network:
        return self._target

    @property
    def last_decision(self) -> Decision | None:
        return self._last_decision

    def stats(self) -> str:
        n = len(self._memory)
        blind = (
            f"; {self._unseen} step{'s' if self._unseen != 1 else ''} showed it no picture "
            "and taught it nothing"
            if self._unseen
            else ""
        )
        return (
            f"Watches the screen. Remembers {n} step{'s' if n != 1 else ''}, learned from "
            f"{self._updates} step{'s' if self._updates != 1 else ''} over {self._episodes} "
            f"go{'es' if self._episodes != 1 else ''}; network trained {self._trained} "
            f"time{'s' if self._trained != 1 else ''}{blind}; "
            f"{tries_words(self._epsilon, self._settings.explore_run)}"
        )

    # ------------------------------------------------------------ acting

    def _q(self, state: State) -> list[float]:
        if state.sight is None:
            return [0.0] * len(self._space)
        row = self._net.forward(inputs([state.sight], len(self._space)))[0]
        return [float(v) for v in row]

    def values(self, state: State) -> tuple[float, ...]:
        """The network's score for every move where the state shows a picture; noughts
        where it shows none."""
        return tuple(self._q(state))

    def value(self, state: State, action: int) -> float:
        self._space[action]
        return self._q(state)[action]

    def _best(self, values: list[float], rng: Random | None = None) -> int:
        top = max(values)
        ties = [i for i, v in enumerate(values) if v == top]
        return ties[0] if rng is None or len(ties) == 1 else rng.choice(ties)

    def choose(self, state: State) -> int:
        values = self._q(state)
        own = self._best(values)  # what it would do; a try is made around it (T-998)
        tried = self._runs.next(
            self._rng, self._epsilon, lambda rng: self._space.sample_near(rng, own)
        )
        if tried is not None:
            action = tried
            explored = True
        else:
            action = self._best(values, self._rng)
            explored = False
        self._last_decision = Decision(action, explored, values[action])
        return action

    def greedy(self, state: State) -> int:
        return self._best(self._q(state))

    def policy(self) -> dict[str, int]:
        """Nothing: its situations are pictures, and there is no list of those to show."""
        return {}

    # ------------------------------------------------------------ learning

    def learn(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool = False,
        *,
        steps: int = 1,
    ) -> float:
        """Remember the step and train on a batch from memory; returns the new value.

        ``reward`` is the return over ``steps`` steps and ``next_state`` the situation
        after them, as the runner hands them (T-963). A step that shows no picture
        before or after teaches nothing and is counted in :attr:`unseen`.
        """
        self._space[action]  # validates the index
        if not isinstance(reward, int | float) or not math.isfinite(reward):
            raise ConfigurationError("reward must be a finite number.")
        _positive(steps, "steps")
        if state.sight is None or next_state.sight is None:
            self._unseen += 1
            return 0.0
        step: _Step = (
            state.sight,
            action,
            float(reward) * self._config.reward_scale,
            next_state.sight,
            bool(done),
            steps,
        )
        if len(self._memory) < self._config.memory:
            self._memory.append(step)
        else:
            self._memory[self._at] = step
            self._at = (self._at + 1) % self._config.memory
        self._updates += 1
        self._epsilon = self._scheduled(self._epsilon)
        if len(self._memory) >= self._config.warmup:
            self._train_batch()
        return self.value(state, action)

    def _train_batch(self) -> None:
        batch = [
            self._memory[self._rng.randrange(len(self._memory))]
            for _ in range(self._config.batch_size)
        ]
        moves = len(self._space)
        x = inputs([step[0] for step in batch], moves)
        x_next = inputs([step[3] for step in batch], moves)
        # The learning network says which move it would make next; the target says
        # what that move is worth.
        chosen = self._net.forward(x_next).argmax(axis=1)
        future = self._target.forward(x_next)[np.arange(len(batch)), chosen]
        targets = np.zeros((len(batch), moves))
        mask = np.zeros_like(targets)
        gamma = self._settings.gamma
        for i, (_, action, reward, _, done, steps) in enumerate(batch):
            targets[i, action] = reward + (0.0 if done else gamma**steps * future[i])
            mask[i, action] = 1.0
        self._net.train_step(x, targets, mask)
        self._trained += 1
        if self._trained % self._config.target_sync == 0:
            self._target.load_from(self._net)

    def resume_from(self, other: Learner) -> None:
        """Carry on another learner's counts. Its exploring is not carried: this
        network knows nothing yet, whatever the other had learnt."""
        self._updates = 0
        self._episodes = other.episodes

    def _scheduled(self, epsilon: float) -> float:
        """Exploring capped by the step schedule, unless results hold it up (T-932)."""
        return min(epsilon, max(self._results.lift, self._settings.exploring_after(self._updates)))

    def end_episode(self, reward: float | None = None) -> float:
        self._episodes += 1
        self._runs.drop()  # a run belongs to the go it started in (T-983)
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

    def configure(self, settings: Settings) -> None:
        if not isinstance(settings, Settings):
            raise ConfigurationError("Learning settings must be a Settings.")
        self._settings = settings
        self._epsilon = settings.epsilon
        self._results.set_rebound(settings.rebound)
        self._runs.set_most(settings.explore_run)

    def reset(self) -> None:
        self._last_decision = None
        self._epsilon = self._settings.epsilon
        self._updates = 0
        self._trained = 0
        self._episodes = 0
        self._unseen = 0
        self._results = ResultsWatch(self._settings.rebound)
        self._runs = Runs(self._settings.explore_run)
        self._build()

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        """Everything but the memory of steps: its pictures would make the file a
        hundred times the size, and the network is what was learnt from them."""
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
            "unseen": self._unseen,
            "results": self._results.to_dict(),
            "run": self._runs.to_dict(),
            "rng": self._rng.getstate(),
            "network": _packed(self._net),
            "target": _packed(self._target),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, space: ActionSpace | None = None) -> ScreenLearner:
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(f"This brain is from a newer version (schema {schema}).")
            if data.get("algorithm", cls.algorithm) != cls.algorithm:
                raise ConfigurationError("This brain was not learned by a screen network.")
            saved = ActionSpace.from_dict(data["space"])
            if space is not None and space != saved:
                raise ConfigurationError(
                    "This brain was trained with different controls than this game has."
                )
            agent = cls(
                saved,
                settings_from_saved(data.get("settings")),
                config=ScreenConfig.from_dict(data.get("config", {})),
            )
            agent._epsilon = _check(data.get("epsilon", agent._epsilon), "epsilon", 0.0, 1.0)
            agent._updates = int(data.get("updates", 0))
            agent._trained = int(data.get("trained", 0))
            agent._episodes = int(data.get("episodes", 0))
            agent._unseen = int(data.get("unseen", 0))
            agent._results.load(data.get("results"))
            agent._runs.load(data.get("run"), actions=len(saved))
            _unpack(agent._net, data["network"])
            if "target" in data:
                _unpack(agent._target, data["target"])
            else:
                agent._target.load_from(agent._net)
            if "rng" in data:
                agent._rng.setstate(_rng_state(data["rng"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad brain data: {exc}") from exc
        return agent


def _packed(network: Network) -> dict[str, Any]:
    """The network's numbers as they are, in text a file can hold: a quarter of a
    million of them written out one by one would be megabytes of digits."""
    flat = np.concatenate([p.ravel() for p in network.parameters]).astype("<f8")
    return {"sizes": list(network.sizes), "numbers": base64.b64encode(flat.tobytes()).decode()}


def _unpack(network: Network, data: Any) -> None:
    if not isinstance(data, dict):
        raise ConfigurationError("Bad brain data: the network is not a mapping.")
    if tuple(data.get("sizes", ())) != network.sizes:
        raise ConfigurationError("The network does not match the controls and the picture.")
    try:
        flat = np.frombuffer(base64.b64decode(data["numbers"], validate=True), dtype="<f8")
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Bad brain data: the network's numbers ({exc}).") from exc
    wanted = sum(p.size for p in network.parameters)
    if flat.size != wanted or not np.isfinite(flat).all():
        raise ConfigurationError("Bad brain data: the network's numbers are not all there.")
    at = 0
    for p in network.parameters:
        p[...] = flat[at : at + p.size].reshape(p.shape)
        at += p.size


__all__ = ["SCHEMA_VERSION", "SCREEN", "ScreenConfig", "ScreenLearner", "inputs"]
