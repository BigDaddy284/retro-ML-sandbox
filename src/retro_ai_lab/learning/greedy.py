"""A learner frozen for a race: best known move every time, nothing learned."""

from __future__ import annotations

from typing import ClassVar

from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.learner import Decision, Learner
from retro_ai_lab.learning.state import State


class GreedyPlayer:
    """Wraps a :class:`Learner` so the episode runner can drive it without changing it."""

    algorithm: ClassVar[str] = "greedy"

    def __init__(self, learner: Learner) -> None:
        self._learner = learner
        self._last: Decision | None = None
        self._episodes = 0

    @property
    def learner(self) -> Learner:
        return self._learner

    @property
    def space(self) -> ActionSpace:
        return self._learner.space

    @property
    def epsilon(self) -> float:
        return 0.0

    @property
    def updates(self) -> int:
        return 0

    @property
    def episodes(self) -> int:
        return self._episodes

    @property
    def known(self) -> int:
        return self._learner.known

    @property
    def last_decision(self) -> Decision | None:
        return self._last

    def stats(self) -> str:
        return f"Playing its best known move every time ({self.known} situations known)."

    def choose(self, state: State) -> int:
        action = self._learner.greedy(state)
        self._last = Decision(action, False, 0.0)
        return action

    def greedy(self, state: State) -> int:
        return self._learner.greedy(state)

    def values(self, state: State) -> tuple[float, ...]:
        return self._learner.values(state)

    def configure(self, settings: object) -> None:
        """A race is played as the AI is; its settings are not touched."""

    def learn(
        self, state: State, action: int, reward: float, next_state: State, done: bool = False
    ) -> float:
        return 0.0  # a race is not a lesson

    def end_episode(self) -> float:
        self._episodes += 1
        return 0.0

    def reset(self) -> None:
        self._last = None
        self._episodes = 0

    def policy(self) -> dict[str, int]:
        return self._learner.policy()

    def to_dict(self) -> dict[str, object]:
        return self._learner.to_dict()


__all__ = ["GreedyPlayer"]
