"""T-707: the child's play, kept for the AI to learn from again."""

from __future__ import annotations

import json

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import (
    ActionSpace,
    Demonstrations,
    QLearningAgent,
    Settings,
    State,
    Transition,
)
from retro_ai_lab.learning.demonstrations import DEFAULT_DEMO_CAPACITY, SCHEMA_VERSION

SPACE = ActionSpace([frozenset(), frozenset({"up"}), frozenset({"down"})])
S = [State((("x", i),)) for i in range(5)]


def shown(n: int, demos: Demonstrations) -> None:
    for i in range(n):
        demos.record(S[i % 5], 1 + i % 2, float(i), S[(i + 1) % 5], i % 7 == 6)


def test_bounds_order_and_sources() -> None:
    demos = Demonstrations(capacity=4)
    assert demos.count == 0 and Demonstrations().capacity == DEFAULT_DEMO_CAPACITY
    shown(6, demos)
    assert demos.count == 4 and demos.shown == 6
    kept = list(demos)
    assert [t.reward for t in kept] == [2.0, 3.0, 4.0, 5.0]
    assert all(t.source == "child" for t in kept)
    assert all(isinstance(t, Transition) for t in kept)
    with pytest.raises(ConfigurationError):
        Demonstrations(capacity=0)
    with pytest.raises(ConfigurationError):
        demos.record(S[0], -1, 0.0, S[1], False)
    with pytest.raises(ConfigurationError):
        demos.record(S[0], 1, float("inf"), S[1], False)
    demos.clear()
    assert demos.count == 0 and demos.shown == 0


def test_json_round_trip_and_bad_data() -> None:
    demos = Demonstrations(capacity=10)
    shown(7, demos)
    data = json.loads(json.dumps(demos.to_dict()))
    assert data["schema"] == SCHEMA_VERSION
    again = Demonstrations.from_dict(data)
    assert again.capacity == 10 and again.count == 7 and again.shown == 7
    assert list(again) == list(demos) and again.to_dict() == demos.to_dict()
    for bad in (
        "nope",
        {**data, "buffer": "x"},
        {**data, "buffer": {**data["buffer"], "capacity": 0}},
    ):
        with pytest.raises(ConfigurationError):
            Demonstrations.from_dict(bad)  # type: ignore[arg-type]
    assert Demonstrations.from_dict({"schema": 1}).count == 0


def test_rehearse_feeds_every_demonstration_again_in_order() -> None:
    demos = Demonstrations()
    shown(6, demos)
    agent = QLearningAgent(SPACE, Settings(epsilon=0.0, epsilon_min=0.0), seed=1)
    assert demos.rehearse(agent) == 6 and agent.updates == 6
    assert set(agent.table.known_states()) == {s.key for s in S}
    assert demos.rehearse(agent, passes=2) == 12 and agent.updates == 18
    assert demos.rehearse(agent, passes=0) == 0
    with pytest.raises(ConfigurationError):
        demos.rehearse(agent, passes=-1)
    seen: list[tuple[str, int, float]] = []

    class Spy:
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
            assert steps == 1  # the child's moves are one-step lessons (T-963)
            seen.append((state.key, action, reward))
            return 0.0

    assert demos.rehearse(Spy()) == 6  # type: ignore[arg-type]
    assert [r for _, _, r in seen] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert Demonstrations().rehearse(agent) == 0
