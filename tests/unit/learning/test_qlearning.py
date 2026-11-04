"""T-503: the tabular Q-learning agent."""

from __future__ import annotations

import json
from collections import Counter
from random import Random

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import ActionSpace, QLearningAgent, QTable, Settings, State

LEFT, RIGHT = 0, 1
SPACE = ActionSpace([frozenset({"left"}), frozenset({"right"})])
A = State((("pos", 0),))
B = State((("pos", 1),))


def test_settings_and_table_validation() -> None:
    for bad in (
        dict(alpha=0.0),
        dict(alpha=1.5),
        dict(gamma=-0.1),
        dict(epsilon=2),
        dict(epsilon=0.1, epsilon_min=0.5),
        dict(epsilon_decay=float("nan")),
        dict(alpha="fast"),
        dict(explore_steps=0),
        dict(explore_steps=2.5),
        dict(explore_steps="many"),
    ):
        with pytest.raises(ConfigurationError):
            Settings(**bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        QTable(0)
    t = QTable(2)
    assert t.values(A) == (0.0, 0.0) and len(t) == 0 and t.best(A) == 0
    assert t.best(A, Random(1)) in (0, 1)
    t.row(A)[1] = 2.0
    assert t.best(A, Random(1)) == 1 and t.value(A, 1) == 2.0 and t.known_states() == (A.key,)
    with pytest.raises(ConfigurationError):
        QTable.from_dict({"actions": 2, "rows": {"x": [1.0]}})


def test_update_formula_and_determinism() -> None:
    agent = QLearningAgent(
        SPACE, Settings(alpha=0.5, gamma=0.9, epsilon=0.0, epsilon_min=0.0), seed=3
    )
    agent.table.row(B)[RIGHT] = 10.0
    new = agent.learn(A, RIGHT, 1.0, B)
    assert new == pytest.approx(0.5 * (1.0 + 0.9 * 10.0))
    final = agent.learn(A, LEFT, 1.0, B, done=True)
    assert final == pytest.approx(0.5 * 1.0)  # no future when the episode is over
    assert agent.updates == 2
    with pytest.raises(ConfigurationError):
        agent.learn(A, 5, 1.0, B)
    with pytest.raises(ConfigurationError):
        agent.learn(A, 0, float("inf"), B)
    one = QLearningAgent(SPACE, seed=7)
    two = QLearningAgent(SPACE, seed=7)
    assert [one.choose(A) for _ in range(20)] == [two.choose(A) for _ in range(20)]


def test_exploration_and_decay() -> None:
    always = QLearningAgent(SPACE, Settings(epsilon=1.0, epsilon_min=0.0), seed=1)
    picks = Counter(always.choose(A) for _ in range(200))
    assert set(picks) == {LEFT, RIGHT}
    never = QLearningAgent(SPACE, Settings(epsilon=0.0, epsilon_min=0.0), seed=1)
    never.table.row(A)[RIGHT] = 1.0
    assert {never.choose(A) for _ in range(50)} == {RIGHT} and never.greedy(A) == RIGHT
    agent = QLearningAgent(SPACE, Settings(epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.5))
    assert agent.end_episode() == 0.25
    assert agent.end_episode() == 0.125
    assert agent.end_episode() == 0.1 and agent.episodes == 3  # floor
    agent.reset()
    assert agent.epsilon == 0.5 and agent.episodes == 0 and len(agent.table) == 0


def test_exploring_falls_with_the_steps_learned() -> None:
    """T-875: a straight line from the start to the floor over explore_steps learned steps."""
    quick = Settings(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.5, explore_steps=100)
    assert quick.exploring_after(0) == 1.0 and quick.exploring_after(50) == pytest.approx(0.525)
    assert quick.exploring_after(100) == 0.05 and quick.exploring_after(1000) == 0.05
    agent = QLearningAgent(SPACE, quick, seed=1)
    for _ in range(50):
        agent.learn(A, RIGHT, 0.0, B)
    assert agent.epsilon == pytest.approx(0.525)
    assert agent.end_episode() == pytest.approx(0.2625)  # the per-go decay still applies
    agent.learn(A, RIGHT, 0.0, B)
    assert agent.epsilon == pytest.approx(0.2625)  # the line (0.515) is higher: no change
    for _ in range(60):
        agent.learn(A, RIGHT, 0.0, B)
    assert agent.epsilon == 0.05 and agent.updates == 111  # the floor, and never below it
    assert Settings().explore_steps == 6000
    data = agent.to_dict()
    del data["settings"]["explore_steps"]  # a brain saved before T-875
    assert QLearningAgent.from_dict(data, space=SPACE).settings.explore_steps == 6000
    assert Settings(explore_steps=3000.0).explore_steps == 3000  # a whole number as a float


def test_learns_a_two_state_chain() -> None:
    """Right from A reaches B; right from B pays. Left never pays."""
    agent = QLearningAgent(SPACE, Settings(alpha=0.2, gamma=0.9, epsilon=0.3, epsilon_min=0.3))
    state = A
    for _ in range(600):
        action = agent.choose(state)
        if state == A:
            nxt, reward = (B, 0.0) if action == RIGHT else (A, 0.0)
        else:
            nxt, reward = (A, 1.0) if action == RIGHT else (B, 0.0)
        agent.learn(state, action, reward, nxt)
        state = nxt
    assert agent.greedy(A) == RIGHT and agent.greedy(B) == RIGHT
    assert agent.table.value(B, RIGHT) > agent.table.value(A, RIGHT) > 0
    assert "2 situations" in agent.stats() and "30%" in agent.stats()


def test_round_trip_and_refusals() -> None:
    agent = QLearningAgent(SPACE, Settings(alpha=0.3, epsilon=0.4), seed=5)
    agent.learn(A, RIGHT, 2.0, B)
    agent.end_episode()
    data = json.loads(json.dumps(agent.to_dict()))
    again = QLearningAgent.from_dict(data, space=SPACE)
    assert again.table.values(A) == agent.table.values(A)
    assert again.epsilon == agent.epsilon and again.updates == 1 and again.episodes == 1
    assert again.settings == agent.settings
    assert [again.choose(A) for _ in range(30)] == [agent.choose(A) for _ in range(30)]
    with pytest.raises(ConfigurationError, match="different controls"):
        QLearningAgent.from_dict(data, space=ActionSpace([frozenset()]))
    with pytest.raises(ConfigurationError, match="newer"):
        QLearningAgent.from_dict({**data, "schema": 99})
    with pytest.raises(ConfigurationError):
        QLearningAgent.from_dict({"schema": 1})
    with pytest.raises(ConfigurationError):
        QLearningAgent.from_dict({**data, "table": {"actions": 3, "rows": {}}})
    assert QLearningAgent.from_dict({k: v for k, v in data.items() if k != "rng"}).updates == 1
