"""T-706: moving a brain between the simple table and the neural network."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import (
    AUTO,
    DQN,
    Q_TABLE,
    ActionSpace,
    DQNConfig,
    DQNLearner,
    QLearningAgent,
    Settings,
    State,
    algorithm_choices,
    algorithms,
    convert,
    demote,
    describe_algorithm,
    promote,
)

SPACE = ActionSpace([frozenset(), frozenset({"left"}), frozenset({"right"})])
QUICK = DQNConfig(hidden=(16,), batch_size=8, warmup=8, target_sync=20, learning_rate=0.01)


def taught_table() -> QLearningAgent:
    """A table with a clear favourite in each of twelve situations."""
    agent = QLearningAgent(SPACE, Settings(alpha=0.5, epsilon=0.2, epsilon_decay=0.5), seed=1)
    for i in range(12):
        state = State((("x", i % 4), ("y", i // 4)))
        best = i % 3
        for _ in range(4):
            for action in range(3):
                agent.learn(state, action, 2.0 if action == best else -1.0, state, True)
    agent.end_episode()
    agent.end_episode()
    return agent


def test_choices_and_labels() -> None:
    assert AUTO == "auto" and algorithm_choices() == (AUTO, Q_TABLE, DQN, "skills")
    assert set(algorithm_choices()) == {AUTO, *algorithms()}
    assert describe_algorithm(AUTO)[0] == "Choose for me"
    assert describe_algorithm(Q_TABLE)[0] == "Simple table"
    assert describe_algorithm(DQN)[0] == "Neural network"
    for name in algorithm_choices():
        label, text = describe_algorithm(name)
        assert label and text.endswith(".")
    with pytest.raises(ConfigurationError, match="magic"):
        describe_algorithm("magic")


def test_promote_distils_the_table_into_a_network() -> None:
    table = taught_table()
    assert table.known == 12 and table.episodes == 2 and table.updates == 144
    dqn, agreement = promote(table, seed=3)  # the default network, as the app uses it
    assert isinstance(dqn, DQNLearner) and agreement >= 0.9
    assert dqn.known == 12 and set(dqn.policy()) == set(table.policy())
    agree = sum(dqn.policy()[k] == v for k, v in table.policy().items()) / 12
    assert agree == agreement
    assert dqn.epsilon == table.epsilon and dqn.episodes == 2 and dqn.updates == 144
    assert dqn.settings == table.settings and dqn.space == table.space
    assert dqn.trained == 0 and len(dqn.replay) == 0  # nothing from memory yet
    # An empty table promotes to an untouched network.
    fresh, agreement = promote(QLearningAgent(SPACE), seed=3, config=QUICK)
    assert fresh.known == 0 and agreement == 1.0


def test_demote_copies_the_networks_values_into_a_table() -> None:
    dqn, _ = promote(taught_table(), seed=4)
    table = demote(dqn, seed=5)
    assert isinstance(table, QLearningAgent) and table.known == 12
    for state in dqn.known_states():
        assert table.table.values(state) == dqn.values(state)
    assert table.policy() == dqn.policy()
    assert table.epsilon == dqn.epsilon and table.episodes == 2 and table.updates == 144
    assert demote(DQNLearner(SPACE, config=QUICK)).known == 0


def test_convert_picks_the_right_direction() -> None:
    table = taught_table()
    assert convert(table, Q_TABLE) is table
    assert convert(table, AUTO) is table
    dqn = convert(table, DQN, seed=6)
    assert isinstance(dqn, DQNLearner) and dqn.known == 12
    assert convert(dqn, DQN) is dqn and convert(dqn, AUTO) is dqn
    back = convert(dqn, Q_TABLE)
    assert isinstance(back, QLearningAgent) and back.policy() == dqn.policy()
    with pytest.raises(ConfigurationError, match="magic"):
        convert(table, "magic")
