"""T-705: the DQN learner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import (
    DQN,
    ActionSpace,
    Decision,
    DQNConfig,
    DQNLearner,
    Learner,
    Settings,
    State,
    algorithms,
    learner_from_dict,
    make_learner,
)
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    ObservationService,
)
from retro_ai_lab.storage import BrainStore
from retro_ai_lab.training import TrainingService
from retro_ai_lab.training.insights import policy_of
from retro_ai_lab.understanding import Evidence, VariableKind

LEFT, RIGHT = 0, 1
SPACE = ActionSpace([frozenset({"left"}), frozenset({"right"})])
A = State((("x", 0),))
B = State((("x", 1),))
QUICK = DQNConfig(hidden=(16,), batch_size=8, warmup=8, target_sync=20, learning_rate=0.003)


def chain(agent: Learner, steps: int) -> None:
    """Right from A reaches B; right from B pays and goes back to A. Left never pays."""
    state = A
    for _ in range(steps):
        action = agent.choose(state)
        if state == A:
            nxt, reward = (B, 0.0) if action == RIGHT else (A, 0.0)
        else:
            nxt, reward = (A, 1.0) if action == RIGHT else (B, 0.0)
        agent.learn(state, action, reward, nxt)
        state = nxt


def test_conformance_registry_and_validation() -> None:
    assert DQN == "dqn" and "dqn" in algorithms()
    agent = make_learner("dqn", SPACE, Settings(epsilon=0.2), seed=3)
    assert isinstance(agent, DQNLearner) and isinstance(agent, Learner)
    assert agent.algorithm == "dqn" and agent.epsilon == 0.2 and agent.space == SPACE
    assert agent.known == 0 and agent.updates == 0 and agent.episodes == 0
    assert agent.last_decision is None and agent.policy() == {}
    action = agent.choose(A)
    assert action in (LEFT, RIGHT) and isinstance(agent.last_decision, Decision)
    assert agent.last_decision.action == action
    assert agent.greedy(A) == agent.greedy(A)
    with pytest.raises(ConfigurationError):
        agent.learn(A, 2, 1.0, B)
    with pytest.raises(ConfigurationError):
        agent.learn(A, RIGHT, float("nan"), B)
    for bad in (
        {"hidden": ()},
        {"hidden": (0,)},
        {"batch_size": 0},
        {"warmup": 4, "batch_size": 8},
        {"target_sync": 0},
        {"learning_rate": 0.0},
        {"replay_capacity": 4, "batch_size": 8},
        {"width": 0},
    ):
        with pytest.raises(ConfigurationError):
            DQNConfig(**bad)  # type: ignore[arg-type]


def test_learns_the_chain_and_the_corridor() -> None:
    agent = DQNLearner(
        SPACE, Settings(gamma=0.9, epsilon=0.3, epsilon_min=0.3), seed=1, config=QUICK
    )
    chain(agent, 400)
    assert agent.greedy(A) == RIGHT and agent.greedy(B) == RIGHT
    assert agent.value(B, RIGHT) > agent.value(A, RIGHT) > 0
    # A corridor of six squares: right moves on, reward only on reaching the end.
    corridor = DQNLearner(
        SPACE, Settings(gamma=0.9, epsilon=0.3, epsilon_min=0.3), seed=2, config=QUICK
    )
    squares = [State((("x", i),)) for i in range(6)]
    x = 0
    for _ in range(1500):
        action = corridor.choose(squares[x])
        nxt = min(x + 1, 5) if action == RIGHT else max(x - 1, 0)
        done = nxt == 5
        corridor.learn(squares[x], action, 1.0 if done else 0.0, squares[nxt], done)
        x = 0 if done else nxt
    assert [corridor.greedy(s) for s in squares[:5]] == [RIGHT] * 5
    assert corridor.value(squares[4], RIGHT) > corridor.value(squares[0], RIGHT)
    assert corridor.known == 6 and set(corridor.policy()) == {s.key for s in squares}


def test_warmup_batches_and_target_sync() -> None:
    agent = DQNLearner(SPACE, Settings(epsilon=1.0), seed=4, config=QUICK)
    before = agent.network.to_dict()
    for _ in range(QUICK.warmup - 1):
        agent.learn(A, RIGHT, 1.0, B)
    assert agent.trained == 0 and agent.network.to_dict() == before and agent.updates == 7
    agent.learn(B, RIGHT, 1.0, A)
    assert agent.trained == 1 and agent.network.steps == 1 and len(agent.replay) == 8
    assert agent.network.to_dict() != before
    assert agent.target.to_dict()["weights"] == before["weights"]  # not synced yet
    for _ in range(QUICK.target_sync - 1):
        agent.learn(A, LEFT, 0.0, A)
    assert agent.trained == QUICK.target_sync
    assert agent.target.to_dict()["weights"] == agent.network.to_dict()["weights"]
    agent.learn(A, LEFT, 0.0, A)
    assert agent.target.to_dict()["weights"] != agent.network.to_dict()["weights"]
    assert "trained" in agent.stats()


def test_exploration_reset_known_and_stats() -> None:
    agent = DQNLearner(
        SPACE, Settings(epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.5), seed=5, config=QUICK
    )
    assert agent.end_episode() == 0.25 and agent.end_episode() == 0.125
    assert agent.end_episode() == 0.1 and agent.episodes == 3
    chain(agent, 30)
    assert agent.known == 2 and set(agent.policy()) == {A.key, B.key}
    assert agent.policy()[A.key] == agent.greedy(A)
    assert "2 situations" in agent.stats() and "30 steps" in agent.stats()
    assert "3 goes" in agent.stats() and "10%" in agent.stats()
    seen = agent.network.to_dict()
    agent.reset()
    assert agent.epsilon == 0.5 and agent.episodes == 0 and agent.updates == 0
    assert agent.known == 0 and agent.policy() == {} and len(agent.replay) == 0
    assert agent.trained == 0 and agent.network.to_dict() != seen
    assert agent.last_decision is None
    assert "Knows 0 situations" in agent.stats()


def test_exploring_falls_with_the_steps_learned() -> None:
    """T-875: the network learner follows the same per-step line as the table."""
    quick = Settings(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.5, explore_steps=100)
    agent = DQNLearner(SPACE, quick, seed=5, config=QUICK)
    chain(agent, 50)
    assert agent.epsilon == pytest.approx(0.525)
    assert agent.end_episode() == pytest.approx(0.2625)
    chain(agent, 60)
    assert agent.epsilon == 0.05 and agent.updates == 110


def test_exploring_rises_again_when_the_goes_get_worse() -> None:
    """T-932: the network learner lifts exploring to the rebound like the table."""
    quick = Settings(epsilon=1.0, epsilon_min=0.05, epsilon_decay=1.0, explore_steps=10)
    agent = DQNLearner(SPACE, quick, seed=5, config=QUICK)
    chain(agent, 10)
    assert agent.epsilon == 0.05
    for reward in (10.0,) * 10 + (0.0,) * 9:
        assert agent.end_episode(reward) == 0.05
    assert agent.end_episode(0.0) == 0.25 and agent.lift == 0.25
    chain(agent, 5)
    assert agent.epsilon == 0.25
    data = json.loads(json.dumps(agent.to_dict()))
    again = learner_from_dict(data, space=SPACE)
    assert isinstance(again, DQNLearner) and again.lift == 0.25 and again.epsilon == 0.25
    assert again.recent_rewards == agent.recent_rewards
    for reward in (0.0,) * 10 + (10.0,) * 10:
        agent.end_episode(reward)  # no longer worse, then better: the lift eases off
    assert agent.lift == 0.0 and agent.epsilon == 0.05


def test_round_trip_and_refusals(tmp_path: Path) -> None:
    agent = DQNLearner(SPACE, Settings(epsilon=0.4), seed=6, config=QUICK)
    chain(agent, 40)
    agent.end_episode()
    data = json.loads(json.dumps(agent.to_dict()))
    assert data["algorithm"] == "dqn"
    again = learner_from_dict(data, space=SPACE)
    assert isinstance(again, DQNLearner)
    assert again.epsilon == agent.epsilon and again.updates == 40 and again.episodes == 1
    assert again.trained == agent.trained and again.known == 2 and again.config == QUICK
    for s in (A, B):
        assert again.value(s, RIGHT) == agent.value(s, RIGHT)
        assert again.value(s, LEFT) == agent.value(s, LEFT)
    assert [again.choose(A) for _ in range(30)] == [agent.choose(A) for _ in range(30)]
    assert again.learn(B, RIGHT, 1.0, A) == agent.learn(B, RIGHT, 1.0, A)
    assert again.to_dict() == agent.to_dict()
    with pytest.raises(ConfigurationError, match="different controls"):
        DQNLearner.from_dict(data, space=ActionSpace([frozenset()]))
    with pytest.raises(ConfigurationError, match="newer"):
        DQNLearner.from_dict({**data, "schema": 99})
    with pytest.raises(ConfigurationError, match="DQN"):
        DQNLearner.from_dict({**data, "algorithm": "q_table"})
    for bad in ({"schema": 1}, {**data, "network": "nope"}, {**data, "replay": []}):
        with pytest.raises(ConfigurationError):
            DQNLearner.from_dict(bad)  # type: ignore[arg-type]
    # Saved without the random states: still loads, just not bit-for-bit reproducible.
    slim = {k: v for k, v in data.items() if k not in ("rng", "replay")}
    assert learner_from_dict(slim, space=SPACE).updates == 40


def test_training_service_runs_a_dqn_and_restores_it(tmp_path: Path) -> None:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
    training = TrainingService(
        connection,
        observation,
        discovery,
        bus,
        frames_per_step=4,
        max_steps=40,
        settings=Settings(epsilon=0.5),
        seed=1,
        store=BrainStore(tmp_path / "b"),
        algorithm="dqn",
    )
    connection.connect("fake", seed=3)
    k = discovery.knowledge
    assert k is not None
    sure = [Evidence("test", "", 6.0)]
    added = (
        k.propose("ram", LAYOUT.lane, kind=VariableKind.POSITION, name="Lane", evidence=sure),
        k.propose("ram", LAYOUT.speed, kind=VariableKind.SPEED, name="Speed", evidence=sure),
        k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, name="Lives", evidence=sure),
        k.propose("ram", LAYOUT.game_over, kind=VariableKind.FLAG, name="Over", evidence=sure),
    )
    bus.publish(DiscoveryUpdated(added, (), (), ()))
    assert training.algorithm == "dqn" and isinstance(training.agent, DQNLearner)
    training.train(3)
    agent = training.agent
    assert isinstance(agent, DQNLearner)
    assert agent.episodes == 3 and agent.updates == 120 and agent.trained > 0
    assert agent.known > 1 and policy_of(agent) == agent.policy()
    before = agent.to_dict()
    connection.disconnect()
    connection.connect("fake", seed=3)
    again = training.agent
    assert isinstance(again, DQNLearner) and again is not agent
    assert again.to_dict() == before
    brain = training.brain()
    assert brain is not None and brain.algorithm == "dqn"
    assert np.isfinite(again.value(State(), 0))
