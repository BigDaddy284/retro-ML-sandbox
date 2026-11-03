"""T-701: any learning algorithm behind one contract."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.learning import (
    Q_TABLE,
    ActionSpace,
    Decision,
    Learner,
    QLearningAgent,
    Settings,
    State,
    algorithms,
    learner_from_dict,
    make_learner,
)
from retro_ai_lab.training import Brain
from retro_ai_lab.training.curve import LearningCurve
from retro_ai_lab.training.insights import compare_policies, policy_of, suggest_experiments

SPACE = ActionSpace.from_buttons(BUTTONS)
A = State((("v1", 1), ("v2", 2)))
B = State((("v1", 2), ("v2", 2)))


def trained() -> QLearningAgent:
    agent = QLearningAgent(SPACE, Settings(epsilon=0.5), seed=4)
    for _ in range(6):
        agent.learn(A, agent.choose(A), 1.0, B, False)
        agent.learn(B, agent.choose(B), -1.0, A, True)
    agent.end_episode()
    return agent


def test_q_learning_agent_is_a_learner_and_reports_its_policy() -> None:
    agent = trained()
    assert isinstance(agent, Learner)
    assert agent.algorithm == Q_TABLE == "q_table"
    assert agent.known == len(agent.table) == 2
    policy = agent.policy()
    assert set(policy) == {A.key, B.key}
    assert policy[A.key] == agent.greedy(A) and policy[B.key] == agent.greedy(B)
    assert agent.to_dict()["algorithm"] == "q_table"
    assert isinstance(agent.last_decision, Decision)


def test_registry_builds_by_name_and_rejects_unknown_names() -> None:
    assert algorithms() == ("q_table",)
    learner = make_learner("q_table", SPACE, Settings(epsilon=0.3), seed=9)
    assert isinstance(learner, QLearningAgent)
    assert learner.epsilon == 0.3 and learner.space == SPACE
    # Same seed, same first choices: the seed went through.
    twin = make_learner("q_table", SPACE, Settings(epsilon=0.3), seed=9)
    assert [learner.choose(A) for _ in range(5)] == [twin.choose(A) for _ in range(5)]
    with pytest.raises(ConfigurationError, match="q_table"):
        make_learner("magic", SPACE)
    with pytest.raises(ConfigurationError):
        make_learner("", SPACE)


def test_learner_from_dict_dispatches_on_the_saved_algorithm() -> None:
    agent = trained()
    data = agent.to_dict()
    again = learner_from_dict(data, space=SPACE)
    assert isinstance(again, QLearningAgent) and again.to_dict() == data
    legacy = dict(data)
    del legacy["algorithm"]  # brains saved before T-701
    assert isinstance(learner_from_dict(legacy, space=SPACE), QLearningAgent)
    with pytest.raises(ConfigurationError, match="magic"):
        learner_from_dict({**data, "algorithm": "magic"}, space=SPACE)
    other = ActionSpace.from_buttons(("A", "B"))
    with pytest.raises(ConfigurationError, match="controls"):
        learner_from_dict(data, space=other)
    with pytest.raises(ConfigurationError):
        learner_from_dict("not a dict", space=SPACE)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        QLearningAgent.from_dict({**data, "algorithm": "magic"})


def test_brain_round_trips_through_the_registry() -> None:
    brain = Brain("toyracer-1", trained(), (), 1, 12)
    data = brain.to_dict()
    assert data["agent"]["algorithm"] == "q_table"
    again = Brain.from_dict(data, space=SPACE)
    assert isinstance(again.agent, QLearningAgent) and again.agent.to_dict() == data["agent"]
    bad = {**data, "agent": {**data["agent"], "algorithm": "magic"}}
    with pytest.raises(ConfigurationError, match="no learning algorithm called 'magic'"):
        Brain.from_dict(bad, space=SPACE)


class Stub:
    """A learner with no table at all: only the protocol."""

    algorithm: ClassVar[str] = "stub"

    def __init__(self, space: ActionSpace, settings: Any = None, *, seed: int | None = None):
        self._space = space
        self.epsilon = 0.9
        self.updates = 0
        self.episodes = 0
        self.last_decision: Decision | None = None
        self.moves = {A.key: 1, B.key: 2}

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def known(self) -> int:
        return len(self.moves)

    def stats(self) -> str:
        return "stub"

    def choose(self, state: State) -> int:
        return self.greedy(state)

    def greedy(self, state: State) -> int:
        return self.moves.get(state.key, 0)

    def learn(
        self, state: State, action: int, reward: float, next_state: State, done: bool = False
    ) -> float:
        self.updates += 1
        return 0.0

    def end_episode(self) -> float:
        self.episodes += 1
        return self.epsilon

    def reset(self) -> None:
        self.moves.clear()

    def policy(self) -> dict[str, int]:
        return dict(self.moves)

    def to_dict(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm}


def test_insights_need_only_the_protocol() -> None:
    stub = Stub(SPACE)
    assert isinstance(stub, Learner)
    before = policy_of(stub)
    stub.moves[A.key] = 3
    changes = compare_policies(before, policy_of(stub))
    assert [(c.state_key, c.before, c.after) for c in changes] == [(A.key, 1, 3)]
    titles = [e.title for e in suggest_experiments(LearningCurve([]), stub, None, None, None)]
    assert titles == ["Keep going"]  # still exploring 90% of the time
