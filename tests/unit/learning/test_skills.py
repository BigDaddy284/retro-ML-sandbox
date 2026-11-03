"""T-708: a skill per goal step."""

from __future__ import annotations

import json

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import (
    DQN,
    Q_TABLE,
    SKILLS,
    ActionSpace,
    DQNLearner,
    Learner,
    QLearningAgent,
    Settings,
    SkillsLearner,
    State,
    algorithm_choices,
    convert,
    describe_algorithm,
    learner_from_dict,
    make_learner,
)
from retro_ai_lab.learning.skills import DEFAULT_SKILL

LEFT, RIGHT = 0, 1
SPACE = ActionSpace([frozenset({"left"}), frozenset({"right"})])
A = State((("x", 0),))
B = State((("x", 1),))


def test_conformance_routing_and_counts() -> None:
    assert SKILLS == "skills" and SKILLS in algorithm_choices()
    assert describe_algorithm(SKILLS)[0] == "Skills per goal step"
    agent = make_learner(SKILLS, SPACE, Settings(epsilon=0.3), seed=1)
    assert isinstance(agent, SkillsLearner) and isinstance(agent, Learner)
    assert agent.algorithm == SKILLS and agent.inner == Q_TABLE
    assert agent.skill == DEFAULT_SKILL and agent.skills == (DEFAULT_SKILL,)
    assert isinstance(agent.skill_learner(DEFAULT_SKILL), QLearningAgent)
    agent.learn(A, RIGHT, 1.0, B)
    assert agent.updates == 1 and agent.known == 1
    jump = agent.set_skill("jump")
    assert agent.skill == "jump" and agent.skills == (DEFAULT_SKILL, "jump")
    assert isinstance(jump, QLearningAgent) and jump.updates == 0
    agent.learn(A, LEFT, 1.0, B)
    agent.learn(B, LEFT, 1.0, A)
    assert agent.updates == 3 and agent.known == 3 and jump.updates == 2
    assert agent.greedy(A) == LEFT  # the active skill answers
    assert agent.set_skill(None) is agent.skill_learner(DEFAULT_SKILL)
    assert agent.greedy(A) == RIGHT
    action = agent.choose(A)
    assert agent.last_decision is not None and agent.last_decision.action == action
    assert agent.epsilon == 0.3
    assert agent.end_episode() == pytest.approx(0.3 * 0.995) and agent.episodes == 1
    assert jump.epsilon == pytest.approx(0.3 * 0.995)  # every skill explores a little less
    with pytest.raises(ConfigurationError):
        agent.set_skill("")
    with pytest.raises(ConfigurationError):
        agent.skill_learner("nope")
    with pytest.raises(ConfigurationError):
        SkillsLearner(SPACE, inner="magic")
    with pytest.raises(ConfigurationError):
        SkillsLearner(SPACE, inner=SKILLS)


def opposite_steps(agent: Learner, rounds: int, *, skills: bool) -> None:
    """Step "go" pays for RIGHT in A, step "back" pays for LEFT in A."""
    for _ in range(rounds):
        for name, good in (("go", RIGHT), ("back", LEFT)):
            if skills:
                assert isinstance(agent, SkillsLearner)
                agent.set_skill(name)
            for _ in range(5):
                action = agent.choose(A)
                agent.learn(A, action, 1.0 if action == good else -1.0, A)


def test_skills_learn_opposite_steps_where_one_table_cannot() -> None:
    settings = Settings(alpha=0.3, epsilon=0.3, epsilon_min=0.3)
    skilled = SkillsLearner(SPACE, settings, seed=2)
    opposite_steps(skilled, 40, skills=True)
    assert skilled.set_skill("go").greedy(A) == RIGHT
    assert skilled.set_skill("back").greedy(A) == LEFT
    plain = QLearningAgent(SPACE, settings, seed=2)
    opposite_steps(plain, 40, skills=False)
    values = plain.table.values(A)
    assert abs(values[LEFT] - values[RIGHT]) < 1.0  # torn between the two


def test_policy_keys_stats_and_reset() -> None:
    agent = SkillsLearner(SPACE, Settings(epsilon=0.0, epsilon_min=0.0), seed=3)
    agent.learn(A, RIGHT, 1.0, B)
    agent.set_skill("jump")
    agent.learn(A, LEFT, 1.0, B)
    policy = agent.policy()
    assert len(policy) == 2
    parsed = {State.from_key(k): v for k, v in policy.items()}
    assert parsed[State((("skill", 0), ("x", 0)))] == RIGHT
    assert parsed[State((("skill", 1), ("x", 0)))] == LEFT
    text = agent.stats()
    assert "2 skills" in text and "jump" in text and "Knows 2 situations" in text
    agent.reset()
    assert agent.skills == (DEFAULT_SKILL,) and agent.skill == DEFAULT_SKILL
    assert agent.updates == 0 and agent.known == 0 and agent.policy() == {}
    assert "1 skill" in agent.stats()


def test_round_trip_and_refusals() -> None:
    agent = SkillsLearner(SPACE, Settings(epsilon=0.4), seed=4)
    agent.learn(A, RIGHT, 1.0, B)
    agent.set_skill("jump").learn(A, LEFT, 2.0, B)
    agent.end_episode()
    data = json.loads(json.dumps(agent.to_dict()))
    assert data["algorithm"] == SKILLS and [s["id"] for s in data["skills"]] == [
        DEFAULT_SKILL,
        "jump",
    ]
    again = learner_from_dict(data, space=SPACE)
    assert isinstance(again, SkillsLearner)
    assert again.skill == "jump" and again.skills == agent.skills and again.inner == Q_TABLE
    assert again.updates == 2 and again.episodes == 1 and again.policy() == agent.policy()
    assert [again.choose(A) for _ in range(20)] == [agent.choose(A) for _ in range(20)]
    assert again.to_dict() == agent.to_dict()
    with pytest.raises(ConfigurationError, match="different controls"):
        SkillsLearner.from_dict(data, space=ActionSpace([frozenset()]))
    with pytest.raises(ConfigurationError, match="newer"):
        SkillsLearner.from_dict({**data, "schema": 99})
    with pytest.raises(ConfigurationError, match="skills"):
        SkillsLearner.from_dict({**data, "algorithm": Q_TABLE})
    with pytest.raises(ConfigurationError):
        SkillsLearner.from_dict({**data, "skills": "nope"})
    with pytest.raises(ConfigurationError):
        SkillsLearner.from_dict({**data, "skills": [{"id": "x"}]})
    as_mapping = {**data, "skills": {s["id"]: s["learner"] for s in data["skills"]}}
    assert SkillsLearner.from_dict(as_mapping, space=SPACE).skills == agent.skills
    # The inner kind can be a network.
    deep = SkillsLearner(SPACE, seed=5, inner=DQN)
    assert isinstance(deep.set_skill("run"), DQNLearner)
    deep.learn(A, RIGHT, 1.0, B)
    back = SkillsLearner.from_dict(json.loads(json.dumps(deep.to_dict())), space=SPACE)
    assert back.inner == DQN and isinstance(back.skill_learner("run"), DQNLearner)


def test_convert_wraps_and_unwraps() -> None:
    table = QLearningAgent(SPACE, Settings(epsilon=0.2), seed=6)
    table.learn(A, RIGHT, 1.0, B)
    table.end_episode()
    wrapped = convert(table, SKILLS)
    assert isinstance(wrapped, SkillsLearner) and wrapped.skills == (DEFAULT_SKILL,)
    assert wrapped.skill_learner(DEFAULT_SKILL) is table
    assert wrapped.updates == 1 and wrapped.episodes == 1 and wrapped.epsilon == table.epsilon
    assert convert(wrapped, SKILLS) is wrapped
    wrapped.set_skill("jump").learn(A, LEFT, 1.0, B)
    unwrapped = convert(wrapped, Q_TABLE)
    assert unwrapped is table  # the default skill, the other skills dropped
    deep = convert(wrapped, DQN, seed=7)
    assert isinstance(deep, DQNLearner) and deep.policy() == table.policy()
    assert deep.episodes == 1
