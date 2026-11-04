"""T-807: research mode - step log, situation values, learning settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.learning import (
    DQN,
    Q_TABLE,
    SKILLS,
    Decision,
    DQNLearner,
    QLearningAgent,
    Settings,
    SkillsLearner,
)
from retro_ai_lab.learning.greedy import GreedyPlayer
from retro_ai_lab.training import (
    COLUMNS,
    Brain,
    Mode,
    SituationReport,
    StepLog,
    StepRecord,
    TrainingStepped,
    situation_values,
)
from tests.unit.learning.test_learner import SPACE, A, B
from tests.unit.training.test_service import Lab

# ------------------------------------------------------------ the step log


def test_log_keeps_every_step_of_the_goes(lab: Lab) -> None:
    log = StepLog(lab.training, lab.bus)
    lab.connect()
    records = lab.training.train(2)
    assert len(log) == sum(r.result.steps for r in records) and log.dropped == 0
    by_go = {n: [s for s in log if s.go == n] for n in (1, 2)}
    assert [s.step for s in by_go[1]] == list(range(1, records[0].result.steps + 1))
    assert [s.step for s in by_go[2]] == list(range(1, records[1].result.steps + 1))
    assert all(s.frame > 0 and "=" in s.situation for s in log)
    assert all(s.explored is not None and s.value is not None for s in log)  # the AI chose
    for n, record in zip((1, 2), records, strict=True):
        assert sum(s.reward for s in by_go[n]) == pytest.approx(record.result.reward)
    buttons = {s.buttons for s in log}
    assert len(buttons) > 1  # it pressed different things
    assert all(s == "+".join(sorted(s.split("+"))) and " " not in s for s in buttons - {""})
    frames = [s.frame for s in by_go[1]]
    assert frames == sorted(frames)


def test_manual_steps_are_logged_without_a_decision(lab: Lab) -> None:
    log = StepLog(lab.training, lab.bus)
    lab.connect()
    t = lab.training
    t.start(Mode.MANUAL)
    for _ in range(3):
        lab.play(["a"], 4)
    t.stop()
    assert len(log) >= 1
    assert all(s.explored is None and s.value is None and s.buttons == "a" for s in log)
    assert all(s.go == 1 for s in log)


def test_log_is_bounded_and_empties_with_the_brain(lab: Lab) -> None:
    with pytest.raises(ConfigurationError, match="at least one"):
        StepLog(lab.training, lab.bus, limit=0)
    log = StepLog(lab.training, lab.bus, limit=5)
    lab.connect()
    lab.training.train(1)
    steps = lab.training.last.result.steps if lab.training.last else 0
    assert steps > 5 and len(log) == 5 and log.limit == 5
    assert log.dropped == steps - 5
    assert [s.step for s in log] == list(range(steps - 4, steps + 1))  # the newest stay
    lab.training.forget()
    assert len(log) == 0 and log.dropped == 0
    lab.training.train(1)
    assert len(log) == 5 and all(s.go == 1 for s in log)  # go numbers restart


def test_log_round_trips_through_csv(lab: Lab, tmp_path: Path) -> None:
    log = StepLog(lab.training, lab.bus)
    bus = EventBus()
    empty = StepLog(lab.training, bus)
    path = tmp_path / "logs" / "steps.csv"
    assert empty.write_csv(path) == 0
    assert path.read_text(encoding="utf-8").strip() == ",".join(COLUMNS)
    assert StepLog.read_csv(path) == ()
    lab.connect()
    lab.training.train(1)
    log.add(StepRecord(2, 1, 99, "", "", None, None, -0.5))  # a childish step, no situation
    assert log.write_csv(path) == len(log)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(COLUMNS) and len(lines) == len(log) + 1
    assert StepLog.read_csv(path) == log.records
    (tmp_path / "other.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not a step log"):
        StepLog.read_csv(tmp_path / "other.csv")
    with pytest.raises(ConfigurationError, match="cells"):
        StepRecord.from_row(("1", "2"))
    with pytest.raises(ConfigurationError, match="Bad step row"):
        StepRecord.from_row(("x", "2", "3", "", "", "", "", "0"))


def test_records_come_from_the_event() -> None:
    event = TrainingStepped(
        40, frozenset({"b", "a"}), Decision(1, True, 0.25), step=3, reward=1.5, situation="x=1"
    )
    record = StepRecord.from_event(7, event)
    assert record == StepRecord(7, 3, 40, "x=1", "a+b", True, 0.25, 1.5)
    assert record.row() == ("7", "3", "40", "x=1", "a+b", "yes", "0.25", "1.5")
    assert StepRecord.from_row(record.row()) == record
    plain = TrainingStepped(1, frozenset(), None)  # the older three-field form still works
    assert StepRecord.from_event(1, plain) == StepRecord(1, 0, 1, "", "", None, None, 0.0)


# ------------------------------------------------------------ situation values


def _learned(learner: QLearningAgent | DQNLearner | SkillsLearner) -> None:
    for _ in range(30):
        learner.learn(A, 1, 1.0, B)
        learner.learn(A, 0, -1.0, B)


def test_situation_values_for_every_learner() -> None:
    table = QLearningAgent(SPACE, Settings(epsilon=0.0, epsilon_min=0.0), seed=1)
    assert tuple((v.label, v.value, v.best) for v in situation_values(table, A)) == tuple(
        (label, 0.0, True) for label in SPACE.labels
    )
    _learned(table)
    values = situation_values(table, A)
    assert [v.label for v in values] == list(SPACE.labels)
    assert [v.best for v in values] == [i == 1 for i in range(len(SPACE))]
    assert values[1].value > 0 > values[0].value
    report = SituationReport.of(table, A)
    assert report.best() == (SPACE.labels[1],)
    assert report.describe().startswith("In the situation ") and "(best)" in report.describe()
    assert report.describe().count("(best)") == 1
    greedy = GreedyPlayer(table)
    assert situation_values(greedy, A) == values
    skills = SkillsLearner(SPACE, Settings(epsilon=0.0, epsilon_min=0.0), seed=1)
    skills.adopt("step1", table)
    skills.set_skill("step1")
    assert situation_values(skills, A) == values
    skills.set_skill(None)
    assert all(v.value == 0.0 for v in situation_values(skills, A))
    dqn = DQNLearner(SPACE, Settings(epsilon=0.0, epsilon_min=0.0), seed=1)
    dqn_values = situation_values(dqn, A)
    assert len(dqn_values) == len(SPACE) and any(v.best for v in dqn_values)
    assert tuple(v.value for v in dqn_values) == tuple(float(x) for x in dqn.values(A))


def test_current_state_follows_the_game(lab: Lab) -> None:
    t = lab.training
    assert t.current_state() is None
    lab.connect()
    assert t.current_state() is None  # nothing observed yet
    lab.play([], 1)
    state = t.current_state()
    assert state is not None and "=" in state.key
    assert t.agent is not None
    report = SituationReport.of(t.agent, state)
    assert len(report.values) == len(t.agent.space)
    lab.connection.disconnect()
    assert t.current_state() is None


# ------------------------------------------------------------ learning settings


def test_settings_apply_to_the_brain_in_use_and_survive_a_reload(lab: Lab) -> None:
    t = lab.training
    assert t.settings == Settings(epsilon=0.5)  # what the lab was built with
    with pytest.raises(ConfigurationError, match="must be a Settings"):
        t.set_settings({"alpha": 0.5})  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="between"):
        Settings(alpha=1.5)
    quick = Settings(alpha=0.5, gamma=0.5, epsilon=0.3, epsilon_min=0.1, epsilon_decay=0.9)
    t.set_settings(quick)
    assert t.settings == quick and lab.changes[-1].reason == "settings"
    lab.connect()
    agent = t.agent
    assert isinstance(agent, QLearningAgent) and agent.settings == quick  # new brains follow
    assert agent.epsilon == 0.3
    t.train(1)
    # The per-step line caps exploring during the go; the per-go decay then multiplies it (T-875).
    assert agent.epsilon == pytest.approx(0.9 * quick.exploring_after(agent.updates))
    t.start()
    t.step()
    with pytest.raises(ConfigurationError, match="Stop training"):
        t.set_settings(Settings())
    t.stop()
    steady = Settings(alpha=0.9, gamma=0.1, epsilon=0.0, epsilon_min=0.0)
    saves = t.saves
    t.set_settings(steady)
    assert t.agent is agent and agent.settings == steady and agent.epsilon == 0.0
    assert t.saves == saves + 1 and lab.changes[-1].reason == "settings"
    assert t.store is not None and t.game_id is not None
    brain = Brain.from_dict(t.store.load(t.game_id))
    assert isinstance(brain.agent, QLearningAgent) and brain.agent.settings == steady
    # The new rate is what it learns with: one lesson moves the value 90% of the way.
    state = t.current_state()
    assert state is not None
    before = agent.values(state)[0]
    agent.learn(state, 0, 10.0, state, done=True)
    assert agent.values(state)[0] == pytest.approx(before + 0.9 * (10.0 - before))


def test_settings_reach_every_kind_of_learner(lab: Lab) -> None:
    t = lab.training
    fresh = Settings(alpha=0.2, epsilon=0.5, epsilon_min=0.05)
    t.set_algorithm(SKILLS)
    lab.connect()
    skills = t.agent
    assert isinstance(skills, SkillsLearner)
    skills.set_skill("later")
    t.set_settings(fresh)
    assert skills.settings == fresh
    assert all(skills.skill_learner(s).settings == fresh for s in skills.skills)  # type: ignore[attr-defined]
    assert skills.set_skill("newer").settings == fresh  # type: ignore[attr-defined]
    assert t.set_algorithm(DQN) and isinstance(t.agent, DQNLearner)
    t.set_settings(Settings(epsilon=0.4, epsilon_min=0.0))
    assert t.agent.settings.epsilon == 0.4 and t.agent.epsilon == 0.4
    t.set_algorithm(Q_TABLE)
    assert isinstance(t.agent, QLearningAgent) and t.agent.settings.epsilon == 0.4
    for learner in (QLearningAgent(SPACE), DQNLearner(SPACE), SkillsLearner(SPACE)):
        with pytest.raises(ConfigurationError, match="must be a Settings"):
            learner.configure("fast")  # type: ignore[arg-type]
    greedy = GreedyPlayer(QLearningAgent(SPACE))
    greedy.configure(fresh)
    assert greedy.learner.settings == Settings()  # type: ignore[attr-defined]
