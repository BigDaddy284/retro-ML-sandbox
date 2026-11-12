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
        dict(rebound=1.5),
        dict(rebound=-0.1),
        dict(explore_steps="many"),
        dict(explore_run=0),
        dict(explore_run=2.5),
        dict(explore_run=101),
        dict(explore_run="long"),
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


def test_exploring_rises_again_when_the_goes_get_worse() -> None:
    """T-932: the last ten goes worse than the ten before lifts exploring to the rebound."""
    quick = Settings(epsilon=1.0, epsilon_min=0.05, epsilon_decay=1.0, explore_steps=10)
    agent = QLearningAgent(SPACE, quick, seed=1)
    for _ in range(10):
        agent.learn(A, RIGHT, 0.0, B)
    assert agent.epsilon == 0.05 and agent.lift == 0.0
    for _ in range(10):
        assert agent.end_episode(10.0) == 0.05  # ten good goes: nothing to compare with yet
    for _ in range(9):
        assert agent.end_episode(0.0) == 0.05  # not judged until a whole window is worse
    assert agent.end_episode(0.0) == 0.25 and agent.lift == 0.25  # ten worse goes: back up
    assert agent.recent_rewards == (10.0,) * 10 + (0.0,) * 10
    agent.learn(A, RIGHT, 0.0, B)
    assert agent.epsilon == 0.25  # the schedule's floor no longer pulls it down
    assert agent.end_episode(0.0) == 0.25  # still worse than the window before: held there
    for _ in range(9):
        agent.end_episode(0.0)
    # Once the latest window is no longer worse than the one before (all poor now), the
    # lift eases off by a window's share each go: two such goes so far.
    assert agent.lift == pytest.approx(0.2)
    for _ in range(10):
        agent.end_episode(10.0)  # ten better goes: the lift is gone
    assert agent.lift == 0.0 and agent.epsilon == 0.05
    agent.end_episode()  # a go with no reward given is counted and judged on nothing
    assert agent.episodes == 41 and agent.recent_rewards[-1] == 10.0
    # A brain keeps its lift and its last goes.
    agent.end_episode(0.0)
    data = json.loads(json.dumps(agent.to_dict()))
    again = QLearningAgent.from_dict(data, space=SPACE)
    assert again.lift == agent.lift and again.recent_rewards == agent.recent_rewards
    del data["results"]  # a brain saved before T-932
    assert QLearningAgent.from_dict(data, space=SPACE).recent_rewards == ()
    # Rebound 0 turns it off; the setting is in the defaults and the saved settings.
    off = QLearningAgent(SPACE, Settings(epsilon_decay=1.0, explore_steps=10, rebound=0.0))
    for _ in range(10):
        off.learn(A, RIGHT, 0.0, B)
    for reward in (10.0,) * 10 + (0.0,) * 10:
        off.end_episode(reward)
    assert off.epsilon == 0.05 and off.lift == 0.0
    assert Settings().rebound == 0.25 and Settings().to_dict()["rebound"] == 0.25
    agent.reset()
    assert agent.lift == 0.0 and agent.recent_rewards == ()


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


def test_the_future_counts_ninety_nine_hundredths_a_step() -> None:
    """T-962: a horizon of about 100 steps, so a death seconds later still reaches its cause."""
    from retro_ai_lab.learning.qlearning import OLD_GAMMA

    assert Settings().gamma == 0.99 and OLD_GAMMA == 0.95
    assert 0.99**45 * 20 > 12 > 0.95**45 * 20  # the user's log: the life counter drops 45 steps on


def test_a_brain_saved_with_the_old_default_gamma_looks_further_ahead_when_loaded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-962: the user's brain was saved with 0.95; picking it up moves it on and says so."""
    import logging

    agent = QLearningAgent(SPACE, Settings(gamma=0.95, epsilon=0.0, epsilon_min=0.0), seed=1)
    agent.learn(A, RIGHT, 1.0, B)
    data = json.loads(json.dumps(agent.to_dict()))
    assert data["settings"]["gamma"] == 0.95
    caplog.set_level(logging.INFO)
    again = QLearningAgent.from_dict(data, space=SPACE)
    assert again.settings.gamma == 0.99 and again.settings.epsilon == 0.0
    assert "looks further ahead: the future counts 0.99 a step instead of 0.95" in caplog.text
    # The update bootstraps with the new gamma: 0.1 * (0 + 0.99 * Q(B, best)) from a zero row.
    best_b = again.learn(B, RIGHT, 10.0, A)  # 0.1 * (10 + 0.99 * Q(A, RIGHT) of 0.1)
    assert best_b == pytest.approx(0.1 * (10.0 + 0.99 * 0.1))
    assert again.learn(A, LEFT, 0.0, B) == pytest.approx(0.1 * 0.99 * best_b)
    # A brain saved with no settings at all gets the default too, quietly.
    caplog.clear()
    del data["settings"]
    assert QLearningAgent.from_dict(data, space=SPACE).settings.gamma == 0.99
    assert "looks further ahead" not in caplog.text


def test_a_chosen_gamma_survives_a_save_and_load(caplog: pytest.LogCaptureFixture) -> None:
    """T-962: any other value was the person's choice in the Research window."""
    import logging

    caplog.set_level(logging.INFO)
    for gamma in (0.9, 0.5, 0.99, 1.0):
        agent = QLearningAgent(SPACE, Settings(gamma=gamma), seed=1)
        again = QLearningAgent.from_dict(json.loads(json.dumps(agent.to_dict())), space=SPACE)
        assert again.settings.gamma == gamma
    assert "looks further ahead" not in caplog.text


def test_a_q_update_bootstraps_over_the_steps_it_spans() -> None:
    """T-963: a return over three steps discounts the future by gamma cubed."""
    agent = QLearningAgent(SPACE, Settings(alpha=0.5, gamma=0.9, epsilon=0.0, epsilon_min=0.0))
    agent.learn(B, RIGHT, 2.0, A)  # Q(B, RIGHT) = 1.0
    assert agent.learn(A, LEFT, 0.6, B, steps=3) == pytest.approx(0.5 * (0.6 + 0.9**3 * 1.0))
    assert agent.learn(A, RIGHT, 0.6, B, done=True, steps=3) == pytest.approx(0.5 * 0.6)
    for bad in (0, -2, 1.5, True):
        with pytest.raises(ConfigurationError):
            agent.learn(A, LEFT, 0.0, B, steps=bad)  # type: ignore[arg-type]


def test_a_fine_situation_starts_from_what_the_coarse_one_knew() -> None:
    """T-982: looking closer (T-979) splits a bucket's knowledge instead of throwing it
    away; the AI had learned the stretch before the trap over thousands of goes."""
    from retro_ai_lab.learning.state import fine_bucket
    from retro_ai_lab.understanding import VariableKind

    t = QTable(2)
    coarse = State((("v1", 12), ("v2", 3)))
    t.row(coarse)[1] = 4.0
    fine = State((("v1", fine_bucket(130, VariableKind.POSITION)), ("v2", 3)))
    assert t.values(fine) == (0.0, 4.0) and t.best(fine) == 1 and len(t) == 1  # read only
    row = t.row(fine)  # the first learn copies it
    assert row == [0.0, 4.0] and len(t) == 2
    row[0] = 9.0
    assert t.values(coarse) == (0.0, 4.0)  # a copy, not a share
    other = State((("v1", fine_bucket(70, VariableKind.POSITION)), ("v2", 3)))
    assert t.values(other) == (0.0, 0.0)  # bucket 11 was never learned
    assert t.values(State((("v1", 12), ("v2", 5)))) == (0.0, 0.0)  # a coarse unknown: nothing


def _streaks(agent: QLearningAgent, state: State, steps: int) -> tuple[int, list[list[int]]]:
    """Explored steps, and the moves of each unbroken stretch of them."""
    explored, streaks, running = 0, [], False
    for _ in range(steps):
        action = agent.choose(state)
        decision = agent.last_decision
        assert decision is not None and decision.action == action
        if decision.explored:
            explored += 1
            if running:
                streaks[-1].append(action)
            else:
                streaks.append([action])
        running = decision.explored
    return explored, streaks


def test_the_table_learner_explores_in_runs() -> None:
    """T-983: a random move is held for a run of steps; a run of one is the old exploring."""
    assert Settings().explore_run == 8 and Settings().to_dict()["explore_run"] == 8
    assert Settings(explore_run=3.0).explore_run == 3  # a whole number as a float
    fixed = dict(epsilon=0.3, epsilon_min=0.3, epsilon_decay=1.0, rebound=0.0)
    # Runs of one: the same draws, in the same order, as before the change.
    single = QLearningAgent(SPACE, Settings(explore_run=1, **fixed), seed=7)
    rng, table = Random(7), QTable(2)
    before = [rng.randrange(2) if rng.random() < 0.3 else table.best(A, rng) for _ in range(300)]
    assert [single.choose(A) for _ in range(300)] == before
    assert single.stats().endswith("tries something new 30% of the time.")
    # Runs of up to eight: the share explored is still 30%, in stretches instead of twitches.
    agent = QLearningAgent(SPACE, Settings(**fixed), seed=7)
    agent.table.row(A)[RIGHT] = 1.0  # what looks best is always RIGHT
    assert agent.stats().endswith(
        "tries something new 30% of the time, holding each try for up to 8 steps."
    )
    steps = 20_000
    explored, streaks = _streaks(agent, A, steps)
    assert abs(explored / steps - 0.3) < 0.02
    assert explored / len(streaks) > 4.0  # a new move every step averaged 1.4
    held_left = [s for s in streaks if len(s) >= 4 and set(s) == {LEFT}]
    assert held_left  # LEFT, never the best move here, is held for four steps and more
    one, _ = _streaks(QLearningAgent(SPACE, Settings(explore_run=1, **fixed), seed=7), A, steps)
    assert abs(one / steps - 0.3) < 0.02
    # A go's end, new settings and a reset each drop the run in hand.
    nothing = {"left": 0, "action": 0}
    for drop in (
        lambda a: a.end_episode(),
        lambda a: a.configure(Settings(**fixed)),
        lambda a: a.reset(),
    ):
        while agent.to_dict()["run"]["left"] == 0:
            agent.choose(A)
        drop(agent)
        assert agent.to_dict()["run"] == nothing
    # Saved part-way through a run and loaded, it carries on with the same moves.
    while agent.to_dict()["run"]["left"] < 2:
        agent.choose(A)
    data = json.loads(json.dumps(agent.to_dict()))
    again = QLearningAgent.from_dict(data, space=SPACE)
    assert [again.choose(A) for _ in range(100)] == [agent.choose(A) for _ in range(100)]
    # A brain saved before runs existed explores in runs of eight; a bad saved run is refused.
    del data["settings"]["explore_run"], data["run"]
    old = QLearningAgent.from_dict(data, space=SPACE)
    assert old.settings.explore_run == 8 and old.to_dict()["run"] == nothing
    for bad in ({"left": 8, "action": 0}, {"left": 1, "action": 2}):
        data["run"] = bad
        with pytest.raises(ConfigurationError):
            QLearningAgent.from_dict(data, space=SPACE)
