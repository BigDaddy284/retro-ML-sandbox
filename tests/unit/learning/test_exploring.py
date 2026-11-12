"""T-932: exploring follows results. T-983: exploring comes in runs."""

from __future__ import annotations

import json
import logging
from random import Random

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings, State
from retro_ai_lab.learning.exploring import (
    MOST_RUN,
    NOTICEABLE,
    RESULTS_WINDOW,
    ResultsWatch,
    Runs,
)


def test_lift_rises_when_the_latest_window_is_worse_and_eases_off_after() -> None:
    watch = ResultsWatch(0.25, window=3)
    assert watch.lift == 0.0 and watch.slipping() is None
    for reward in (9.0, 9.0, 9.0, 9.0, 9.0):
        assert watch.note(reward, goes=1) == 0.0
    assert watch.slipping() is None  # six needed for two windows
    # (9, 9, 9) then (9, 9, 3): worse by 2 on a spread of 6, more than NOTICEABLE's tenth.
    assert watch.note(3.0, goes=6) == 0.25
    assert watch.rewards == (9.0, 9.0, 9.0, 9.0, 9.0, 3.0)
    assert watch.slipping() == (7.0, 9.0)
    # The last two windows together are kept, never more.
    assert watch.note(3.0, goes=7) == 0.25 and watch.rewards == (9.0, 9.0, 9.0, 9.0, 3.0, 3.0)
    assert watch.note(3.0, goes=8) == 0.25
    # Recovered: the lift falls by a window's share each go that is not worse.
    assert watch.note(9.0, goes=9) == 0.25  # (9, 3, 3) vs (3, 3, 9): the same
    assert watch.note(9.0, goes=10) == pytest.approx(0.25 - 0.25 / 3)
    assert watch.note(9.0, goes=11) == pytest.approx(0.25 - 2 * 0.25 / 3)
    assert watch.note(9.0, goes=12) == 0.0
    assert watch.note(9.0, goes=13) == 0.0  # never below zero


def test_small_drops_are_noise_and_rebound_zero_turns_it_off() -> None:
    watch = ResultsWatch(0.5, window=2)
    for reward in (10.0, 0.0, 9.0, 0.0):
        watch.note(reward, goes=1)
    assert watch.slipping() is None  # 0.5 down on a spread of 10 is within NOTICEABLE's tenth
    assert watch.lift == 0.0
    assert NOTICEABLE == 0.1 and RESULTS_WINDOW == 10
    flat = ResultsWatch(0.5, window=2)
    for reward in (1.0, 1.0, 1.0, 1.0):
        assert flat.note(reward, goes=1) == 0.0  # no spread: nothing is worse
    off = ResultsWatch(0.0, window=2)
    for reward in (10.0, 10.0, 0.0, 0.0):
        assert off.note(reward, goes=1) == 0.0
    assert off.slipping() == (0.0, 10.0)  # it still sees the drop; it just does nothing


def test_rebound_can_change_and_the_watch_round_trips() -> None:
    watch = ResultsWatch(0.25, window=2)
    for reward in (10.0, 10.0, 0.0, 0.0):
        watch.note(reward, goes=1)
    assert watch.lift == 0.25
    watch.set_rebound(0.1)
    assert watch.lift == 0.1  # a lower rebound caps the lift at once
    again = ResultsWatch(0.1, window=2)
    again.load(watch.to_dict())
    assert again.lift == 0.1 and again.rewards == watch.rewards
    again.load(None)  # a brain saved before T-932 has nothing to load
    assert again.rewards == watch.rewards
    with pytest.raises(ConfigurationError):
        ResultsWatch(0.25, window=0)
    with pytest.raises(ConfigurationError):
        again.load({"rewards": "lots"})
    with pytest.raises(ConfigurationError):
        again.load({"rewards": [], "hold": 3})  # more goes of hold than the window has


def test_the_rise_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    watch = ResultsWatch(0.3, window=2)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.learning.exploring"):
        for go, reward in enumerate((10.0, 10.0, 0.0, 0.0, 0.0), start=1):
            watch.note(reward, goes=go)
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [
        "Exploring back up to 30% after go 4: the last 2 goes averaged +0.0 points, "
        "down from +10.0."
    ]


def _move(rng: Random) -> int:
    return rng.randrange(6)


def test_runs_of_one_are_the_old_exploring() -> None:
    """T-983: a run of one is a roll and a draw a step, exactly as the learners did it."""
    runs = Runs(1)
    new, old = Random(5), Random(5)
    for epsilon in (0.0, 0.05, 0.3, 1.0):
        assert runs.chance(epsilon) == epsilon
        for _ in range(500):
            before = old.randrange(6) if old.random() < epsilon else None
            assert runs.next(new, epsilon, _move) == before
            assert runs.left == 0
    assert new.getstate() == old.getstate()  # draw for draw


def test_a_run_holds_one_move_and_keeps_the_explored_share() -> None:
    """T-983: a random move is held for 1 to 8 steps, and the explored share stays epsilon."""
    assert Runs().most == 1 and Runs(8).most == 8
    assert Runs(8).chance(0.0) == 0.0 and Runs(8).chance(1.0) == 1.0
    assert Runs(8).chance(0.05) == pytest.approx(0.05 / (4.5 * 0.95 + 0.05))
    steps = 60_000
    for epsilon in (0.05, 0.25, 1.0):
        runs, rng = Runs(8), Random(11)
        explored, lengths, held = 0, [], -1
        for _ in range(steps):
            before = runs.left
            move = runs.next(rng, epsilon, _move)
            if move is None:
                assert before == 0  # a run is never cut short
                continue
            explored += 1
            if before:
                assert move == held and runs.left == before - 1
                lengths[-1] += 1
            else:
                held = move
                lengths.append(1)
        assert abs(explored / steps - epsilon) < 0.01
        assert set(lengths) == set(range(1, 9))
        assert sum(lengths) / len(lengths) == pytest.approx(4.5, abs=0.15)
    assert explored == steps  # at 100% every step is explored
    # A go's end drops the run in hand; so does a new length.
    runs, rng = Runs(8), Random(2)
    while runs.left == 0:
        runs.next(rng, 1.0, _move)
    runs.drop()
    assert runs.left == 0 and runs.to_dict() == {"left": 0, "action": 0}
    while runs.left == 0:
        runs.next(rng, 1.0, _move)
    runs.set_most(4)
    assert runs.most == 4 and runs.left == 0


def test_a_run_is_saved_and_a_bad_run_is_refused() -> None:
    """T-983: a run saved part-way carries on with the same moves; nonsense is refused."""
    for bad in (0, -1, MOST_RUN + 1, 2.5, True, "long"):
        with pytest.raises(ConfigurationError):
            Runs(bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        Runs(8).set_most(0)
    runs, rng = Runs(8), Random(3)
    while runs.left < 2:
        runs.next(rng, 1.0, _move)
    saved = json.loads(json.dumps(runs.to_dict()))
    assert saved["left"] == runs.left and 0 <= saved["action"] < 6
    again, rng2 = Runs(8), Random()
    rng2.setstate(rng.getstate())
    again.load(saved, actions=6)
    assert [again.next(rng2, 0.3, _move) for _ in range(200)] == [
        runs.next(rng, 0.3, _move) for _ in range(200)
    ]
    fresh = Runs(8)
    fresh.load(None, actions=6)
    fresh.load({}, actions=6)
    assert fresh.left == 0
    for bad_run in (
        {"left": 8, "action": 0},  # a run of 8 has at most 7 steps left
        {"left": -1, "action": 0},
        {"left": 2, "action": 6},  # a move these controls don't have
        {"left": 2, "action": -1},
    ):
        with pytest.raises(ConfigurationError):
            Runs(8).load(bad_run, actions=6)
    with pytest.raises(ConfigurationError):
        Runs(1).load({"left": 1, "action": 0}, actions=6)  # runs of one never have steps left


STAY, WALK, JUMP, BACK = range(4)
CORRIDOR = ActionSpace([frozenset(), frozenset({"right"}), frozenset({"a"}), frozenset({"left"})])
END, GAPS, HOLD = 14, (4, 9), 4  # squares; a gap is cleared by holding jump for HOLD steps


def _first_clear(agent: QLearningAgent, goes: int) -> int | None:
    """The go on which the agent first reaches the end of the corridor, if it does.

    Walking into a gap, or letting go of jump over one, ends the go at -20. Each new
    square is worth 1 and the end 10. Nothing but holding one move gets past a gap.
    """
    for go in range(1, goes + 1):
        x = air = furthest = 0
        total = 0.0
        for _ in range(100):
            state = State((("x", x), ("air", air)))
            action = agent.choose(state)
            reward, done = 0.0, False
            if air:
                if action == JUMP:
                    air += 1
                    if air >= HOLD:
                        x, air = (x + 2 if x + 1 in GAPS else x + 1), 0
                elif x + 1 in GAPS:
                    reward, done = -20.0, True
                else:
                    air = 0
            elif action == JUMP:
                air = 1
            elif action == WALK:
                if x + 1 in GAPS:
                    reward, done = -20.0, True
                else:
                    x += 1
            elif action == BACK:
                x = max(0, x - 1)
            if not done and x > furthest:
                reward += float(x - furthest)
                furthest = x
            if not done and x >= END:
                reward, done = reward + 10.0, True
            agent.learn(state, action, reward, State((("x", x), ("air", air))), done)
            total += reward
            if done:
                break
        agent.end_episode(total)
        if x >= END:
            return go
    return None


def test_runs_find_a_move_that_only_pays_when_held() -> None:
    """T-983: a gap needs jump held four steps; runs find the way across, twitches rarely."""
    goes, seeds = 300, range(10)
    in_runs = [
        _first_clear(QLearningAgent(CORRIDOR, Settings(), seed=seed), goes) for seed in seeds
    ]
    one_step = [
        _first_clear(QLearningAgent(CORRIDOR, Settings(explore_run=1), seed=seed), goes)
        for seed in seeds
    ]
    assert all(go is not None for go in in_runs)  # every seed crosses both gaps
    assert sum(go is not None for go in one_step) <= len(seeds) // 2  # half at most, without
