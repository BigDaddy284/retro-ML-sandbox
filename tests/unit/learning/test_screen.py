"""T-1055: the screen network, a learner fed the picture of the screen."""

from __future__ import annotations

import base64
import itertools
import json
from random import Random
from typing import Any

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import (
    NO_MOVE,
    PICTURE_HEIGHT,
    PICTURE_SIZE,
    PICTURE_WIDTH,
    SCREEN,
    ActionSpace,
    Decision,
    DQNLearner,
    Learner,
    QLearningAgent,
    ScreenConfig,
    ScreenLearner,
    Settings,
    Sight,
    State,
    algorithm_choices,
    algorithms,
    describe_algorithm,
    learner_from_dict,
    make_learner,
)
from retro_ai_lab.learning.screen import inputs

LEFT, RIGHT = 0, 1
SPACE = ActionSpace([frozenset({"left"}), frozenset({"right"})])
DARK = bytes([20]) * PICTURE_SIZE
BRIGHT = bytes([200]) * PICTURE_SIZE
QUICK = ScreenConfig(
    hidden=(16,), batch_size=8, warmup=8, target_sync=20, learning_rate=0.003, memory=200
)
# The toy problems below are learnt with two small layers, on every seed from 1 to 10.
TOY = ScreenConfig(
    hidden=(32, 16), batch_size=8, warmup=8, target_sync=20, learning_rate=0.002, memory=200
)
STEADY = dict(epsilon=0.3, epsilon_min=0.3, epsilon_decay=1.0, rebound=0.0)


def seen(now: bytes, before: bytes | None = None, held: int = NO_MOVE) -> State:
    """A situation that shows a picture; its parts say nothing."""
    return State((), Sight(now, now if before is None else before, held))


A = seen(DARK)
B = seen(BRIGHT)
BLIND = State((("x", 1),))


def quick(seed: int = 1, **settings: Any) -> ScreenLearner:
    return ScreenLearner(SPACE, Settings(**{**STEADY, **settings}), seed=seed, config=QUICK)


def toy(seed: int, **settings: Any) -> ScreenLearner:
    return ScreenLearner(SPACE, Settings(**{**STEADY, **settings}), seed=seed, config=TOY)


def start_a_run(agent: ScreenLearner) -> None:
    """Choose until a try is being held for some steps more."""
    for _ in range(100):
        agent.choose(A)
        if agent._runs.to_dict()["left"] > 0:
            return
    raise AssertionError("no run of tries began in a hundred choices")


def spy_on_training(agent: ScreenLearner, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every (inputs, targets, mask) the network is trained on."""
    calls: list[Any] = []
    real = agent.network.train_step

    def train_step(x: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
        calls.append((x.copy(), targets.copy(), mask.copy()))
        return real(x, targets, mask)

    monkeypatch.setattr(agent.network, "train_step", train_step)
    return calls


# ----------------------------------------------------------- what it is, and its name


def test_it_is_a_learner_the_app_can_pick() -> None:
    assert SCREEN == "screen" and "screen" in algorithms() and "screen" in algorithm_choices()
    label, sentence = describe_algorithm("screen")
    assert label == "Screen network"
    assert sentence.startswith("Watches the screen itself, shrunk to a small picture")
    assert "Needs nothing found in the game's memory" in sentence
    agent = make_learner("screen", SPACE, Settings(epsilon=0.2), seed=3)
    assert isinstance(agent, ScreenLearner) and isinstance(agent, Learner)
    assert agent.algorithm == "screen" and agent.wants_picture is True
    assert agent.epsilon == 0.2 and agent.space == SPACE and agent.config == ScreenConfig()
    assert agent.settings == Settings(epsilon=0.2)
    assert (agent.known, agent.updates, agent.episodes, agent.trained, agent.unseen) == (0,) * 5
    assert agent.last_decision is None and agent.policy() == {}
    assert agent.network.sizes == (2 * PICTURE_SIZE + 2, 128, 64, 2)
    assert agent.target.sizes == agent.network.sizes and agent.target is not agent.network
    action = agent.choose(A)
    assert action in (LEFT, RIGHT) and isinstance(agent.last_decision, Decision)
    assert agent.last_decision.action == action
    # The table and the first network do not ask for pictures.
    assert not getattr(QLearningAgent(SPACE), "wants_picture", False)
    assert not getattr(DQNLearner(SPACE), "wants_picture", False)


def test_its_settings_are_checked() -> None:
    assert ScreenConfig().hidden == (128, 64) and ScreenConfig().memory == 30_000
    assert ScreenConfig(hidden=[8, 4]).hidden == (8, 4)  # type: ignore[arg-type]
    for bad in (
        {"hidden": ()},
        {"hidden": (0,)},
        {"hidden": (8, True)},
        {"batch_size": 0},
        {"warmup": 4, "batch_size": 8},
        {"target_sync": 0},
        {"memory": 400},  # below the warm-up of 500
        {"memory": 0},
        {"learning_rate": 0.0},
        {"learning_rate": 1.0},
        {"learning_rate": "fast"},
        {"reward_scale": 0.0},
        {"reward_scale": 1.5},
        {"reward_scale": True},
        {"reward_scale": "small"},
    ):
        with pytest.raises(ConfigurationError):
            ScreenConfig(**bad)  # type: ignore[arg-type]
    assert ScreenConfig(reward_scale=1).reward_scale == 1
    assert ScreenConfig.from_dict(QUICK.to_dict()) == QUICK
    own = ScreenConfig(
        hidden=(8, 4),
        batch_size=4,
        warmup=6,
        target_sync=7,
        learning_rate=0.01,
        memory=50,
        reward_scale=0.5,
    )
    assert own.to_dict() == {
        "hidden": [8, 4],
        "batch_size": 4,
        "warmup": 6,
        "target_sync": 7,
        "learning_rate": 0.01,
        "memory": 50,
        "reward_scale": 0.5,
    }
    assert ScreenConfig.from_dict(own.to_dict()) == own  # nothing of it falls back
    assert ScreenConfig.from_dict({}) == ScreenConfig()
    assert ScreenConfig.from_dict({"hidden": [32]}) == ScreenConfig(hidden=(32,))
    for bad_data in ({"hidden": 5}, {"batch_size": "many"}, {"hidden": ["a"]}, "nope"):
        with pytest.raises(ConfigurationError):
            ScreenConfig.from_dict(bad_data)  # type: ignore[arg-type]
    agent = quick()
    with pytest.raises(ConfigurationError, match="must be a Settings"):
        agent.configure({"epsilon": 0.1})  # type: ignore[arg-type]
    agent.configure(Settings(epsilon=0.6, explore_run=2))
    assert agent.epsilon == 0.6 and agent.settings.explore_run == 2
    assert agent.stats().endswith("holding each try for up to 2 steps.")


# ----------------------------------------------------------------- what it is fed


def test_it_is_fed_the_picture_what_changed_and_the_move_last_made() -> None:
    mixed = bytes(range(256)) * 3 + bytes(128)
    rows = inputs([Sight(BRIGHT, DARK, 1), Sight(mixed, mixed), Sight(DARK, BRIGHT, 0)], 3)
    assert rows.shape == (3, 2 * PICTURE_SIZE + 3)
    assert rows[0, :PICTURE_SIZE] == pytest.approx(200 / 255)
    assert rows[0, PICTURE_SIZE : 2 * PICTURE_SIZE] == pytest.approx(180 / 255)
    assert list(rows[0, 2 * PICTURE_SIZE :]) == [0.0, 1.0, 0.0]
    assert rows[1, :PICTURE_SIZE] == pytest.approx(np.frombuffer(mixed, np.uint8) / 255)
    assert not rows[1, PICTURE_SIZE:].any()  # nothing changed, and no move made yet
    assert rows[2, :PICTURE_SIZE] == pytest.approx(20 / 255)
    assert rows[2, PICTURE_SIZE : 2 * PICTURE_SIZE] == pytest.approx(-180 / 255)
    assert list(rows[2, 2 * PICTURE_SIZE :]) == [1.0, 0.0, 0.0]
    # The last move of the three has its slot; a move the controls do not have has none.
    assert list(inputs([Sight(DARK, DARK, 2)], 3)[0, 2 * PICTURE_SIZE :]) == [0.0, 0.0, 1.0]
    assert not inputs([Sight(DARK, DARK, 3)], 3)[0, 2 * PICTURE_SIZE :].any()
    assert inputs([Sight(DARK, DARK, 3)], 3).shape == (1, 2 * PICTURE_SIZE + 3)


def test_its_scores_come_from_the_network_and_ties_go_to_the_first_move() -> None:
    agent = quick()
    wanted = agent.network.forward(inputs([B.sight], 2))[0]  # type: ignore[list-item]
    assert agent.values(B) == pytest.approx(tuple(wanted))
    assert agent.value(B, LEFT) == agent.values(B)[LEFT]
    assert agent.value(B, RIGHT) == agent.values(B)[RIGHT]
    assert agent.greedy(B) == int(np.argmax(wanted)) == agent.greedy(B)
    assert agent.values(A) != agent.values(B)  # another picture, other scores
    assert agent.values(seen(BRIGHT, held=LEFT)) != agent.values(seen(BRIGHT, held=RIGHT))
    assert agent.values(seen(BRIGHT, DARK)) != agent.values(B)  # what moved counts
    with pytest.raises(ConfigurationError):
        agent.value(B, 2)
    # No picture: noughts for every move, and the first move is its answer.
    assert agent.values(BLIND) == (0.0, 0.0) and agent.greedy(BLIND) == LEFT
    assert agent.value(BLIND, RIGHT) == 0.0


def test_with_no_exploring_it_does_what_it_thinks_best() -> None:
    agent = quick(epsilon=0.0, epsilon_min=0.0)
    for state in (A, B, seen(BRIGHT, DARK, 1)):
        best = agent.greedy(state)
        assert [agent.choose(state) for _ in range(5)] == [best] * 5
        decision = agent.last_decision
        assert decision == Decision(best, False, agent.values(state)[best])


def test_it_explores_in_runs() -> None:
    agent = quick(seed=4)
    assert agent.stats().endswith(
        "tries something new 30% of the time, holding each try for up to 8 steps."
    )
    steps = 3000
    explored, starts = 0, 0
    running = False
    for _ in range(steps):
        action = agent.choose(A)
        decision = agent.last_decision
        assert decision is not None and decision.action == action
        if decision.explored:
            explored += 1
            starts += not running
        running = decision.explored
    assert abs(explored / steps - 0.3) < 0.04
    assert explored / starts > 4.0  # a new move every step would average 1.4


def test_what_it_tries_is_half_the_time_one_button_from_what_it_would_do() -> None:
    space = ActionSpace.from_buttons(("up", "down", "left", "right", "a", "b"))
    always = dict(epsilon=1.0, epsilon_min=1.0, epsilon_decay=1.0, rebound=0.0, explore_run=1)
    agent = ScreenLearner(space, Settings(**always), seed=5, config=QUICK)
    held = 0
    for shade in range(0, 240, 20):
        state = seen(bytes([shade]) * PICTURE_SIZE, held=shade % len(space))
        own = agent.greedy(state)  # what the network, as it stands, would do here
        near = set(space.near(own))
        if not near:  # it would hold nothing: there is nothing to keep doing
            continue
        held += 1
        picks = [agent.choose(state) for _ in range(3000)]
        expected = 0.5 + 0.5 * len(near) / len(space)
        assert abs(sum(p in near for p in picks) / len(picks) - expected) < 0.04
        assert set(picks) == set(range(len(space)))
        assert agent.greedy(state) == own  # choosing teaches it nothing
    assert held >= 3


def test_a_decision_holds_what_it_thinks_the_move_it_made_is_worth() -> None:
    agent = quick(seed=3, epsilon=1.0, epsilon_min=1.0, explore_run=1)
    state = seen(BRIGHT, DARK, LEFT)
    values = agent.values(state)
    assert values[LEFT] != values[RIGHT]
    made = set()
    for _ in range(40):
        action = agent.choose(state)
        made.add(action)
        assert agent.last_decision == Decision(action, True, values[action])
    assert made == {LEFT, RIGHT}  # the worse move was tried too, and valued as itself


def test_where_every_move_scores_the_same_it_picks_among_them() -> None:
    """With no picture every move is worth nought. Asked for its best it names the
    first; playing, it does not hold the first move for ever."""
    agent = quick(epsilon=0.0, epsilon_min=0.0)
    picks = [agent.choose(BLIND) for _ in range(200)]
    assert set(picks) == {LEFT, RIGHT} and 60 < picks.count(LEFT) < 140
    assert agent.last_decision is not None and not agent.last_decision.explored
    assert agent.greedy(BLIND) == LEFT


def test_new_settings_reach_its_runs_and_its_rebound() -> None:
    agent = quick(seed=4)
    agent.configure(
        Settings(epsilon=1.0, epsilon_min=0.05, epsilon_decay=1.0, explore_steps=10, rebound=0.5)
    )
    for _ in range(10):
        agent.learn(A, LEFT, 0.0, B)
    for reward in (10.0,) * 10 + (0.0,) * 10:
        agent.end_episode(reward)
    assert agent.lift == 0.5 and agent.epsilon == 0.5  # the new rebound, not the old nought
    short = quick(seed=4)
    short.configure(Settings(epsilon=1.0, epsilon_min=1.0, epsilon_decay=1.0, explore_run=1))
    picks = [short.choose(A) for _ in range(400)]
    changes = sum(a != b for a, b in itertools.pairwise(picks))
    assert changes > 120  # a new try every step; held for up to 8 it changed far less often
    held = quick(seed=4, epsilon=1.0, epsilon_min=1.0)
    picks = [held.choose(A) for _ in range(400)]
    assert sum(a != b for a, b in itertools.pairwise(picks)) < 100


# ------------------------------------------------------------------- it learns


HALF = PICTURE_WIDTH // 2


def lit(world: Random, side: int) -> bytes:
    """A picture bright on one half and dark on the other; no two are alike."""

    def row() -> bytes:
        return bytes(
            world.randrange(150, 256) if (x < HALF) == (side == LEFT) else world.randrange(61)
            for x in range(PICTURE_WIDTH)
        )

    return b"".join(row() for _ in range(PICTURE_HEIGHT))


def noise(world: Random) -> bytes:
    return bytes(world.randrange(256) for _ in range(PICTURE_SIZE))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_it_learns_which_move_pays_when_only_the_picture_says(seed: int) -> None:
    """The move on the bright side pays. Every picture is a new one, and the parts of
    the state are the same throughout: only looking tells it."""
    agent = toy(seed)
    world = Random(7)
    side = world.randrange(2)
    state = seen(lit(world, side))
    for _ in range(1000):
        action = agent.choose(state)
        following = world.randrange(2)
        nxt = seen(lit(world, following))
        agent.learn(state, action, 10.0 if action == side else -10.0, nxt, done=True)
        state, side = nxt, following
    assert agent.updates == 1000 and agent.known == 200 and agent.trained == 993
    fresh = Random(99)  # pictures it has never been shown
    for side in (LEFT, RIGHT):
        shown = [seen(lit(fresh, side)) for _ in range(10)]
        assert sum(agent.greedy(state) == side for state in shown) >= 9
        paying = sum(agent.value(state, side) for state in shown) / 10
        other = sum(agent.value(state, 1 - side) for state in shown) / 10
        assert paying > 0.5 > -0.5 > other  # near the plus and minus one it was paid


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_it_learns_which_move_pays_when_only_the_last_move_says(seed: int) -> None:
    """The pictures are noise. A move pays only when it is not the move just made:
    a button that wants a fresh press."""
    agent = toy(seed)
    world = Random(7)
    held = NO_MOVE
    picture = noise(world)
    state = seen(picture, held=held)
    for _ in range(1000):
        action = agent.choose(state)
        reward = 10.0 if action != held else -10.0
        held = action
        following = noise(world)
        nxt = seen(following, picture, held)
        agent.learn(state, action, reward, nxt, done=True)
        state, picture = nxt, following
    fresh = Random(99)
    for held in (LEFT, RIGHT):
        shown = [seen(noise(fresh), held=held) for _ in range(10)]
        assert sum(agent.greedy(state) == 1 - held for state in shown) >= 9
        paying = sum(agent.value(state, 1 - held) for state in shown) / 10
        other = sum(agent.value(state, held) for state in shown) / 10
        assert paying > 0.3 > -0.3 > other


def test_what_comes_later_reaches_back_through_the_pictures() -> None:
    """Dark leads to bright whatever is pressed; only bright pays. Dark is then worth
    what bright will pay, less the wait."""
    agent = toy(3, gamma=0.9)
    state = A
    for _ in range(1500):
        action = agent.choose(state)
        if state is A:
            agent.learn(A, action, 0.0, B)
            state = B
        else:
            agent.learn(B, action, 10.0, A, done=True)
            state = A
    for move in (LEFT, RIGHT):
        assert agent.value(B, move) == pytest.approx(1.0, abs=0.15)
        assert agent.value(A, move) == pytest.approx(0.9, abs=0.15)


# ----------------------------------------------------- how a step is learnt from


class Fixed:
    """A stand-in network that scores every next situation the same."""

    def __init__(self, row: list[float]) -> None:
        self._row = row

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tile(np.array(self._row), (len(x), 1))


def test_the_next_move_is_chosen_by_the_network_and_valued_by_the_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = quick(gamma=0.9)
    calls = spy_on_training(agent, monkeypatch)
    for _ in range(QUICK.warmup - 1):
        agent.learn(A, RIGHT, 20.0, B)
    assert calls == []
    # The network prefers the right-hand move next; the copy thinks left worth 7 and
    # right worth 3. The step is aimed at the copy's 3, not at its best, 7.
    monkeypatch.setattr(agent.network, "forward", Fixed([1.0, 5.0]))
    monkeypatch.setattr(agent.target, "forward", Fixed([7.0, 3.0]))
    agent._train_batch()
    (x, targets, mask) = calls[0]
    assert x.shape == (8, 2 * PICTURE_SIZE + 2) and targets.shape == mask.shape == (8, 2)
    assert x == pytest.approx(inputs([A.sight] * 8, 2))  # type: ignore[list-item]
    assert targets[:, RIGHT] == pytest.approx(2.0 + 0.9 * 3.0)
    assert not targets[:, LEFT].any()
    assert mask[:, RIGHT].all() and not mask[:, LEFT].any()
    # Preferring left, it is aimed at the copy's 7.
    monkeypatch.setattr(agent.network, "forward", Fixed([5.0, 1.0]))
    agent._train_batch()
    assert calls[1][1][:, RIGHT] == pytest.approx(2.0 + 0.9 * 7.0)


def test_a_reward_is_scaled_spans_its_steps_and_ends_where_the_game_starts_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaled = ScreenConfig(
        hidden=(16,), batch_size=8, warmup=8, target_sync=20, memory=200, reward_scale=0.5
    )
    agent = ScreenLearner(SPACE, Settings(gamma=0.9, **STEADY), seed=1, config=scaled)
    calls = spy_on_training(agent, monkeypatch)
    monkeypatch.setattr(agent.target, "forward", Fixed([4.0, 4.0]))
    for _ in range(7):
        agent.learn(A, LEFT, 6.0, B, steps=3)
    agent._train_batch()  # a three-step return: the future counts gamma cubed
    assert calls[0][1][:, LEFT] == pytest.approx(3.0 + 0.9**3 * 4.0)
    assert agent._memory[0] == (A.sight, LEFT, 3.0, B.sight, False, 3)
    ended = ScreenLearner(SPACE, Settings(gamma=0.9, **STEADY), seed=1, config=scaled)
    calls = spy_on_training(ended, monkeypatch)
    monkeypatch.setattr(ended.target, "forward", Fixed([4.0, 4.0]))
    for _ in range(7):
        ended.learn(A, LEFT, -6.0, B, done=True, steps=2)
    ended._train_batch()  # the game started over: nothing comes after
    assert calls[0][1][:, LEFT] == pytest.approx(-3.0)
    assert ended._memory[0] == (A.sight, LEFT, -3.0, B.sight, True, 2)


def test_a_step_is_checked_before_it_is_learnt_from() -> None:
    agent = quick()
    for bad_reward in (float("nan"), float("inf"), "10", None):
        with pytest.raises(ConfigurationError, match="reward must be a finite number"):
            agent.learn(A, RIGHT, bad_reward, B)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        agent.learn(A, 2, 1.0, B)
    for bad_steps in (0, -1, 1.5, True):
        with pytest.raises(ConfigurationError, match="steps must be a whole number"):
            agent.learn(A, RIGHT, 1.0, B, steps=bad_steps)  # type: ignore[arg-type]
    assert agent.updates == 0 and agent.known == 0 and agent.unseen == 0


def test_a_step_that_shows_no_picture_is_counted_and_teaches_nothing() -> None:
    agent = quick(epsilon=1.0, explore_steps=100)
    before = [p.copy() for p in agent.network.parameters]
    assert agent.learn(BLIND, RIGHT, 5.0, B) == 0.0
    assert agent.learn(A, RIGHT, 5.0, BLIND) == 0.0
    assert agent.learn(BLIND, LEFT, 5.0, BLIND, done=True) == 0.0
    assert (agent.unseen, agent.updates, agent.known, agent.trained) == (3, 0, 0, 0)
    assert agent.epsilon == 1.0  # and exploring does not fall for it
    assert "; 3 steps showed it no picture and taught it nothing; " in agent.stats()
    for _ in range(20):
        agent.learn(BLIND, RIGHT, 5.0, B)
    assert agent.trained == 0
    assert all((p == q).all() for p, q in zip(agent.network.parameters, before, strict=True))
    one = quick()
    one.learn(BLIND, RIGHT, 5.0, B)
    assert "; 1 step showed it no picture and taught it nothing; " in one.stats()
    assert "no picture" not in quick().stats()


def test_it_waits_for_enough_steps_then_trains_every_step_and_copies_itself() -> None:
    agent = quick(seed=4)
    start = [p.copy() for p in agent.network.parameters]

    def same(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
        return all((p == q).all() for p, q in zip(a, b, strict=True))

    for _ in range(QUICK.warmup - 1):
        agent.learn(A, RIGHT, 10.0, B)
    assert agent.trained == 0 and agent.updates == 7 and agent.known == 7
    assert same(agent.network.parameters, start)
    told = agent.learn(B, RIGHT, 10.0, A)
    assert told == agent.value(B, RIGHT) != 0.0  # what it now thinks that move is worth
    assert agent.trained == 1 and agent.network.steps == 1 and agent.known == 8
    assert "; network trained 1 time; " in agent.stats()
    assert not same(agent.network.parameters, start)
    assert same(agent.target.parameters, start)  # the copy is not brought up yet
    for _ in range(QUICK.target_sync - 1):
        agent.learn(A, LEFT, 0.0, A)
    assert agent.trained == QUICK.target_sync
    assert same(agent.target.parameters, agent.network.parameters)
    agent.learn(A, LEFT, 0.0, A)
    assert not same(agent.target.parameters, agent.network.parameters)
    assert agent.stats().startswith(
        "Watches the screen. Remembers 28 steps, learned from 28 steps over 0 goes; "
        "network trained 21 times; tries something new 30% of the time"
    )


def test_its_memory_holds_so_many_steps_and_the_oldest_go_first() -> None:
    small = ScreenConfig(hidden=(8,), batch_size=2, warmup=4, target_sync=5, memory=4)
    agent = ScreenLearner(SPACE, Settings(**STEADY), seed=1, config=small)
    for reward in range(1, 5):
        agent.learn(A, LEFT, float(reward * 10), B)
    assert [step[2] for step in agent._memory] == [1.0, 2.0, 3.0, 4.0]
    agent.learn(A, LEFT, 50.0, B)
    assert [step[2] for step in agent._memory] == [5.0, 2.0, 3.0, 4.0]
    agent.learn(A, LEFT, 60.0, B)
    agent.learn(A, LEFT, 70.0, B)
    agent.learn(A, LEFT, 80.0, B)
    assert [step[2] for step in agent._memory] == [5.0, 6.0, 7.0, 8.0]
    agent.learn(A, LEFT, 90.0, B)
    assert [step[2] for step in agent._memory] == [9.0, 6.0, 7.0, 8.0]
    assert agent.known == 4 and agent.updates == 9
    assert agent.stats().startswith(
        "Watches the screen. Remembers 4 steps, learned from 9 steps over 0 goes; "
    )
    first = quick()
    first.learn(A, LEFT, 1.0, B)
    assert first.stats().startswith(
        "Watches the screen. Remembers 1 step, learned from 1 step over 0 goes; "
        "network trained 0 times; "
    )
    first.end_episode()
    assert "over 1 go; " in first.stats()


# ---------------------------------------------------------------- exploring


def test_exploring_falls_with_the_goes_and_with_the_steps() -> None:
    by_goes = ScreenLearner(
        SPACE, Settings(epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.5), seed=5, config=QUICK
    )
    assert by_goes.end_episode() == 0.25 and by_goes.end_episode() == 0.125
    assert by_goes.end_episode() == 0.1 and by_goes.episodes == 3
    assert "3 goes" in by_goes.stats() and "10%" in by_goes.stats()
    by_steps = ScreenLearner(
        SPACE,
        Settings(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.5, explore_steps=100),
        seed=5,
        config=QUICK,
    )
    for _ in range(50):
        by_steps.learn(A, LEFT, 0.0, B)
    assert by_steps.epsilon == pytest.approx(0.525)
    assert by_steps.end_episode() == pytest.approx(0.2625)
    for _ in range(60):
        by_steps.learn(A, LEFT, 0.0, B)
    assert by_steps.epsilon == 0.05 and by_steps.updates == 110


def test_exploring_rises_again_when_the_goes_get_worse() -> None:
    agent = ScreenLearner(
        SPACE,
        Settings(epsilon=1.0, epsilon_min=0.05, epsilon_decay=1.0, explore_steps=10),
        seed=5,
        config=QUICK,
    )
    for _ in range(10):
        agent.learn(A, LEFT, 0.0, B)
    assert agent.epsilon == 0.05
    for reward in (10.0,) * 10 + (0.0,) * 9:
        assert agent.end_episode(reward) == 0.05
    assert agent.end_episode(0.0) == 0.25 and agent.lift == 0.25
    agent.learn(A, LEFT, 0.0, B)
    assert agent.epsilon == 0.25
    again = learner_from_dict(json.loads(json.dumps(agent.to_dict())), space=SPACE)
    assert isinstance(again, ScreenLearner) and again.lift == 0.25 and again.epsilon == 0.25
    assert again.recent_rewards == agent.recent_rewards and len(agent.recent_rewards) > 0
    for reward in (0.0,) * 10 + (10.0,) * 10:
        agent.end_episode(reward)
    assert agent.lift == 0.0 and agent.epsilon == 0.05


def test_a_run_of_tries_ends_with_the_go() -> None:
    agent = quick(seed=4, epsilon=1.0, epsilon_min=1.0)
    start_a_run(agent)
    assert agent._runs.to_dict() != quick()._runs.to_dict()  # a run is on
    agent.end_episode()
    assert agent._runs.to_dict() == quick()._runs.to_dict()


def test_started_again_it_knows_nothing() -> None:
    agent = quick(seed=6, epsilon=0.5)
    for _ in range(30):
        agent.learn(A, agent.choose(A), 10.0, B)
    agent.learn(BLIND, LEFT, 1.0, B)
    agent.end_episode(3.0)
    start_a_run(agent)
    assert agent._runs.to_dict() != quick()._runs.to_dict()
    learnt = [p.copy() for p in agent.network.parameters]
    assert agent.known == 30 and agent.trained > 0 and agent.unseen == 1
    agent.reset()
    assert agent._runs.to_dict() == quick()._runs.to_dict()
    assert (agent.known, agent.updates, agent.episodes, agent.trained, agent.unseen) == (0,) * 5
    assert agent.epsilon == 0.5 and agent.last_decision is None
    assert agent.recent_rewards == () and agent._memory == [] and agent._at == 0
    assert any((p != q).any() for p, q in zip(agent.network.parameters, learnt, strict=True))
    assert all(
        (p == q).all()
        for p, q in zip(agent.network.parameters, agent.target.parameters, strict=True)
    )
    assert agent.stats().startswith("Watches the screen. Remembers 0 steps, learned from 0 steps")


def test_it_carries_on_another_learners_goes_and_explores_from_the_start() -> None:
    table = QLearningAgent(SPACE, Settings(epsilon=1.0, explore_steps=10), seed=1)
    for _ in range(10):
        table.learn(BLIND, LEFT, 1.0, BLIND)
    for _ in range(7):
        table.end_episode()
    assert table.episodes == 7 and table.epsilon < 0.1
    agent = quick(epsilon=1.0, epsilon_min=0.05)
    agent.resume_from(table)
    assert agent.episodes == 7 and agent.updates == 0
    assert agent.epsilon == 1.0  # the other's exploring is not taken: this one knows nothing
    counted = quick()
    counted.learn(A, LEFT, 1.0, B)
    assert counted.updates == 1
    counted.resume_from(table)
    assert counted.updates == 0 and counted.episodes == 7  # not the other's ten steps


# -------------------------------------------------------------- the same again


def test_the_same_seed_learns_the_same_and_another_seed_does_not() -> None:
    def run(seed: int) -> tuple[list[int], tuple[float, ...]]:
        agent = quick(seed=seed)
        moves = []
        state = A
        for _ in range(60):
            action = agent.choose(state)
            moves.append(action)
            nxt = B if state is A else A
            agent.learn(state, action, 10.0 if action == RIGHT else 0.0, nxt)
            state = nxt
        return moves, agent.values(A)

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_saved_and_loaded_it_chooses_and_learns_the_same() -> None:
    agent = quick(seed=6, epsilon=0.4, epsilon_min=0.4)
    state = A
    for _ in range(40):
        action = agent.choose(state)
        nxt = B if state is A else A
        agent.learn(state, action, 10.0 if action == RIGHT else 0.0, nxt)
        state = nxt
    agent.learn(BLIND, LEFT, 1.0, B)
    agent.end_episode(4.0)
    start_a_run(agent)  # saved in the middle of a try it is holding
    text = json.dumps(agent.to_dict())
    data = json.loads(text)
    assert data["run"]["left"] > 0
    assert data["algorithm"] == "screen" and data["schema"] == 1
    # Without its memory of steps and without the copy: a few kilobytes a go would
    # otherwise be megabytes.
    assert "memory" not in data and "target" not in data and "replay" not in data
    assert set(data["network"]) == {"sizes", "numbers"}
    assert data["network"]["sizes"] == [2 * PICTURE_SIZE + 2, 16, 2]
    numbers = (2 * PICTURE_SIZE + 2) * 16 + 16 + 16 * 2 + 2
    assert len(base64.b64decode(data["network"]["numbers"])) == 8 * numbers
    assert len(text) < 2 * 8 * numbers
    again = learner_from_dict(data, space=SPACE)
    assert isinstance(again, ScreenLearner) and again.config == QUICK
    assert again.settings == agent.settings and again.space == SPACE
    assert (again.epsilon, again.updates, again.episodes) == (agent.epsilon, 40, 1)
    assert (again.trained, again.unseen) == (agent.trained, 1)
    assert again.recent_rewards == (4.0,)
    assert again._runs.to_dict() == agent._runs.to_dict() == data["run"]
    assert again.known == 0  # the memory starts empty
    held = data["run"]["action"]  # the try goes on where it was
    assert [again.choose(A) for _ in range(data["run"]["left"])] == [held] * data["run"]["left"]
    assert [agent.choose(A) for _ in range(data["run"]["left"])] == [held] * data["run"]["left"]
    for state in (A, B, seen(BRIGHT, DARK, RIGHT), seen(DARK, BRIGHT, LEFT)):
        assert again.values(state) == agent.values(state)
        assert again.greedy(state) == agent.greedy(state)
    assert [again.choose(A) for _ in range(40)] == [agent.choose(A) for _ in range(40)]
    assert all(
        (p == q).all()
        for p, q in zip(again.target.parameters, again.network.parameters, strict=True)
    )
    assert again.to_dict() == agent.to_dict()
    # Without the random state it still loads, and plays the same with no exploring.
    slim = {k: v for k, v in data.items() if k != "rng"}
    assert learner_from_dict(slim, space=SPACE).greedy(B) == agent.greedy(B)
    # Loaded, it goes on learning.
    for _ in range(QUICK.warmup):
        again.learn(A, RIGHT, 10.0, B)
    assert again.trained == agent.trained + 1 and again.known == QUICK.warmup


def test_what_is_not_its_brain_is_refused() -> None:
    data = json.loads(json.dumps(quick().to_dict()))
    with pytest.raises(ConfigurationError, match="different controls"):
        ScreenLearner.from_dict(data, space=ActionSpace([frozenset()]))
    with pytest.raises(ConfigurationError, match=r"newer version \(schema 99\)"):
        ScreenLearner.from_dict({**data, "schema": 99})
    with pytest.raises(ConfigurationError, match="not learned by a screen network"):
        ScreenLearner.from_dict({**data, "algorithm": "dqn"})
    with pytest.raises(ConfigurationError, match="Bad brain data"):
        ScreenLearner.from_dict({"schema": 1})
    with pytest.raises(ConfigurationError, match="Bad brain data"):
        ScreenLearner.from_dict({k: v for k, v in data.items() if k != "network"})
    with pytest.raises(ConfigurationError, match="the network is not a mapping"):
        ScreenLearner.from_dict({**data, "network": "nope"})
    net = data["network"]
    with pytest.raises(ConfigurationError, match="does not match the controls and the picture"):
        ScreenLearner.from_dict({**data, "network": {**net, "sizes": [10, 16, 2]}})
    with pytest.raises(ConfigurationError, match="does not match the controls and the picture"):
        ScreenLearner.from_dict({**data, "network": {"numbers": net["numbers"]}})
    # Another shape of network inside the same brain.
    other = {**data, "config": {**data["config"], "hidden": [8]}}
    with pytest.raises(ConfigurationError, match="does not match the controls and the picture"):
        ScreenLearner.from_dict(other)
    with pytest.raises(ConfigurationError, match="the network's numbers"):
        ScreenLearner.from_dict({**data, "network": {**net, "numbers": "not base64!"}})
    with pytest.raises(ConfigurationError, match="the network's numbers"):
        ScreenLearner.from_dict({**data, "network": {**net, "numbers": 5}})
    stray = net["numbers"][:8] + "!?" + net["numbers"][8:]  # not skipped over: refused
    with pytest.raises(ConfigurationError, match="the network's numbers"):
        ScreenLearner.from_dict({**data, "network": {**net, "numbers": stray}})
    with pytest.raises(ConfigurationError, match="the network's numbers"):
        ScreenLearner.from_dict({**data, "network": {**net, "numbers": net["numbers"][:-16]}})
    raw = bytearray(base64.b64decode(net["numbers"]))
    for wrong in (bytes(raw[:-8]), bytes(raw) + bytes(8)):  # a number short, one too many
        packed = base64.b64encode(wrong).decode()
        with pytest.raises(ConfigurationError, match="numbers are not all there"):
            ScreenLearner.from_dict({**data, "network": {**net, "numbers": packed}})
    raw[:8] = np.array([np.nan], dtype="<f8").tobytes()
    broken = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(ConfigurationError, match="the network's numbers are not all there"):
        ScreenLearner.from_dict({**data, "network": {**net, "numbers": broken}})
    with pytest.raises(ConfigurationError):
        ScreenLearner.from_dict({**data, "epsilon": 7})
    with pytest.raises(ConfigurationError):
        ScreenLearner.from_dict({**data, "config": {"batch_size": 0}})
