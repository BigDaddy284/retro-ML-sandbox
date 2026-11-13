"""T-1015: the furthest way played is remembered in training, and each go begins with it."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import EpisodeRunner, TrainingService
from retro_ai_lab.training import service as module
from retro_ai_lab.training.records import Mode
from tests.unit.training import test_going_back_in_training as in_training
from tests.unit.training.test_going_back_in_training import GAP_ENDS, NO_SAVES, Lab, walking_right

lab = in_training.lab  # the practice game behind the real services (pytest fixture)

NOT_LISTENING = "The game is not listening to the AI; trying each button to get it going."
REMEMBERS = (
    "It remembers the best go it has played from the spot the goes start from, by the "
    "rewards you set, and begins each go by playing that way again."
)


def goes(t: TrainingService, count: int) -> None:
    """Step until ``count`` more goes are over."""
    upto = t.episodes + count
    while t.episodes < upto:
        assert t.step()


def counting(t: TrainingService, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Start training with a brain that only walks right; every time it is asked is
    counted in the list returned."""
    walking_right(t, monkeypatch)
    agent = t.agent
    assert agent is not None
    walk = agent.space.index_of(["right"])
    asked: list[int] = []
    monkeypatch.setattr(agent, "choose", lambda state: (asked.append(1), walk)[1])
    return asked


def alike(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every go from here on looks like the one before it for its first steps, with the
    AI pressing more than one thing: what a game that ignores the AI looks like."""
    monkeypatch.setattr(EpisodeRunner, "fingerprint", property(lambda self: ("same",) * 8))
    monkeypatch.setattr(EpisodeRunner, "actions_varied", property(lambda self: True))


def test_the_next_go_in_training_begins_with_the_way(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = lab.training()
    game = lab.connect()
    asked = counting(t, monkeypatch)
    assert t.remembered == 0
    goes(t, 1)  # it walks into the first gap, goes back and gets across
    first = t.history[-1].result
    assert first.remembered == 0 and first.ways_found >= 1
    way = t.remembered
    assert 0 < way < first.steps  # none of the steps that fell in, nor the standing about
    # The second go plays the way before the learner is asked for anything.
    asked.clear()
    assert t.step()
    go = t.episode
    assert go is not None and go.remembered == way and go.route is not None
    assert go.route.moves and len(go.route.moves) == way
    for _ in range(way - 1):
        assert t.step() and t.episode is go
    assert asked == []
    # It is across the gap that took a life from the first go, with every life and no search.
    assert game.state.x > GAP_ENDS and game.state.lives == toy_hopper.START_LIVES
    assert (go.ways_sought, go.times_back) == (0, 0)
    assert t.step() and asked == [1]  # from here on it is the learner's go again
    goes(t, 1)
    second = t.history[-1].result
    assert second.remembered == way and second.steps > way
    assert t.remembered >= way  # a go never leaves it remembering less ground
    t.stop()


def test_goes_that_begin_from_memory_are_not_a_game_that_ignores_the_ai(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    t = lab.training()
    lab.connect()
    alike(monkeypatch)
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
        harness = t.harness
        assert harness is not None and harness.taken == 1
        goes(t, 3)
        assert t.step()  # a fourth go has begun: nothing was done to the game between goes
    began = [record.result.remembered for record in t.history]
    assert began[0] == 0 and began[1] > 0 and began[2] > 0
    # The spot the goes start from is the one taken at Start, and the way is still there.
    assert harness.taken == 1 and lab.saved.count(module.CHECKPOINT_SLOT) == 1
    assert t.wake_button is None and t.remembered >= began[2]
    assert not any(NOT_LISTENING in rec.getMessage() for rec in caplog.records)
    t.stop()


def test_goes_with_nothing_remembered_are_still_compared(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The same two goes with no way behind them are taken for a game that ignores the
    AI, as they were before: only a go that began from memory is left out."""
    t = lab.training(max_steps=12, unstick_after=None)
    lab.connect()
    alike(monkeypatch)
    with caplog.at_level(logging.INFO):
        t.start(Mode.INDEPENDENT)
        agent = t.agent
        assert agent is not None
        still = agent.space.index_of([])
        monkeypatch.setattr(agent, "choose", lambda state: still)  # it gets nowhere
        harness = t.harness
        assert harness is not None and harness.taken == 1
        goes(t, 2)
        assert [record.result.remembered for record in t.history] == [0, 0]
        assert t.remembered == 0
        assert not any(NOT_LISTENING in rec.getMessage() for rec in caplog.records)
        t.step()  # between the goes the game is woken, and where it is becomes the start
    assert sum(NOT_LISTENING in rec.getMessage() for rec in caplog.records) == 1
    assert harness.taken == 2
    t.stop()


def test_the_way_outlives_the_runner_and_is_forgotten_from_scratch(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = lab.training(promote_at=3)  # the simple table is moved to a network after a go
    lab.connect()
    assert t.algorithm == "auto"
    counting(t, monkeypatch)
    assert t.step()
    first, brain = t.episode, t.agent
    goes(t, 1)
    way = t.remembered
    assert way > 0
    assert t.step()
    second = t.episode
    assert t.agent is not brain  # promoted: the runner was built again with the new brain
    assert second is not None and second is not first
    assert first is not None and second.route is first.route
    assert second.remembered == way
    t.stop()
    assert t.remembered >= way  # stopping forgets nothing
    t.forget()  # from scratch is from scratch
    assert t.remembered == 0


@pytest.mark.parametrize("without", ["going back", "save states", "the switch"])
def test_no_route_without_save_states_or_with_going_back_off(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, without: str
) -> None:
    if without == "going back":
        t = lab.training(ways_per_go=0)
        lab.connect()
    elif without == "the switch":
        for nonsense in (None, 1, "yes"):
            with pytest.raises(ConfigurationError, match="remember"):
                lab.training(remember=nonsense)
        t = lab.training(remember=False)  # it still goes back, and keeps nothing of it
        lab.connect()
    else:
        t = lab.training()
        lab.connect(capabilities=NO_SAVES)
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
        assert t.step()
        go = t.episode
        assert go is not None and go.route is None
        goes(t, 2)
    assert [record.result.remembered for record in t.history] == [0, 0]
    assert t.remembered == 0
    said = [rec.getMessage() for rec in caplog.records]
    assert REMEMBERS not in said and not any("it remembered" in line for line in said)
    t.stop()


def test_a_start_that_moves_forgets_the_way_and_says_so(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    t = lab.training()
    lab.connect()
    counting(t, monkeypatch)
    goes(t, 1)
    way = t.remembered
    assert way > 0
    t.stop()
    harness = t.harness
    assert harness is not None and harness.taken == 1
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)  # Start again: where the game is now is where goes start
        assert harness.taken == 2
        assert t.remembered == way  # until a go begins from the new spot
        assert t.step()
        go = t.episode
        assert go is not None and go.remembered == 0 and t.remembered == 0
        goes(t, 2)
    forgot = (
        "The spot the goes start from has changed, so the way the AI remembered from the "
        f"old one ({way} steps) is forgotten."
    )
    said = [rec.getMessage() for rec in caplog.records]
    assert said.count(forgot) == 1
    assert t.history[-2].result.remembered == 0  # the first go from the new spot
    t.stop()


def test_the_log_says_what_it_began_with(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    t = lab.training()
    lab.connect()
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
        goes(t, 3)
    said = [rec.getMessage() for rec in caplog.records]
    assert said.count(REMEMBERS) == 1  # once, not once a go
    will = next(i for i, line in enumerate(said) if "goes back a little" in line)
    assert said[will + 1] == REMEMBERS
    first, second = (
        next(line for line in said if line.startswith(f"Go {n} ended")) for n in (1, 2)
    )
    assert "remembered" not in first
    began = t.history[1].result
    words = began.describe_remembered()
    assert (
        began.remembered > 0 and words == f", began with the {began.remembered} steps it remembered"
    )
    assert words in second
    if began.describe_losses():  # the way it began with is said before what it lost
        assert second.index(words) < second.index(began.describe_losses())
    grew = [line for line in said if line.startswith("The AI did better than the way it")]
    assert grew and f"it now remembers {t.history[1].result.remembered} steps" in grew[0]
    t.stop()


def test_nothing_is_remembered_that_the_rewards_do_not_pay_for(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The way is the best go by the person's rewards: with every reward that pays
    switched off (only losing a life costs), no go is better than nothing, and no
    way is kept, however far the go got."""
    t = lab.training()
    game = lab.connect()
    counting(t, monkeypatch)
    goes(t, 1)
    assert t.remembered > 0  # paid for getting somewhere new: remembered
    t.stop()
    profile = lab.rewards.profile
    assert profile is not None
    paying = [b.id for b in profile if b.enabled and b.value > 0]
    assert paying
    for behaviour in paying:
        lab.rewards.set_enabled(behaviour, False)
    counting(t, monkeypatch)
    furthest = 0
    for _ in range(2):
        go = t.episodes
        while t.episodes == go:
            assert t.step()
            furthest = max(furthest, game.state.x)
    assert furthest > GAP_ENDS  # it still goes back and gets across the gap
    assert all(record.result.reward <= 0 for record in t.history[-2:])
    assert [record.result.remembered for record in t.history[-2:]] == [0, 0]
    assert t.remembered == 0
    t.stop()
