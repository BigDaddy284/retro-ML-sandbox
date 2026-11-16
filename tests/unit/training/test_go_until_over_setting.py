"""T-1044: the choice that a go lasts until the game is over, kept for each game."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_racer
from retro_ai_lab.services import ButtonsHeld, GoStarted
from retro_ai_lab.training import Brain, Mode, service
from retro_ai_lab.training.service import UNTIL_OVER_MOST_FRAMES, until_over_most_steps
from tests.unit.training import test_service as the_service
from tests.unit.training.test_service import Lab

lab = the_service.lab  # the practice racer behind the training service (pytest fixture)

TIME = 50  # the lab's "Each go lasts", in steps
MOST = 27_000  # 30 minutes of console time at the lab's 4 frames a step


def test_the_most_a_go_may_last_is_thirty_minutes_of_console_time() -> None:
    assert UNTIL_OVER_MOST_FRAMES == 30 * 60 * 60
    assert until_over_most_steps(4) == MOST
    assert [until_over_most_steps(frames) for frames in (1, 6, 7)] == [108_000, 18_000, 15_428]
    assert until_over_most_steps(108_000) == 1 == until_over_most_steps(500_000)  # a step at least
    for bad in (0, -4, 2.5, True, "4", None):
        with pytest.raises(ConfigurationError, match="frames_per_step must be a whole number"):
            until_over_most_steps(bad)  # type: ignore[arg-type]


def test_it_is_off_until_asked_for_and_asked_for_only_when_idle(lab: Lab) -> None:
    t = lab.training
    assert t.go_until_over is False and t.go_limits == (TIME, None)  # with no game yet, too
    lab.connect()
    assert t.go_until_over is False and t.go_limits == (TIME, None)
    brain = t.brain()
    assert brain is not None and brain.until_over is False and "until_over" not in brain.to_dict()
    for bad in (1, 0, "yes", None, 1.0):
        with pytest.raises(ConfigurationError, match="until the game is over, or it does not"):
            t.set_go_until_over(bad)  # type: ignore[arg-type]
    assert t.go_until_over is False
    saves, changes = t.saves, len(lab.changes)
    t.set_go_until_over(True)
    assert t.go_until_over is True
    # The time chosen is kept: it is now how long a go may be paid nothing.
    assert t.max_steps == TIME and t.go_limits == (MOST, TIME)
    assert len(lab.changes) == changes + 1 and lab.changes[-1].reason == "go settings"
    assert t.saves == saves + 1  # written to disk at once
    brain = t.brain()
    assert brain is not None and brain.until_over is True
    assert brain.to_dict()["until_over"] is True and brain.go_length is None
    # Not while training, either way.
    t.start(Mode.INDEPENDENT)
    for on in (False, True):
        with pytest.raises(ConfigurationError, match="Stop training before changing"):
            t.set_go_until_over(on)
    t.stop()
    assert t.go_until_over is True
    # Another time is another wait with no pay, and the timer again once this is off.
    t.set_go_length(20)
    assert t.go_limits == (MOST, 20)
    t.set_go_until_over(False)
    assert t.go_until_over is False and t.go_limits == (20, None) and t.max_steps == 20
    brain = t.brain()
    assert brain is not None and brain.go_length == 20 and "until_over" not in brain.to_dict()


def test_it_is_kept_for_the_game(lab: Lab) -> None:
    k = lab.connect()
    t = lab.training
    t.set_go_until_over(True)
    lab.connection.disconnect()
    assert t.go_until_over is True  # untouched until a game is connected
    lab.connect()
    assert t.go_until_over is True and k.game_id == t.game_id and t.go_limits == (MOST, TIME)
    # Another game starts with a timer; the first has its choice when it comes back.
    lab.connection.disconnect()
    lab.connection.connect("fake", seed=9)
    assert t.go_until_over is False and t.go_limits == (TIME, None)
    lab.connection.disconnect()
    lab.connect()
    assert t.go_until_over is True
    # Switched off, that is kept too.
    t.set_go_until_over(False)
    lab.connection.disconnect()
    lab.connect()
    assert t.go_until_over is False


def test_the_brain_carries_it(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    t.set_go_until_over(True)
    brain = t.brain()
    assert brain is not None
    data = brain.to_dict()
    space = brain.agent.space
    assert Brain.from_dict(data, space=space).until_over is True
    # A brain saved before there was such a choice has a timer.
    old = dict(data)
    del old["until_over"]
    assert Brain.from_dict(old, space=space).until_over is False
    assert Brain.from_dict({**data, "until_over": False}, space=space).until_over is False
    # Anything but yes or no is no brain.
    for bad in ("yes", 1, 0, None, [True]):
        with pytest.raises(ConfigurationError, match="until_over must be true or false"):
            Brain.from_dict({**data, "until_over": bad}, space=space)
    with pytest.raises(ConfigurationError, match="until_over must be true or false"):
        Brain(brain.game_id, brain.agent, until_over="no")  # type: ignore[arg-type]
    # A brain made with nothing said about it has a timer.
    bare = Brain(brain.game_id, brain.agent)
    assert bare.until_over is False and "until_over" not in bare.to_dict()


def test_what_a_go_is_built_with(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[object, object]] = []
    real = service.EpisodeRunner

    def building(*args: object, **kw: object) -> object:
        built.append((kw["max_steps"], kw["gain_within"]))
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(service, "EpisodeRunner", building)
    lab.connect()
    t = lab.training
    t.train(1)
    assert built == [(TIME, None)]
    t.set_go_until_over(True)
    t.train(1)
    assert built == [(TIME, None), (MOST, TIME)]
    t.set_go_length(20)
    t.train(1)
    assert built[-1] == (MOST, 20)
    t.set_go_until_over(False)
    t.train(1)
    assert built[-1] == (20, None) and len(built) == 4


def test_goes_on_the_practice_game_end_when_the_game_is_over(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    lab.connect()
    t = lab.training
    t.set_go_until_over(True)
    game = lab.adapter().game
    # What the game was at when each go made its first move.
    fresh, began = [False], []

    def first_move(event: ButtonsHeld) -> None:
        if fresh[0]:
            fresh[0] = False
            began.append((game.state.lives, game.state.game_over))

    lab.bus.subscribe(GoStarted, lambda event: fresh.__setitem__(0, True))
    lab.bus.subscribe(ButtonsHeld, first_move)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.service"):
        records = t.train(8)
    results = [r.result for r in records]
    # No go ends by the time, and each ends the time after it was last paid.
    assert [(r.ended_by, r.steps) for r in results] == [
        ("no_gain", 83),
        ("no_gain", 50),
        ("no_gain", 109),
        ("no_gain", 50),
        ("no_gain", 50),
        ("no_gain", 74),
        ("no_gain", 50),
        ("no_gain", 58),
    ]
    # Goes that gain something last past the time. (Since T-1048 the seventh loses a
    # life on its last step and is paid nothing: points taken off are not a gain.)
    gained = [any(e.amount > 0 for e in r.earned) for r in results]
    assert [r.steps > TIME for r in results] == gained
    assert [r.reward for r in results] == [35.0, 0.0, 7.0, 0.0, 0.0, -18.0, -10.0, -18.0]
    # Three of them lost every life: the game was over, which paid them the last time
    # ("Game over switches on"), and they ended. With no timer they never did.
    assert [r.setbacks for r in results] == [2, 0, 3, 0, 0, 3, 1, 3]
    # The restart is the next go: each begins from the checkpoint with every life.
    assert began == [(toy_racer.START_LIVES, False)] * 8
    said = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("Go ")]
    assert len(said) == 8 and said[0].startswith(
        "Go 1 ended by no_gain after 83 steps (332 frames)"
    )


def test_with_a_timer_out_of_reach_such_a_go_does_not_end(lab: Lab) -> None:
    """What the choice is for: the try-out's go, which sat at the game-over screen."""
    lab.connect()
    t = lab.training
    t.set_go_length(3_000)
    first, second = (record.result for record in t.train(2))
    # The first gives up with no wake button known; the second, with one, goes round.
    assert (first.ended_by, first.setbacks) == ("stuck", toy_racer.START_LIVES)
    assert (second.ended_by, second.steps, second.setbacks) == ("max_steps", 3_000, 3)
    assert lab.adapter().game.state.game_over
    # With the choice made, the goes that follow end: the time after their last pay.
    t.set_go_length(TIME)
    t.set_go_until_over(True)
    after = [record.result for record in t.train(5)]  # (four, before T-1048)
    assert all(r.ended_by == "no_gain" and TIME <= r.steps < 500 for r in after)
    assert any(r.setbacks == toy_racer.START_LIVES for r in after)  # game over among them
