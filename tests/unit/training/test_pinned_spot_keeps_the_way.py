"""T-1063: a pinned spot is the same spot at every Start, and the way from it is kept.

Every Start put the game back to the pinned spot and saved the checkpoint the goes
start from again. A checkpoint saved again is a new spot to everything worked out
from the old one, so the way the AI remembered was forgotten at every Start: with a
pinned spot as without one. Where the checkpoint already is the pinned spot, a Start
now leaves it as it is.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import FakeEmulatorAdapter, toy_hopper
from retro_ai_lab.training import Mode, TrainingService
from retro_ai_lab.training import service as module
from retro_ai_lab.training.service import PRACTISE_SLOT
from tests.unit.training import test_going_back_in_training as in_training
from tests.unit.training.test_going_back_in_training import GAP_ENDS, Lab
from tests.unit.training.test_remembered_way_in_training import counting, goes

lab = in_training.lab  # the practice game behind the real services (pytest fixture)

CHANGED = "The spot the goes start from has changed"
SLOT = module.CHECKPOINT_SLOT


def a_way(t: TrainingService, monkeypatch: pytest.MonkeyPatch) -> int:
    """Start, play one go that leaves a way across the first gap, and stop. Returns
    how many steps of it the AI remembers."""
    counting(t, monkeypatch)
    goes(t, 1)
    way = t.remembered
    assert way > 0
    t.stop()
    return way


def played_on(lab: Lab, frames: int = 24) -> None:
    """The child plays on to the right, outside any go."""
    adapter = lab.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    right = next(b for b in adapter.buttons() if b.name == "right")
    adapter.step(frames, pressed=[right])


def test_a_pinned_spot_keeps_the_way_over_stop_and_start(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    t = lab.training()
    game = lab.connect()
    spot = game.state.x
    t.pin_spot()
    way = a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None and harness.taken == 1 and lab.saved.count(SLOT) == 1
    with caplog.at_level(logging.INFO):
        for _ in range(2):
            assert game.state.x != spot  # the game is where the last go left it
            loads = lab.loaded.count(SLOT)
            asked = counting(t, monkeypatch)  # Start again
            # The spot the goes start from is the one it was: not saved again, and the
            # game put back to it.
            assert harness.taken == 1 and lab.saved.count(SLOT) == 1
            assert lab.loaded.count(SLOT) == loads + 1 and game.state.x == spot
            assert t.remembered == way
            # The go begins by playing the way, before the learner is asked for anything.
            assert t.step()
            go = t.episode
            assert go is not None and go.remembered == way and t.remembered == way
            for _ in range(way - 1):
                assert t.step() and t.episode is go
            assert asked == []
            # It is across the gap that cost the first go a life, with every life.
            assert game.state.x > GAP_ENDS and game.state.lives == toy_hopper.START_LIVES
            assert t.step() and asked == [1]
            t.stop()
            assert t.remembered >= way
            way = t.remembered
    assert not any(CHANGED in rec.getMessage() for rec in caplog.records)
    assert lab.saved.count(PRACTISE_SLOT) == 1


def test_with_no_pinned_spot_a_start_takes_the_spot_the_game_is_at(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """As before: the game is wherever the last go left it, and the way from the old
    spot does not lead from there."""
    t = lab.training()
    lab.connect()
    way = a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None and harness.taken == 1
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
        assert harness.taken == 2 and lab.saved.count(SLOT) == 2
        assert t.step()
        go = t.episode
        assert go is not None and go.remembered == 0 and t.remembered == 0
    said = [rec.getMessage() for rec in caplog.records if CHANGED in rec.getMessage()]
    forgot = f"{CHANGED}, so the way the AI remembered from the old one ({way} steps) is forgotten."
    assert said == [forgot]
    t.stop()


def test_a_spot_pinned_again_is_another_spot(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    t = lab.training()
    game = lab.connect()
    t.pin_spot()
    way = a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None and harness.taken == 1
    # The child takes the game somewhere else and pins that.
    played_on(lab)
    here = game.state.x
    t.pin_spot()
    played_on(lab)
    assert game.state.x != here
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
        assert harness.taken == 2 and game.state.x == here
        assert t.step()
        go = t.episode
        assert go is not None and go.remembered == 0 and t.remembered == 0
    assert sum(CHANGED in rec.getMessage() for rec in caplog.records) == 1
    assert f"({way} steps) is forgotten" in caplog.text
    t.stop()
    # From the new spot on it is one spot again.
    counting(t, monkeypatch)
    assert harness.taken == 2 and game.state.x == here
    t.stop()


def test_the_same_spot_pinned_again_is_saved_again(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The app does not look at what a pin holds: a spot pinned again is taken for
    another, wherever the game was."""
    t = lab.training()
    lab.connect()
    t.pin_spot()
    a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None and harness.taken == 1
    counting(t, monkeypatch)  # back at the spot
    t.stop()
    assert harness.taken == 1
    t.pin_spot()
    counting(t, monkeypatch)
    assert harness.taken == 2
    assert t.step()
    go = t.episode
    assert go is not None and go.remembered == 0
    t.stop()


def test_unpinned_again_the_next_start_is_where_the_game_is(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = lab.training()
    game = lab.connect()
    spot = game.state.x
    t.pin_spot()
    way = a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None and t.remembered == way
    t.unpin_spot()
    left_at = game.state.x
    counting(t, monkeypatch)  # no spot to go back to: the game is where the go left it
    assert harness.taken == 2 and game.state.x == left_at != spot
    assert t.step()
    go = t.episode
    assert go is not None and go.remembered == 0 and t.remembered == 0
    t.stop()
    # Pinned once more, there, that is the spot from then on.
    saves = harness.taken
    t.pin_spot()
    counting(t, monkeypatch)
    assert harness.taken == saves + 1
    t.stop()
    counting(t, monkeypatch)
    assert harness.taken == saves + 1
    t.stop()


def test_handing_over_to_the_ai_while_running_keeps_the_way(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change of mode while training runs prepares the spot as a Start does."""
    t = lab.training()
    game = lab.connect()
    spot = game.state.x
    t.pin_spot()
    way = a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None
    t.start(Mode.MANUAL)  # the child plays: no go of the AI's, and the spot stays held
    assert harness.taken == 1
    played_on(lab)
    t.set_mode(Mode.INDEPENDENT)
    assert harness.taken == 1 and game.state.x == spot and t.remembered == way
    assert t.step()
    go = t.episode
    assert go is not None and go.remembered == way
    t.stop()


def test_a_checkpoint_taken_since_is_not_the_pinned_spot(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever else saves the checkpoint (a game that had to be woken, a start that
    was moved on) leaves it somewhere other than the pinned spot: the next Start goes
    back to the pin and saves it, as before."""
    t = lab.training()
    game = lab.connect()
    spot = game.state.x
    t.pin_spot()
    a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None and harness.taken == 1
    played_on(lab)
    harness.checkpoint()  # as the service does when it moves the start on
    assert harness.taken == 2
    counting(t, monkeypatch)
    assert harness.taken == 3 and game.state.x == spot
    assert t.step()
    go = t.episode
    assert go is not None and go.remembered == 0
    t.stop()
    counting(t, monkeypatch)
    assert harness.taken == 3
    t.stop()


def test_a_harness_that_lost_its_checkpoint_is_given_one(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = lab.training()
    game = lab.connect()
    spot = game.state.x
    t.pin_spot()
    a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None and harness.taken == 1
    harness._has_checkpoint = False  # as a trial that lost the link leaves it
    counting(t, monkeypatch)
    assert harness.has_checkpoint and harness.taken == 2 and game.state.x == spot
    t.stop()


def test_after_the_link_went_the_spot_is_pinned_and_saved_anew(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save states do not outlive the link: the pin is gone, a new harness is made
    with the new link, and nothing of the old spot is taken for held."""
    t = lab.training()
    lab.connect()
    t.pin_spot()
    a_way(t, monkeypatch)
    old = t.harness
    assert old is not None and old.taken == 1
    lab.connection.disconnect()
    lab.connect()
    assert not t.spot_pinned
    new = t.harness
    assert new is not None and new is not old and new.taken == 0
    t.pin_spot()
    counting(t, monkeypatch)
    assert new.taken == 1
    t.stop()
    counting(t, monkeypatch)
    assert new.taken == 1
    t.stop()
