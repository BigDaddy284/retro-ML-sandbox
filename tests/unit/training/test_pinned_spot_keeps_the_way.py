"""T-1063: a pinned spot is the same spot at every Start, and the way from it is kept.

Every Start puts the game back to the pinned spot and saves the checkpoint the goes
start from again. That save was counted as a new spot, and the way the AI remembered
was forgotten at every Start: with a pinned spot as without one.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import FakeEmulatorAdapter, toy_hopper
from retro_ai_lab.training import TrainingService
from retro_ai_lab.training import service as module
from tests.unit.training import test_going_back_in_training as in_training
from tests.unit.training.test_going_back_in_training import GAP_ENDS, Lab
from tests.unit.training.test_remembered_way_in_training import counting, goes

lab = in_training.lab  # the practice game behind the real services (pytest fixture)

CHANGED = "The spot the goes start from has changed"


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
    t.pin_spot()
    way = a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None and (harness.taken, harness.spots) == (1, 1)
    with caplog.at_level(logging.INFO):
        for start in (2, 3):
            asked = counting(t, monkeypatch)  # Start again: back to the spot, saved again
            assert (harness.taken, harness.spots) == (start, 1)
            assert lab.saved.count(module.CHECKPOINT_SLOT) == start
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


def test_with_no_pinned_spot_a_start_takes_the_spot_the_game_is_at(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """As before: the game is wherever the last go left it, and the way from the old
    spot does not lead from there."""
    t = lab.training()
    lab.connect()
    way = a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None and (harness.taken, harness.spots) == (1, 1)
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
        assert (harness.taken, harness.spots) == (2, 2)
        assert t.step()
        go = t.episode
        assert go is not None and go.remembered == 0 and t.remembered == 0
    said = [rec.getMessage() for rec in caplog.records if CHANGED in rec.getMessage()]
    assert said == [
        f"{CHANGED}, so the way the AI remembered from the old one ({way} steps) is forgotten."
    ]
    t.stop()


def test_a_pinned_spot_that_was_moved_is_another_spot(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    t = lab.training()
    game = lab.connect()
    t.pin_spot()
    way = a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None
    # The child takes the game somewhere else and pins that.
    t.unpin_spot()
    t.pin_spot()  # wherever the go left it
    played_on(lab)
    here = game.state.x
    t.pin_spot()
    played_on(lab)
    assert game.state.x != here
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
        assert harness.spots == 2 and game.state.x == here
        assert t.step()
        go = t.episode
        assert go is not None and go.remembered == 0 and t.remembered == 0
    assert sum(CHANGED in rec.getMessage() for rec in caplog.records) == 1
    assert f"({way} steps) is forgotten" in caplog.text
    t.stop()
    # And from the new spot on it is one spot again.
    goes_from_here = harness.spots
    counting(t, monkeypatch)
    assert harness.spots == goes_from_here
    t.stop()


def test_unpinned_again_the_next_start_is_where_the_game_is(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = lab.training()
    lab.connect()
    t.pin_spot()
    way = a_way(t, monkeypatch)
    harness = t.harness
    assert harness is not None and t.remembered == way
    t.unpin_spot()
    counting(t, monkeypatch)  # no spot to go back to: the game is where the go left it
    assert harness.spots == 2
    assert t.step()
    go = t.episode
    assert go is not None and go.remembered == 0 and t.remembered == 0
    t.stop()
