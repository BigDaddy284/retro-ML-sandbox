"""T-508: a pinned spot (DESIGN.md §9 checkpoint training)."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import Capabilities, ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.learning import QLearningAgent, Settings
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.training import Mode, Status
from retro_ai_lab.training.service import PRACTISE_SLOT
from tests.unit.training.test_service import Lab


def _play_on(lab: Lab, frames: int = 40) -> None:
    lab.play(["up"], frames)


def test_pin_restarts_every_go_and_every_start_from_the_spot(lab: Lab) -> None:
    t = lab.training
    with pytest.raises(ConfigurationError, match="Connect a game"):
        t.pin_spot()
    lab.connect()
    assert not t.spot_pinned
    t.unpin_spot()  # harmless when nothing is pinned
    assert [c.reason for c in lab.changes] == []
    _play_on(lab)
    spot = lab.adapter().game.ram
    t.pin_spot()
    assert t.spot_pinned and lab.changes[-1].reason == "spot pinned"
    assert lab.adapter().slots and PRACTISE_SLOT in lab.adapter().slots  # type: ignore[attr-defined]
    _play_on(lab)  # the child wanders off
    assert lab.adapter().game.ram != spot
    t.start(Mode.INDEPENDENT)
    t.step()
    assert lab.observation.ring[-2].get("ram") == spot  # go 1 began at the spot
    while t.episodes < 1:
        t.step()
    t.step()  # go 2 begins at the spot too
    assert lab.observation.ring[-2].get("ram") == spot
    with pytest.raises(ConfigurationError, match="Stop training"):
        t.pin_spot()
    t.stop()
    _play_on(lab)
    t.start()
    t.step()
    assert lab.observation.ring[-2].get("ram") == spot  # a later Start restarts there
    t.stop()
    t.unpin_spot()
    assert not t.spot_pinned and lab.changes[-1].reason == "spot forgotten"
    # The child waits a while with nothing held: driving on from wherever the AI's last
    # step left the car can end the game, and a Start on a finished game restarts it (T-983).
    lab.play([], 40)
    here = lab.adapter().game.ram
    assert here != spot and not lab.adapter().game.state.game_over
    t.start()
    t.step()
    assert lab.observation.ring[-2].get("ram") == here  # back to "wherever the game is"
    t.stop()


def test_pin_needs_save_states_and_dies_with_the_link(lab: Lab) -> None:
    t = lab.training
    lab.connect(
        capabilities=Capabilities(
            memory_read=True, screen_capture=True, frame_advance=True, input=True
        )
    )
    with pytest.raises(UnsupportedCapabilityError, match="no save states"):
        t.pin_spot()
    assert not t.spot_pinned
    lab.connection.disconnect()
    lab.connect()
    t.pin_spot()
    assert t.spot_pinned
    lab.connection.disconnect()
    assert not t.spot_pinned and t.status is Status.IDLE
    lab.connect()
    assert not t.spot_pinned  # the emulator's slots don't outlive the link
    t.forget()
    t.pin_spot()
    t.forget()  # the brain is forgotten; the spot is about the game and stays
    assert t.spot_pinned


def test_a_race_leaves_the_game_at_the_spot(lab: Lab) -> None:
    t = lab.training
    lab.connect()
    _play_on(lab)
    t.pin_spot()
    spot = lab.adapter().game.ram
    _play_on(lab)
    space = ActionSpace.from_buttons(lab.adapter().buttons())
    entrants = [("A", QLearningAgent(space, Settings(), seed=1))]
    results = t.compete(entrants, goes=1, max_steps=5)
    assert results["A"][0].steps == 5
    assert lab.adapter().game.ram == spot
