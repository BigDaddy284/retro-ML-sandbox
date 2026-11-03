"""T-844: the AI works out which button gets the game going, by experiment."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTON_NAMES
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.services import ConnectionService
from retro_ai_lab.understanding import find_wake_button, inputs_matter
from retro_ai_lab.understanding.wakeup import START_SLOT, WAKE_SLOT


@pytest.fixture
def connection() -> ConnectionService:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    connection.connect("fake", seed=3)
    return connection


def adapter_of(connection: ConnectionService) -> FakeEmulatorAdapter:
    a = connection.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    return a


def drive_to_game_over(connection: ConnectionService) -> None:
    game = adapter_of(connection).game
    while not game.state.game_over:
        game.step(["up"])


def test_inputs_matter_in_play_but_not_on_the_game_over_screen(
    connection: ConnectionService,
) -> None:
    game = adapter_of(connection).game
    assert inputs_matter(connection, BUTTON_NAMES)
    before = game.snapshot()
    assert inputs_matter(connection, BUTTON_NAMES, per_button=2)
    assert game.snapshot() == before  # the game is put back where it was
    drive_to_game_over(connection)
    # Only Start restarts a game that is over; the pad is otherwise ignored (the
    # controller mirror byte settles back to 0 in the idle frames at the end).
    assert not inputs_matter(connection, [n for n in BUTTON_NAMES if n != "start"])
    assert game.state.game_over
    assert inputs_matter(connection, BUTTON_NAMES)  # Start was among them


def test_find_wake_button_learns_start_from_the_game_over_screen(
    connection: ConnectionService, caplog: pytest.LogCaptureFixture
) -> None:
    game = adapter_of(connection).game
    drive_to_game_over(connection)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.understanding.wakeup")
    # Start is tried last, so every other button is ruled out first.
    playing = [n for n in BUTTON_NAMES if n not in ("start", "select")]
    order = [n for n in BUTTON_NAMES if n != "start"] + ["start"]
    found = find_wake_button(connection, order, playing, wait=4, per_button=2)
    assert found == "start"
    assert not game.state.game_over  # left running after the press
    assert "Pressing Start gets the game going." in caplog.text
    # A remembered button is tried first: one press, no search through the others.
    drive_to_game_over(connection)
    a = adapter_of(connection)
    before = a.stats.copy() if hasattr(a, "stats") else None
    found = find_wake_button(connection, order, playing, first="start", wait=4, per_button=2)
    assert found == "start"
    assert not game.state.game_over
    assert before is None or a.stats != before


def test_no_button_helps_leaves_the_game_where_it_was(
    connection: ConnectionService, caplog: pytest.LogCaptureFixture
) -> None:
    game = adapter_of(connection).game
    drive_to_game_over(connection)
    before = game.snapshot()
    caplog.set_level(logging.INFO, logger="retro_ai_lab.understanding.wakeup")
    playing = ["up", "down", "left", "right", "a", "b"]
    assert find_wake_button(connection, ["up", "down", "left"], playing, wait=4) is None
    assert game.state.game_over and game.snapshot() == before
    assert "No button got the game going: tried up, down, left." in caplog.text
    slots = adapter_of(connection)._slots
    assert START_SLOT in slots and WAKE_SLOT not in slots  # no press changed anything


def test_misuse(connection: ConnectionService) -> None:
    with pytest.raises(ConfigurationError):
        inputs_matter(connection, [])
    with pytest.raises(ConfigurationError):
        inputs_matter(connection, ["up"], per_button=0)
    with pytest.raises(ConfigurationError):
        inputs_matter(connection, ["fire"])
    with pytest.raises(ConfigurationError):
        find_wake_button(connection, [], ["up"])
    with pytest.raises(ConfigurationError):
        find_wake_button(connection, ["start"], [])
    with pytest.raises(ConfigurationError):
        find_wake_button(connection, ["up"], ["up"], hold=0)
    with pytest.raises(ConfigurationError):
        find_wake_button(connection, ["up", "fire"], ["up"])
    with pytest.raises(ConfigurationError):
        find_wake_button(connection, ["up"], ["fire"])
