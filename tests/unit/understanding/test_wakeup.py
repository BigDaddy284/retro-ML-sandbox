"""T-844/T-851: the AI works out which button gets the game going, by experiment."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import _BUTTON_BIT, BUTTON_NAMES, LAYOUT, ToyRacer
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.services import ConnectionService
from retro_ai_lab.understanding import find_wake_button, inputs_matter, probe_inputs
from retro_ai_lab.understanding.wakeup import DEFAULT_PRESSES, START_SLOT, WAKE_SLOT

TITLE = 0xEE  # the lap byte's value while the two-press racer shows its "title screen"


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


class TwoPressRacer(ToyRacer):
    """A Toy Racer whose game-over screen needs Start twice, like a title demo.

    The first (fresh) press of Start reaches a "title screen": a lasting change
    (the lap byte) after which the pad is still ignored. The next fresh press
    restarts the game. With ``presses_to_start`` above 2 every fresh press
    just changes the title byte and the game never starts.
    """

    def __init__(self, seed: int = 0, presses_to_start: int = 2) -> None:
        super().__init__(seed)
        self.presses_to_start = presses_to_start

    def step(self, pressed: Iterable[str] = ()) -> None:
        pressed = frozenset(pressed)
        held = pressed
        ram = self._ram
        fresh = "start" in held and not ram[LAYOUT.input] & _BUTTON_BIT["start"]
        if ram[LAYOUT.game_over] and "start" in held:
            if fresh:  # the title screen reacts to each press, but stays a title screen
                ram[LAYOUT.lap] = TITLE if ram[LAYOUT.lap] < TITLE else ram[LAYOUT.lap] + 1
            if not fresh or ram[LAYOUT.lap] - TITLE + 1 < self.presses_to_start:
                held = held - {"start"}
        super().step(held)
        if "start" in pressed:  # the pad mirror still records the press
            ram[LAYOUT.input] |= _BUTTON_BIT["start"]


def two_press(connection: ConnectionService, presses_to_start: int = 2) -> TwoPressRacer:
    adapter = adapter_of(connection)
    game = TwoPressRacer(3, presses_to_start)
    adapter._game = game
    drive_to_game_over(connection)
    assert game.state.game_over and game.state.lap < TITLE
    return game


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


def test_a_button_that_needs_two_presses_is_found(
    connection: ConnectionService, caplog: pytest.LogCaptureFixture
) -> None:
    game = two_press(connection)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.understanding.wakeup")
    playing = [n for n in BUTTON_NAMES if n != "start"]
    found = find_wake_button(connection, BUTTON_NAMES, playing, wait=4, per_button=2)
    assert found == "start"
    assert not game.state.game_over  # left running after the second press
    assert "Pressing Start 2 times gets the game going." in caplog.text


def test_one_press_only_misses_it(
    connection: ConnectionService, caplog: pytest.LogCaptureFixture
) -> None:
    game = two_press(connection)
    before = game.snapshot()
    caplog.set_level(logging.INFO, logger="retro_ai_lab.understanding.wakeup")
    playing = [n for n in BUTTON_NAMES if n != "start"]
    assert find_wake_button(connection, ["start"], playing, wait=4, presses=1) is None
    assert game.state.game_over and game.snapshot() == before
    assert "No button got the game going: tried start." in caplog.text


def test_a_button_that_keeps_changing_things_is_not_pressed_forever(
    connection: ConnectionService,
) -> None:
    game = two_press(connection, presses_to_start=DEFAULT_PRESSES + 5)
    before = game.snapshot()
    playing = [n for n in BUTTON_NAMES if n != "start"]
    assert find_wake_button(connection, ["start"], playing, wait=4) is None
    assert game.state.game_over and game.snapshot() == before
    # ... and with enough presses allowed it is found after exactly that many.
    game = two_press(connection, presses_to_start=4)
    assert find_wake_button(connection, ["start"], playing, wait=4, presses=4) == "start"
    assert not game.state.game_over


def test_the_probe_names_what_moves_by_itself(connection: ConnectionService) -> None:
    """T-855: the timer and frame counter move whatever is held; the score needs the pad."""
    game = adapter_of(connection).game
    playing = [n for n in BUTTON_NAMES if n != "start"]
    before = game.snapshot()
    probe = probe_inputs(connection, playing, per_button=4)
    assert game.snapshot() == before  # the game is put back where it was
    assert probe.matters
    offsets = {o for region, o in probe.self_moving if region == "ram"}
    assert {LAYOUT.frame, LAYOUT.timer} <= offsets
    assert set(range(LAYOUT.noise, LAYOUT.noise + 4)) <= offsets  # seeded noise replays
    assert not {LAYOUT.score, LAYOUT.score + 1, LAYOUT.lives, LAYOUT.position} & offsets
    # Coasting, the score and position move by themselves but answer to the pad (B
    # brakes, Up speeds up), so they are not "self-moving": a racer's score is real.
    for _ in range(8):
        game.step(["up"])
    probe = probe_inputs(connection, playing, per_button=4)
    offsets = {o for region, o in probe.self_moving if region == "ram"}
    assert {LAYOUT.frame, LAYOUT.timer} <= offsets
    assert not {LAYOUT.score, LAYOUT.position} & offsets


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
        find_wake_button(connection, ["up"], ["up"], presses=0)
    with pytest.raises(ConfigurationError):
        find_wake_button(connection, ["up", "fire"], ["up"])
    with pytest.raises(ConfigurationError):
        find_wake_button(connection, ["up"], ["fire"])
