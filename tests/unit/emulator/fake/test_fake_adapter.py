"""T-005: FakeEmulatorAdapter — the contract suite plus fake-specific behaviour."""

from __future__ import annotations

from typing import Any

import pytest

from retro_ai_lab.core import (
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
    GameInfo,
)
from retro_ai_lab.emulator.base import EmulatorAdapter
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter, ToyRacer
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, LAYOUT, RAM_SIZE
from retro_ai_lab.testing import AdapterContract

BY_NAME = {b.name: b for b in BUTTONS}


class TestFakeAdapterContract(AdapterContract):
    def make_adapter(self, capabilities: Capabilities | None = None) -> EmulatorAdapter:
        return FakeEmulatorAdapter(seed=3, capabilities=capabilities)


class TestFakeHopperContract(AdapterContract):
    """T-985: the contract every adapter is held to, with Toy Hopper in the console."""

    def make_adapter(self, capabilities: Capabilities | None = None) -> EmulatorAdapter:
        return FakeEmulatorAdapter(seed=3, capabilities=capabilities, game="hopper")


@pytest.fixture
def fake() -> FakeEmulatorAdapter:
    a = FakeEmulatorAdapter(seed=3)
    a.connect()
    return a


# ------------------------------------------------------------- identity


def test_identity_and_game_info(fake: FakeEmulatorAdapter) -> None:
    assert fake.name == "fake"
    assert fake.game_info() == GameInfo("TOY", "Toy Racer", rom_hash="toyracer-3")
    assert fake.buttons() == BUTTONS
    assert fake.capabilities() == Capabilities.all()


def test_memory_is_the_games_ram(fake: FakeEmulatorAdapter) -> None:
    (region,) = fake.memory_regions()
    assert region.size == RAM_SIZE
    assert fake.read_memory("ram", 0, RAM_SIZE) == fake.game.ram
    assert fake.read_memory("ram", LAYOUT.lives, 1) == bytes([fake.game.state.lives])


# ---------------------------------------------------------- behaviour


def test_inputs_are_held_until_changed(fake: FakeEmulatorAdapter) -> None:
    fake.set_input([BY_NAME["up"]])
    fake.advance_frames(3)
    assert fake.game.state.speed == 3
    fake.advance_frames(2)
    assert fake.game.state.speed == 4  # still accelerating: input is held
    fake.set_input(())
    fake.advance_frames(5)
    assert fake.game.state.speed == 4  # coasting


def test_adapter_matches_direct_game_play() -> None:
    a = FakeEmulatorAdapter(seed=11)
    a.connect()
    g = ToyRacer(11)
    a.set_input([BY_NAME["up"], BY_NAME["left"]])
    a.advance_frames(40)
    for _ in range(40):
        g.step(("up", "left"))
    assert a.read_memory("ram", 0, RAM_SIZE) == g.ram
    assert a.capture_frame() == g.render()


def test_frame_number_tracks_game_frames(fake: FakeEmulatorAdapter) -> None:
    assert fake.capture_frame().number == 0
    fake.advance_frames(7)
    assert fake.capture_frame().number == 7


def test_speed_is_recorded(fake: FakeEmulatorAdapter) -> None:
    assert fake.speed == 1.0
    fake.set_speed(8)
    assert fake.speed == 8.0


def test_save_states_are_per_slot(fake: FakeEmulatorAdapter) -> None:
    fake.save_state("start")
    fake.set_input([BY_NAME["up"]])
    fake.advance_frames(10)
    fake.save_state("later")
    assert fake.slots == ("later", "start")
    later_ram = fake.game.ram
    fake.load_state("start")
    assert fake.game.state.frame == 0
    fake.load_state("later")
    assert fake.game.ram == later_ram


def test_disconnect_releases_held_input(fake: FakeEmulatorAdapter) -> None:
    fake.set_input([BY_NAME["up"]])
    fake.disconnect()
    assert fake.held == frozenset()
    fake.connect()
    fake.advance_frames(1)
    assert fake.game.state.speed == 0


def test_reduced_capabilities_are_honoured() -> None:
    a = FakeEmulatorAdapter(capabilities=Capabilities(screen_capture=True, input=True))
    a.connect()
    assert a.capabilities().enabled() == ("screen_capture", "input")
    a.capture_frame()
    with pytest.raises(Exception, match="memory_read"):
        a.memory_regions()


# ------------------------------------------------------- fault injection


def test_refused_connection() -> None:
    a = FakeEmulatorAdapter(fail_connect=EmulatorError("Nothing is listening."))
    with pytest.raises(EmulatorError, match="listening"):
        a.connect()
    assert not a.is_connected


def test_slow_connection_times_out() -> None:
    a = FakeEmulatorAdapter(connect_delay_s=2.0)
    with pytest.raises(EmulatorError, match="within 1 seconds"):
        a.connect(timeout_s=1)
    assert not a.is_connected
    a.connect(timeout_s=3)
    assert a.is_connected


def test_link_drops_after_n_operations() -> None:
    a = FakeEmulatorAdapter(drop_after_ops=2)
    a.connect()
    a.capture_frame()
    a.capture_frame()
    assert a.operations == 2
    with pytest.raises(EmulatorDisconnectedError, match="stopped responding"):
        a.capture_frame()
    assert not a.is_connected
    with pytest.raises(EmulatorDisconnectedError):
        a.buttons()
    # Reconnecting resets the budget.
    a.connect()
    a.capture_frame()
    assert a.operations == 1


def test_description_calls_do_not_count_as_operations(fake: FakeEmulatorAdapter) -> None:
    fake.capabilities()
    fake.game_info()
    fake.buttons()
    fake.memory_regions()
    assert fake.operations == 0


def test_drop_link_simulates_an_unexpected_disconnect(fake: FakeEmulatorAdapter) -> None:
    fake.drop_link()
    assert not fake.is_connected
    with pytest.raises(EmulatorDisconnectedError):
        fake.capture_frame()


@pytest.mark.parametrize("kwargs", [{"connect_delay_s": -1}, {"drop_after_ops": -1}])
def test_bad_fault_settings_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        FakeEmulatorAdapter(**kwargs)


def test_a_watched_step_answers_what_the_toy_pad_holds(fake: FakeEmulatorAdapter) -> None:
    """T-887: the fake game has no real controller; what is held is what it reports."""
    assert fake.step(1, watch=True).pressed == frozenset()
    fake.set_input({BY_NAME["up"], BY_NAME["a"]})
    result = fake.step(2, watch=True)
    assert result.pressed == frozenset({"up", "a"})
    assert fake.step(1).pressed is None


# ------------------------------------------------------------ T-985: Toy Hopper


def test_the_practice_console_runs_toy_hopper() -> None:
    from retro_ai_lab.emulator.fake import ToyHopper, toy_hopper
    from retro_ai_lab.emulator.fake.adapter import DEFAULT_GAME, GAMES

    assert GAMES == ("racer", "hopper") and DEFAULT_GAME == "racer"
    racer = FakeEmulatorAdapter(seed=4, game="racer")  # asked for by name: what it always was
    racer.connect()
    assert racer.game_info() == GameInfo("TOY", "Toy Racer", rom_hash="toyracer-4")
    assert tuple(racer.buttons()) == BUTTONS and isinstance(racer.game, ToyRacer)
    assert racer.display_name == FakeEmulatorAdapter.display_name == "Practice console (Toy Racer)"
    assert racer.memory_regions()[0].size == RAM_SIZE

    hopper = FakeEmulatorAdapter(seed=4, game="hopper")
    assert isinstance(hopper, EmulatorAdapter) and isinstance(hopper.game, ToyHopper)
    assert hopper.display_name == "Practice console (Toy Hopper)"
    hopper.connect()
    assert hopper.game_info() == GameInfo("TOY", "Toy Hopper", rom_hash="toyhopper-4")
    assert tuple(hopper.buttons()) == toy_hopper.BUTTONS
    (region,) = hopper.memory_regions()
    assert (region.name, region.size) == ("ram", toy_hopper.RAM_SIZE)
    assert hopper.capabilities() == Capabilities.all()
    # Held buttons drive the hopper exactly as the game played directly.
    direct = ToyHopper(4)
    by_name = {b.name: b for b in hopper.buttons()}
    for held, frames in ((["right"], 30), (["right", "a"], 20), ([], 10), (["left"], 8)):
        hopper.set_input([by_name[name] for name in held])
        hopper.advance_frames(frames)
        for _ in range(frames):
            direct.step(held)
        assert hopper.read_memory("ram", 0, toy_hopper.RAM_SIZE) == direct.ram
    assert hopper.read_memory("ram", toy_hopper.LAYOUT.lives, 1) == bytes([toy_hopper.START_LIVES])
    frame = hopper.capture_frame()
    assert (frame.width, frame.height, frame.channels) == (256, 240, 3)
    assert frame == direct.render() and frame.number == direct.state.frame == 68
    # A save state brings the exact game back.
    hopper.save_state("spot")
    saved = hopper.read_memory("ram", 0, toy_hopper.RAM_SIZE)
    hopper.set_input([by_name["right"]])
    hopper.advance_frames(40)
    assert hopper.read_memory("ram", 0, toy_hopper.RAM_SIZE) != saved
    hopper.load_state("spot")
    assert hopper.read_memory("ram", 0, toy_hopper.RAM_SIZE) == saved


@pytest.mark.parametrize("game", ["pong", "", None, 3, "Hopper"])
def test_an_unknown_practice_game_is_refused(game: Any) -> None:
    with pytest.raises(ConfigurationError, match=r"no game called .*choices: racer, hopper"):
        FakeEmulatorAdapter(game=game)


def test_the_services_connect_to_toy_hopper() -> None:
    from retro_ai_lab.core.events import EventBus
    from retro_ai_lab.emulator.fake import toy_hopper
    from retro_ai_lab.emulator.registry import default_registry
    from retro_ai_lab.services import ConnectionService, ObservationService

    bus = EventBus()
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    session = connection.connect("fake", game="hopper", seed=2)
    assert session.display_name == "Practice console (Toy Hopper)"
    assert session.game_info == GameInfo("TOY", "Toy Hopper", rom_hash="toyhopper-2")
    assert [b.name for b in session.buttons] == list(toy_hopper.BUTTON_NAMES)
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    adapter.set_input([b for b in adapter.buttons() if b.name == "right"])
    for _ in range(12):
        adapter.advance_frames(1)
        observation.advance(1)
    latest = observation.ring[-1]
    assert latest.get("ram") == adapter.game.ram and adapter.game.state.x > toy_hopper.START_X
    connection.disconnect()
    with pytest.raises(ConfigurationError, match="no game called"):
        connection.connect("fake", game="pong")
    assert connection.connect("fake").game_info.title == "Toy Racer"  # the default stands
