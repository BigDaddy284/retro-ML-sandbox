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
