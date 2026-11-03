"""T-111: BizHawk adapter connect, handshake, disconnect and dropped links."""

from __future__ import annotations

import time

import pytest

from retro_ai_lab.core import (
    Button,
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
    MemoryRegion,
    ProtocolError,
    UnsupportedCapabilityError,
)
from retro_ai_lab.emulator.bizhawk import (
    PROTOCOL_VERSION,
    SCRIPT_PATH,
    BizHawkAdapter,
    BizHawkSimulator,
)
from retro_ai_lab.emulator.bizhawk.simulator import ROM_NAME, SYSTEM
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS


def wait_until(check: object, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():  # type: ignore[operator]
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def test_handshake_disconnect_and_idempotence() -> None:
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port, call_timeout_s=2.0)
        assert not adapter.is_connected and adapter.hello == {}
        adapter.connect(timeout_s=2.0)
        assert adapter.is_connected
        assert adapter.hello["script"] == PROTOCOL_VERSION
        assert adapter.hello["system"] == SYSTEM
        assert sim.paused  # hold mode: the emulator waits for the app
        info = adapter.game_info()
        assert info.system == SYSTEM and info.title == ROM_NAME
        assert info.rom_hash == "toyracer-0"
        # capabilities follow the simulator's commands: only screenshots so far
        assert adapter.capabilities() == Capabilities(screen_capture=True)
        assert adapter.description.commands == {"hello", "ping", "mode", "bye", "screenshot"}
        assert adapter.buttons() == tuple(Button(b.name, b.label) for b in BUTTONS)
        assert adapter.description.button_names["start"] == "Start"
        assert adapter.description.regions == (MemoryRegion("ram", 32, description="RAM"),)
        with pytest.raises(UnsupportedCapabilityError):  # claimed only once 'read' exists
            adapter.memory_regions()
        adapter.connect(timeout_s=2.0)  # no-op
        assert sim.connections == 1
        adapter.ping()
        assert [r["cmd"] for r in sim.requests] == ["hello", "ping"]
        assert sim.requests[0]["script"] == PROTOCOL_VERSION
        adapter.disconnect()
        assert not adapter.is_connected and adapter.hello == {}
        assert sim.requests[-1]["cmd"] == "bye"
        wait_until(lambda: not sim.paused)
        adapter.disconnect()  # idempotent
        with pytest.raises(EmulatorDisconnectedError):
            adapter.ping()
        # and it can come back
        adapter.connect(timeout_s=2.0)
        assert adapter.is_connected and sim.connections == 2
        adapter.disconnect()


def test_no_bridge_and_silent_bridge() -> None:
    with BizHawkSimulator() as sim:
        port = sim.port
    adapter = BizHawkAdapter(port=port)
    with pytest.raises(EmulatorError, match="Couldn't find BizHawk"):
        adapter.connect(timeout_s=1.0)
    assert not adapter.is_connected
    with BizHawkSimulator(answer=False) as sim:
        adapter = BizHawkAdapter(port=sim.port)
        with pytest.raises(EmulatorError, match=r"did not answer within 0\.3 seconds"):
            adapter.connect(timeout_s=0.3)
        assert not adapter.is_connected and adapter.hello == {}
        wait_until(lambda: sim.connections == 1)
    with BizHawkSimulator(no_game=True) as sim:
        adapter = BizHawkAdapter(port=sim.port)
        with pytest.raises(ProtocolError, match="Load a game in EmuHawk first"):
            adapter.connect(timeout_s=1.0)
        assert not adapter.is_connected
        with pytest.raises(EmulatorDisconnectedError):
            adapter.description  # noqa: B018 - the property is the check


def test_wrong_version_and_garbage_are_protocol_errors() -> None:
    with BizHawkSimulator(script_version=PROTOCOL_VERSION + 1) as sim:
        adapter = BizHawkAdapter(port=sim.port)
        with pytest.raises(ProtocolError, match="speaks protocol 2 but this app needs 1"):
            adapter.connect(timeout_s=1.0)
        assert not adapter.is_connected
    with BizHawkSimulator(garbage=True) as sim:
        adapter = BizHawkAdapter(port=sim.port)
        with pytest.raises(ProtocolError, match="not JSON"):
            adapter.connect(timeout_s=1.0)
        assert not adapter.is_connected


def test_dropped_link_is_noticed_on_the_next_call() -> None:
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port)
        adapter.connect(timeout_s=1.0)
        sim.drop_client()
        with pytest.raises(EmulatorDisconnectedError):
            adapter.ping()
        assert not adapter.is_connected and adapter.hello == {}
    with BizHawkSimulator(drop_after=1) as sim:  # answers hello, then hangs up
        adapter = BizHawkAdapter(port=sim.port)
        adapter.connect(timeout_s=1.0)
        with pytest.raises(EmulatorDisconnectedError):
            adapter.ping()
        assert not adapter.is_connected


def test_capture_frame_through_the_simulator() -> None:
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port)
        adapter.connect(timeout_s=1.0)
        assert adapter.capabilities().screen_capture
        assert "screenshot" in adapter.description.commands
        expected = sim.game.render()
        frame = adapter.capture_frame()
        assert (frame.width, frame.height, frame.channels) == (expected.width, expected.height, 3)
        assert frame.pixels == bytes(v for v in expected.pixels for _ in range(3))
        assert frame.number == sim.game.state.frame
        adapter.disconnect()
    with BizHawkSimulator(screenshot=None) as sim:
        adapter = BizHawkAdapter(port=sim.port)
        adapter.connect(timeout_s=1.0)
        with pytest.raises(EmulatorError, match="did not take a screenshot"):
            adapter.capture_frame()
        assert adapter.is_connected  # a refused screenshot is not a lost link
        adapter.disconnect()
    with BizHawkSimulator(screenshot="garbage") as sim:
        adapter = BizHawkAdapter(port=sim.port)
        adapter.connect(timeout_s=1.0)
        with pytest.raises(ProtocolError, match="not valid base64"):
            adapter.capture_frame()
        adapter.disconnect()


def test_options_are_validated_and_script_ships() -> None:
    adapter = BizHawkAdapter()
    assert (adapter.host, adapter.port) == ("127.0.0.1", 43110)
    assert BizHawkAdapter.name == "bizhawk" and BizHawkAdapter.display_name
    for bad in ({"host": ""}, {"port": 0}, {"port": 70000}, {"port": True}, {"call_timeout_s": 0}):
        with pytest.raises(ConfigurationError):
            BizHawkAdapter(**bad)  # type: ignore[arg-type]
    assert SCRIPT_PATH.is_file()
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert f"SCRIPT_VERSION = {PROTOCOL_VERSION}" in text
    assert "43110" in text
