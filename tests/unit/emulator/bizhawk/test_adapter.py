"""T-111: BizHawk adapter connect, handshake, disconnect and dropped links."""

from __future__ import annotations

import re
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
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.bizhawk import (
    PROTOCOL_VERSION,
    SCRIPT_PATH,
    BizHawkAdapter,
    BizHawkSimulator,
)
from retro_ai_lab.emulator.bizhawk.simulator import ROM_NAME, SYSTEM
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.emulator.registry import AdapterRegistry
from retro_ai_lab.observation.memory import MemoryCapture
from retro_ai_lab.services.connection import ConnectionService, EmulatorDisconnected


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
        # capabilities follow the simulator's commands: all of them now
        assert adapter.capabilities() == Capabilities.all()
        assert adapter.description.commands == {
            "hello",
            "ping",
            "mode",
            "bye",
            "screenshot",
            "input",
            "read",
            "step",
            "speed",
            "save",
            "load",
        }
        assert adapter.buttons() == tuple(Button(b.name, b.label) for b in BUTTONS)
        assert adapter.description.button_names["start"] == "Start"
        assert adapter.description.regions == (MemoryRegion("ram", 32, description="RAM"),)
        assert adapter.memory_regions() == adapter.description.regions
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
        with pytest.raises(ProtocolError, match="Load a game in the emulator first"):
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


def test_read_memory_through_the_simulator() -> None:
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port)
        adapter.connect(timeout_s=1.0)
        assert adapter.capabilities().memory_read
        assert adapter.memory_regions() == (MemoryRegion("ram", 32, description="RAM"),)
        assert adapter.read_memory("ram", 0, 32) == sim.game.ram
        assert adapter.read_memory("ram", 5, 3) == sim.game.ram[5:8]
        assert adapter.read_memory("ram", 32, 0) == b""
        assert sim.requests[-1]["reads"] == [{"domain": "RAM", "addr": 32, "len": 0}]
        sent = len(sim.requests)
        with pytest.raises(ConfigurationError, match="Unknown memory region"):
            adapter.read_memory("vram", 0, 1)
        with pytest.raises(ConfigurationError, match="outside region"):
            adapter.read_memory("ram", 30, 4)
        assert len(sim.requests) == sent  # the base class refused before sending
        # the bridge's own checks, spoken to directly
        for reads, message in [
            ([{"domain": "VRAM", "addr": 0, "len": 1}], "Unknown memory domain 'VRAM'"),
            ([{"domain": "RAM", "addr": -1, "len": 1}], "whole-number address"),
            ([{"domain": "RAM", "addr": 0, "len": 1.5}], "whole-number address"),
            ([{"domain": "RAM", "addr": 30, "len": 4}], "outside 'RAM' \\(size 32\\)"),
            ("RAM", "must be a list"),
        ]:
            with pytest.raises(EmulatorError, match=message):
                adapter._call("read", reads=reads)
        assert adapter.is_connected
        # the memory observer works on top of it unchanged
        capture = MemoryCapture(_connection_for(adapter))
        snapshot = capture.capture(frame=7)
        assert snapshot.regions == {"ram": sim.game.ram}
        adapter.disconnect()
    with BizHawkSimulator(short_reads=True) as sim:
        adapter = BizHawkAdapter(port=sim.port)
        adapter.connect(timeout_s=1.0)
        with pytest.raises(ConfigurationError, match="Backend returned 31 bytes"):
            adapter.read_memory("ram", 0, 32)
        adapter.disconnect()
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port, max_region_bytes=16)
        adapter.connect(timeout_s=1.0)
        assert not adapter.capabilities().memory_read
        assert adapter.description.regions == (MemoryRegion("ram", 32, description="RAM"),)
        with pytest.raises(UnsupportedCapabilityError):
            adapter.memory_regions()
        adapter.disconnect()
    for bad in ({"max_region_bytes": 0}, {"max_region_bytes": True}, {"max_region_bytes": 2.5}):
        with pytest.raises(ConfigurationError):
            BizHawkAdapter(**bad)  # type: ignore[arg-type]


def _connection_for(adapter: BizHawkAdapter) -> ConnectionService:
    """A connection service already holding ``adapter`` (connected)."""
    registry = AdapterRegistry()
    registry.register(_Held)
    _Held.instance = adapter
    service = ConnectionService(registry, EventBus())
    service.connect("held")
    return service


class _Held(BizHawkAdapter):
    """Registry entry that hands out an existing adapter instead of a new one."""

    name = "held"
    display_name = "Held"
    instance: BizHawkAdapter

    def __new__(cls) -> BizHawkAdapter:  # type: ignore[misc]
        return cls.instance


def test_set_input_through_the_simulator() -> None:
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port)
        adapter.connect(timeout_s=1.0)
        assert adapter.capabilities().input
        assert adapter.held == frozenset()
        up, a = Button("up", "Up"), Button("a", "A")
        adapter.set_input({up, a})
        assert adapter.held == {up, a}
        expected = dict.fromkeys(sim.held, False)
        expected.update({"Up": True, "A": True})
        assert sim.held == expected
        assert sim.requests[-1]["buttons"] == expected  # the whole pad, every time
        with pytest.raises(ConfigurationError, match="Unknown buttons"):
            adapter.set_input({Button("turbo", "Turbo")})
        assert sim.requests[-1]["cmd"] == "input" and adapter.held == {up, a}
        adapter.set_input(())
        assert adapter.held == frozenset() and not any(sim.held.values())
        adapter.set_input({up})
        adapter.disconnect()
        assert adapter.held == frozenset()
        wait_until(lambda: not any(sim.held.values()))


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


def test_frame_advance_speed_and_states_through_the_simulator() -> None:
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port, call_timeout_s=2.0)
        adapter.connect(timeout_s=2.0)
        game = sim.game
        start = game.state.frame
        adapter.set_input({Button("up", "Up")})
        adapter.advance_frames(5)
        assert game.state.frame == start + 5
        assert game.state.speed > 0  # the held button was applied each frame
        assert sim.requests[-1] == {"id": sim.requests[-1]["id"], "cmd": "step", "frames": 5}
        assert sim.paused  # still held between steps
        adapter.set_input(())
        adapter.set_speed(1.5)
        assert sim.speed == 150
        adapter.set_speed(0.001)
        assert sim.speed == 1  # never below one percent
        adapter.save_state("lap_1")
        assert sim.slots == ("lap_1",)
        before = game.snapshot()
        adapter.advance_frames(20)
        assert game.snapshot() != before
        adapter.load_state("lap_1")
        assert game.snapshot() == before
        with pytest.raises(ConfigurationError, match=r"No save state in slot 'lap_2'"):
            adapter.load_state("lap_2")
        assert adapter.is_connected
        # the base class validates the arguments before anything is sent
        count = len(sim.requests)
        for call in (
            lambda: adapter.advance_frames(0),
            lambda: adapter.set_speed(0),
            lambda: adapter.save_state("bad slot"),
            lambda: adapter.load_state(""),
        ):
            with pytest.raises(ConfigurationError):
                call()
        assert len(sim.requests) == count
        # and the script's own refusals come back as emulator errors
        with pytest.raises(EmulatorError, match="whole number from 1 to 3600"):
            adapter._call("step", frames=4000)
        with pytest.raises(EmulatorError, match="letters, digits and underscores"):
            adapter._call("save", slot="no/slashes")
        with pytest.raises(EmulatorError, match="number above zero"):
            adapter._call("speed", percent="fast")
        adapter.disconnect()


def test_free_run_switch_changes_capabilities() -> None:
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port, call_timeout_s=2.0)
        with pytest.raises(EmulatorDisconnectedError):
            adapter.set_free_run(True)
        adapter.connect(timeout_s=2.0)
        assert not adapter.free_run and adapter.capabilities().frame_advance
        with pytest.raises(ConfigurationError):
            adapter.set_free_run("yes")  # type: ignore[arg-type]
        adapter.set_free_run(True)
        assert adapter.free_run and sim.free_run and not sim.paused
        assert not adapter.capabilities().frame_advance
        with pytest.raises(UnsupportedCapabilityError):
            adapter.advance_frames(1)
        adapter.set_free_run(False)
        assert not adapter.free_run and sim.paused
        assert adapter.capabilities() == Capabilities.all()
        adapter.advance_frames(1)
        adapter.disconnect()


def test_episode_runner_drives_the_simulator() -> None:
    from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
    from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings, StateEncoder
    from retro_ai_lab.services import DiscoveryService, DiscoveryUpdated, ObservationService
    from retro_ai_lab.training import EpisodeRunner
    from retro_ai_lab.understanding import Encoding, Evidence, VariableKind

    sure = [Evidence("test", "", 6.0)]
    with BizHawkSimulator() as sim:
        registry = AdapterRegistry()
        registry.register(BizHawkAdapter)
        bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
        connection = ConnectionService(registry, bus)
        observation = ObservationService(connection, bus)
        discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
        connection.connect("bizhawk", port=sim.port, call_timeout_s=2.0)
        k = discovery.knowledge
        assert k is not None
        added = (
            k.propose("ram", LAYOUT.lane, kind=VariableKind.POSITION, name="Lane", evidence=sure),
            k.propose("ram", LAYOUT.speed, kind=VariableKind.SPEED, name="Speed", evidence=sure),
            k.propose(
                "ram",
                LAYOUT.score,
                encoding=Encoding.U16LE,
                kind=VariableKind.SCORE,
                name="Score",
                evidence=sure,
            ),
        )
        bus.publish(DiscoveryUpdated(added, (), (), ()))
        agent = QLearningAgent(ActionSpace.from_buttons(BUTTONS), Settings(epsilon=0.5), seed=1)
        runner = EpisodeRunner(
            connection, observation, bus, agent, StateEncoder(k), frames_per_step=3, max_steps=20
        )
        start = sim.game.state.frame
        result = runner.run_episode()
        assert result.steps == 20 and result.frames == 60 and result.ended_by == "max_steps"
        assert sim.game.state.frame == start + 60
        assert [r["cmd"] for r in sim.requests].count("step") == 20
        assert observation.inputs.by_source("ai")
        connection.disconnect()


def test_silent_drop_is_noticed_by_the_watchdog_check() -> None:
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port, call_timeout_s=2.0)
        assert not adapter.check_link()
        adapter.connect(timeout_s=2.0)
        service = _connection_for(adapter)
        losses: list[EmulatorDisconnected] = []
        service._bus.subscribe(EmulatorDisconnected, losses.append)
        sent = len(sim.requests)
        for _ in range(3):
            assert service.check()  # healthy and idle: nothing goes over the wire
        time.sleep(0.05)
        assert len(sim.requests) == sent
        sim.drop_client()
        wait_until(lambda: not service.check())
        assert not adapter.is_connected and adapter.hello == {}
        assert len(losses) == 1 and not losses[0].expected
        assert not service.check()  # reported once, stays quiet
        assert len(losses) == 1


def test_missing_emulator_advice_names_the_script() -> None:
    with BizHawkSimulator() as sim:
        port = sim.port
    adapter = BizHawkAdapter(port=port, call_timeout_s=1.0)
    with pytest.raises(EmulatorError, match=re.escape(str(SCRIPT_PATH))) as info:
        adapter.connect(timeout_s=1.0)
    assert "Couldn't find BizHawk" in str(info.value)
    assert not adapter.is_connected
