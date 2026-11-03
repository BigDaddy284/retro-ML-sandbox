"""T-811: the mGBA adapter over the real script (lupa) and the protocol simulator."""

from __future__ import annotations

import re
import socket

import pytest

from retro_ai_lab.core import Button, EmulatorError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.bizhawk import BizHawkAdapter, BizHawkSimulator
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, LAYOUT
from retro_ai_lab.emulator.mgba import MGBA_HINT, SCRIPT_PATH, MgbaAdapter
from retro_ai_lab.emulator.registry import AdapterRegistry, default_registry
from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings, StateEncoder
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    ObservationService,
)
from retro_ai_lab.training import EpisodeRunner
from retro_ai_lab.understanding import Encoding, Evidence, VariableKind
from tests.unit.emulator.mgba.lua_host import MgbaScriptHost


def test_adapter_drives_the_real_script() -> None:
    with MgbaScriptHost() as host:
        adapter = MgbaAdapter(port=host.port, call_timeout_s=5.0)
        assert adapter.hold  # until a handshake says otherwise
        adapter.connect(timeout_s=5.0)
        assert adapter.name == "mgba" and adapter.display_name == "mGBA"
        assert adapter.description.bizhawk == "mGBA"
        assert not adapter.hold and not adapter.free_run
        info = adapter.game_info()
        assert (info.system, info.title) == ("GBA", "TOYRACER")
        caps = adapter.capabilities()
        assert caps.frame_advance and caps.input and caps.memory_read and caps.save_states
        assert caps.screen_capture and not caps.speed_control
        adapter.set_input({Button("up", "UP")})
        before = adapter._call("ping")["frame"]
        adapter.advance_frames(3)  # a step of 3 frames, answered once they have passed
        assert adapter._call("ping")["frame"] >= before + 3 and host.pressed == ("up",)
        assert len(adapter.read_memory("wram", 0, 4)) == 4
        adapter.disconnect()
        assert adapter.hold
        assert host.error is None


def test_connect_failure_names_mgba_and_its_script() -> None:
    with socket.socket() as probe:  # a port nobody is listening on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(EmulatorError) as caught:
        MgbaAdapter(port=port).connect(timeout_s=0.5)
    message = str(caught.value)
    assert message.startswith(f"Couldn't find mGBA on port {port}. {MGBA_HINT}")
    assert re.search(re.escape(str(SCRIPT_PATH)), message) and "mgba" in str(SCRIPT_PATH)
    with pytest.raises(EmulatorError, match="Couldn't find BizHawk on port") as other:
        BizHawkAdapter(port=port).connect(timeout_s=0.5)
    assert "EmuHawk" in str(other.value) and "bizhawk" in str(other.value)


def test_episode_runner_over_the_simulator_through_the_mgba_adapter() -> None:
    sure = [Evidence("test", "", 6.0)]
    with BizHawkSimulator() as sim:
        registry = AdapterRegistry()
        registry.register(MgbaAdapter)
        bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
        connection = ConnectionService(registry, bus)
        observation = ObservationService(connection, bus)
        discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
        connection.connect("mgba", port=sim.port, call_timeout_s=2.0)
        adapter = connection.adapter
        assert isinstance(adapter, MgbaAdapter) and adapter.hold  # the simulator holds
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
            connection, observation, bus, agent, StateEncoder(k), frames_per_step=3, max_steps=10
        )
        start = sim.game.state.frame
        result = runner.run_episode()
        assert result.steps == 10 and result.frames == 30 and result.ended_by == "max_steps"
        assert sim.game.state.frame == start + 30
        connection.disconnect()


def test_default_registry_offers_mgba_third() -> None:
    reg = default_registry()
    assert [(i.name, i.display_name) for i in reg.describe()][2] == ("mgba", "mGBA")
