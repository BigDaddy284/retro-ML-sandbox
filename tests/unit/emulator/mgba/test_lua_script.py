"""T-810: the mGBA bridge script, run under lupa against stubs of mGBA's API."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import (
    Button,
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
    ProtocolError,
)
from retro_ai_lab.emulator.bizhawk import PROTOCOL_VERSION, BizHawkAdapter
from retro_ai_lab.emulator.fake.toy_racer import ToyRacer
from retro_ai_lab.emulator.mgba import SCRIPT_PATH
from tests.unit.emulator.mgba.lua_host import (
    CODE,
    GBA_KEYS,
    TITLE,
    MgbaScriptHost,
    load_internals,
)

COMMANDS = ["bye", "hello", "input", "load", "mode", "ping", "read", "save", "screenshot", "step"]


def wait_until(check: Any, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


def connected(host: MgbaScriptHost) -> BizHawkAdapter:
    """The protocol is BizHawk's, so its adapter drives the mGBA script (T-811 adds `mgba`)."""
    adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
    adapter.connect(timeout_s=5.0)
    return adapter


def test_handshake_free_run_and_reconnect() -> None:
    assert SCRIPT_PATH.is_file()
    with MgbaScriptHost() as host:
        assert host.log and b"waiting for the app on port" in host.log[0]
        assert host.callback_names() == ("frame", "keysRead", "stop")
        adapter = connected(host)
        hello = adapter.hello
        assert hello["script"] == PROTOCOL_VERSION and hello["emulator"] == "mGBA"
        assert hello["system"] == "GBA" and hello["hold"] is False
        assert hello["rom"] == TITLE.decode() and hello["code"] == CODE.decode()
        assert hello["hash"] == "ABCDEF01"
        assert hello["domains"] == [{"name": "iwram", "size": 32768}, {"name": "wram", "size": 32}]
        assert hello["buttons"] == sorted(k.decode() for k in GBA_KEYS)
        assert hello["commands"] == COMMANDS
        assert [b.name for b in adapter.buttons()] == [
            "a",
            "b",
            "down",
            "l",
            "left",
            "r",
            "right",
            "select",
            "start",
            "up",
        ]
        info = adapter.game_info()
        assert (info.system, info.title, info.rom_hash) == ("GBA", "TOYRACER", "ABCDEF01")
        caps = adapter.capabilities()
        assert caps.memory_read and caps.screen_capture and caps.input and caps.frame_advance
        assert caps.save_states and not caps.speed_control
        # free run: the game goes on between requests
        before = host.frames
        wait_until(lambda: host.frames > before + 5)
        adapter.ping()
        adapter.disconnect()
        wait_until(lambda: any(b"connected" in line for line in host.log))
        adapter.connect(timeout_s=5.0)
        assert adapter.is_connected
        adapter.disconnect()
        assert host.error is None


def test_memory_reads_through_the_real_script() -> None:
    with MgbaScriptHost() as host:
        adapter = connected(host)
        assert [r.name for r in adapter.memory_regions()] == ["iwram", "wram"]
        host.hold()
        assert adapter.read_memory("wram", 0, 32) == host.game.ram
        assert adapter.read_memory("wram", 3, 5) == host.game.ram[3:8]
        assert adapter.read_memory("wram", 0, 0) == b""
        assert adapter.read_memory("iwram", 100, 3) == bytes(3)
        assert "readRange" in host.calls
        answer = adapter._call(
            "read",
            reads=[
                {"domain": "wram", "addr": 0, "len": 4},
                {"domain": "wram", "addr": 4, "len": 4},
            ],
        )
        assert [base64.b64decode(r) for r in answer["reads"]] == [
            host.game.ram[0:4],
            host.game.ram[4:8],
        ]
        for reads, message in [
            ([{"domain": "vram", "addr": 0, "len": 1}], "Unknown memory domain 'vram'"),
            ([{"domain": "wram", "addr": -1, "len": 1}], "whole-number address"),
            ([{"domain": "wram", "addr": 0, "len": 1.5}], "whole-number address"),
            ([{"domain": "wram", "addr": 30, "len": 4}], "outside 'wram' \\(size 32\\)"),
            ("wram", "must be a list"),
            ([1], "must be a list"),
        ]:
            with pytest.raises(EmulatorError, match=message):
                adapter._call("read", reads=reads)
        host.release()
        adapter.disconnect()
        assert host.error is None


def test_input_is_a_key_mask_applied_every_frame_and_step_counts_frames() -> None:
    with MgbaScriptHost() as host:
        adapter = connected(host)
        host.hold()
        a_bit, up_bit = 1 << GBA_KEYS[b"A"], 1 << GBA_KEYS[b"UP"]
        adapter.set_input({Button("a", "A"), Button("up", "UP")})
        assert host.mask == a_bit | up_bit and host.pressed == ("a", "up")
        with pytest.raises(ConfigurationError):
            adapter.set_input({Button("turbo", "Turbo")})
        with pytest.raises(EmulatorError, match="has no button 'Turbo'"):
            adapter._call("input", buttons={"Turbo": True})
        assert host.mask == a_bit | up_bit  # a refused input changes nothing
        # a step of 5 frames is answered after exactly 5 frames, keys held on each
        reference = ToyRacer(0)
        reference.restore(host.game.snapshot())
        link = adapter._link
        assert link is not None
        channel = link._channel
        channel.send_line(b'{"id": 1, "cmd": "step", "frames": 5, "buttons": {"UP": true}}\n')
        frame0 = host.game.state.frame
        for _ in range(4):
            host.tick()
            with pytest.raises(EmulatorError, match="did not answer"):
                channel.receive_line(0.05)
        host.tick()
        answer = json.loads(channel.receive_line(5.0))
        assert answer == {"id": 1, "ok": True, "frame": frame0 + 5}
        assert host.masks[-5:] == [up_bit] * 5
        for _ in range(5):
            reference.step(("up",))
        assert host.game.snapshot() == reference.snapshot()
        # another request while a step runs is refused, and the step still answers
        channel.send_line(b'{"id": 2, "cmd": "step", "frames": 2}\n')
        channel.send_line(b'{"id": 3, "cmd": "ping"}\n')
        host.tick()
        refused = json.loads(channel.receive_line(5.0))
        assert refused == {
            "id": 3,
            "ok": False,
            "error": "A step is still running; wait for its answer.",
        }
        host.tick()
        assert json.loads(channel.receive_line(5.0))["id"] == 2
        for bad in ({"frames": 0}, {"frames": 3601}, {"frames": 1.5}, {"frames": "3"}):
            channel.send_line(json.dumps({"id": 9, "cmd": "step", **bad}).encode() + b"\n")
            host.tick()
            assert "whole number from 1 to 3600" in json.loads(channel.receive_line(5.0))["error"]
        host.release()
        adapter.disconnect()
        assert host.error is None


def test_bundled_step_reads_and_screenshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with MgbaScriptHost() as host:
        adapter = connected(host)
        frame = adapter.capture_frame()
        expected = host.game.render()
        assert (frame.width, frame.height, frame.channels) == (expected.width, expected.height, 3)
        assert [Path(p.decode()) for p in host.screenshots] == [Path(adapter.image_path)]  # T-825
        assert not Path(adapter.image_path).exists() and list(tmp_path.iterdir()) == []
        answer = adapter._call(
            "step",
            frames=2,
            buttons={"A": True},
            reads=[{"domain": "wram", "addr": 0, "len": 8}],
            screenshot=True,
        )
        assert base64.b64decode(answer["image"])[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(base64.b64decode(answer["reads"][0])) == 8
        target = tmp_path / "named.png"
        answer = adapter._call("step", frames=1, screenshot=True, image_file=str(target))
        assert answer["image_file"] == str(target) and "image" not in answer
        assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        with pytest.raises(EmulatorError, match="image_file"):
            adapter._call("step", frames=1, screenshot=True, image_file="notes.txt")
        with pytest.raises(EmulatorError, match="image_file"):
            adapter._call("screenshot", image_file=7)
        host.write_screenshots = False
        with pytest.raises(EmulatorError, match="did not write a screenshot"):
            adapter.capture_frame()
        with pytest.raises(EmulatorError, match="did not write a screenshot"):
            adapter._call("step", frames=1, screenshot=True)
        assert adapter.is_connected
        adapter.disconnect()
        assert host.error is None


def test_save_states_are_buffers_by_slot_name() -> None:
    with MgbaScriptHost() as host:
        adapter = connected(host)
        host.hold()
        adapter.set_input({Button("up", "UP")})
        adapter.save_state("start")
        start = host.game.snapshot()
        host.release()
        wait_until(lambda: host.game.snapshot() != start)
        host.hold()
        adapter.load_state("start")
        assert host.game.snapshot() == start
        assert host.calls.count("saveStateBuffer") == 1 and "loadStateBuffer" in host.calls
        with pytest.raises(ConfigurationError, match="No save state in slot 'never'"):
            adapter.load_state("never")
        with pytest.raises(EmulatorError, match="letters, digits and underscores"):
            adapter._call("save", slot="../x")
        host.release()
        adapter.disconnect()
        assert host.error is None


def test_script_survives_bad_requests_and_a_vanished_app() -> None:
    with MgbaScriptHost() as host:
        adapter = connected(host)
        link = adapter._link
        assert link is not None
        channel = link._channel
        channel.send_line(b"this is not json\n")
        assert json.loads(channel.receive_line(5.0)) == {
            "ok": False,
            "error": "That was not a JSON object.",
        }
        channel.send_line(b'{"id": 7, "cmd": "dance"}\n')
        assert json.loads(channel.receive_line(5.0)) == {
            "id": 7,
            "ok": False,
            "error": "Unknown command 'dance'.",
        }
        channel.send_line(b'{"id": 8, "cmd": "mode", "free_run": "yes"}\n')
        answer = json.loads(channel.receive_line(5.0))
        assert answer["ok"] is False and "true or false" in answer["error"]
        adapter.set_free_run(True)  # accepted, and nothing changes: free run is all there is
        adapter.ping()
        channel.close()
        wait_until(lambda: any(b"disconnected" in line for line in host.log))
        assert host.mask == 0  # the pad is released when the app goes
        with pytest.raises(EmulatorDisconnectedError):
            adapter.ping()
        adapter.connect(timeout_s=5.0)
        adapter.disconnect()
        assert host.error is None


def test_second_connection_is_turned_away_while_one_is_served() -> None:
    import socket

    with MgbaScriptHost() as host:
        adapter = connected(host)
        other = socket.create_connection(("127.0.0.1", host.port), timeout=5.0)
        try:
            wait_until(lambda: other.recv(1) == b"")  # closed by the script
        finally:
            other.close()
        adapter.ping()
        adapter.disconnect()
        assert host.error is None


# ---------------------------------------------------------------- codecs

json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-(2**53), 2**53) | st.text(max_size=20),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=12,
)


@settings(max_examples=100, deadline=None)
@given(json_values)
def test_lua_json_agrees_with_python(value: Any) -> None:
    internals = _INTERNALS
    text = json.dumps(value)
    decoded = internals.json.decode(text.encode("utf-8"))
    encoded = internals.json.encode(decoded)
    assert json.loads(encoded) == _lossy(value)


@settings(max_examples=100, deadline=None)
@given(st.binary(max_size=200))
def test_lua_base64_agrees_with_python(data: bytes) -> None:
    internals = _INTERNALS
    assert internals.base64.encode(data) == base64.b64encode(data)
    assert internals.base64.decode(base64.b64encode(data)) == data


_INTERNALS = load_internals()


def _lossy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _lossy(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_lossy(v) for v in value if v is not None]
    return value


def test_protocol_errors_are_the_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    assert issubclass(ProtocolError, EmulatorError)
