"""T-111: the shipped Lua bridge script, run under lupa against stubs."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import Button, ConfigurationError, EmulatorDisconnectedError, EmulatorError
from retro_ai_lab.emulator.bizhawk import PROTOCOL_VERSION, BizHawkAdapter
from retro_ai_lab.emulator.fake.toy_racer import ToyRacer
from tests.unit.emulator.bizhawk.lua_host import (
    BIZHAWK_VERSION,
    PAD_BUTTONS,
    ROM_HASH,
    ROM_NAME,
    SYSTEM,
    LuaScriptHost,
    load_internals,
)


def _registry() -> Any:
    from retro_ai_lab.emulator.registry import default_registry

    return default_registry()


def wait_until(check: Any, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def test_handshake_hold_mode_and_reconnect() -> None:
    with LuaScriptHost() as host:
        assert host.log and b"waiting for the app on port" in host.log[0]
        assert host.exit_hooks == 1
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        hello = adapter.hello
        assert hello["script"] == PROTOCOL_VERSION
        assert hello["bizhawk"] == BIZHAWK_VERSION.decode()
        assert hello["system"] == SYSTEM.decode()
        assert hello["rom"] == ROM_NAME.decode()
        assert hello["hash"] == ROM_HASH.decode()
        assert hello["domains"] == [
            {"name": "RAM", "size": 32},
            {"name": "WRAM", "size": 65536},
            {"name": "System Bus", "size": 65536},
            {"name": "PRG ROM", "size": 32768},
            {"name": "CHR VROM", "size": 8192},
        ]
        assert hello["buttons"] == sorted(["Up", "Down", "Left", "Right", "A", "B", "Start"])
        assert hello["commands"] == [
            "bye",
            "hello",
            "input",
            "load",
            "mode",
            "ping",
            "read",
            "save",
            "screenshot",
            "speed",
            "step",
        ]
        assert isinstance(hello["frame"], int)
        assert [b.name for b in adapter.buttons()] == [
            "a",
            "b",
            "down",
            "left",
            "right",
            "start",
            "up",
        ]
        info = adapter.game_info()
        assert (info.system, info.title, info.rom_hash) == (
            "NES",
            "Toy Racer (stub)",
            "ABCDEF0123456789",
        )
        # hold mode: paused on hello, and no frames pass while we talk
        assert host.paused and "pause" in host.calls
        frames_before = host.frames
        for _ in range(3):
            adapter.ping()
        assert host.frames == frames_before
        adapter.disconnect()
        wait_until(lambda: not host.paused)
        assert host.calls[-2:] == ["unpause", "displaymessages"]  # hands the emulator back
        # the emulator runs again, and the script takes a new connection
        wait_until(lambda: host.frames > frames_before)
        adapter.connect(timeout_s=5.0)
        assert adapter.is_connected and host.paused
        adapter.disconnect()
        assert host.error is None


def test_memory_reads_through_the_real_script() -> None:
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        assert adapter.capabilities().memory_read
        # Only live memory is offered (T-827): the bus and the ROMs the script lists are not.
        assert [r.name for r in adapter.memory_regions()] == ["ram", "wram"]
        assert [r.name for r in adapter.description.regions] == [
            "ram",
            "wram",
            "system_bus",
            "prg_rom",
            "chr_vrom",
        ]
        assert adapter.read_memory("ram", 0, 32) == host.game.ram
        assert adapter.read_memory("ram", 3, 5) == host.game.ram[3:8]
        assert adapter.read_memory("ram", 0, 0) == b""
        assert adapter.read_memory("wram", 0, 65536) == bytes(65536)  # chunked unpack
        with pytest.raises(ConfigurationError, match="Unknown memory region 'system_bus'"):
            adapter.read_memory("system_bus", 0, 16)
        answer = adapter._call(
            "read",
            reads=[
                {"domain": "RAM", "addr": 0, "len": 4},
                {"domain": "RAM", "addr": 4, "len": 4},
            ],
        )
        assert [base64.b64decode(r) for r in answer["reads"]] == [
            host.game.ram[0:4],
            host.game.ram[4:8],
        ]
        for reads, message in [
            ([{"domain": "VRAM", "addr": 0, "len": 1}], "Unknown memory domain 'VRAM'"),
            ([{"domain": "RAM", "addr": -1, "len": 1}], "whole-number address"),
            ([{"domain": "RAM", "addr": 0, "len": 1.5}], "whole-number address"),
            ([{"domain": "RAM", "addr": 30, "len": 4}], "outside 'RAM' \\(size 32\\)"),
            ("RAM", "must be a list"),
            ([1], "must be a list"),
        ]:
            with pytest.raises(EmulatorError, match=message):
                adapter._call("read", reads=reads)
        adapter.ping()
        adapter.disconnect()
        assert host.error is None


def test_input_is_held_and_applied_every_frame() -> None:
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        assert adapter.capabilities().input
        adapter.set_input({Button("up", "Up")})
        assert host.game.state.speed == 0  # hold mode: nothing moves yet
        adapter._call("mode", free_run=True)
        wait_until(lambda: host.game.state.speed >= 3)
        assert host.last_pad is not None
        table, controller = host.last_pad
        assert controller == 1
        assert table == {**dict.fromkeys(PAD_BUTTONS, False), b"Up": True}
        adapter.set_input(())
        wait_until(lambda: host.last_pad is not None and not any(host.last_pad[0].values()))
        speed = host.game.state.speed
        frames = host.frames
        wait_until(lambda: host.frames > frames + 10)
        assert host.game.state.speed <= speed  # no more acceleration
        # the script refuses bad input and stays up
        for bad, message in [
            ("yes", "must be a table"),
            ({"Turbo": True}, "has no button 'Turbo'"),
            ({"Up": 1}, "must be true or false"),
        ]:
            with pytest.raises(EmulatorError, match=message):
                adapter._call("input", buttons=bad)
        adapter.ping()
        adapter.set_input({Button("up", "Up")})
        adapter.disconnect()
        # once the app is gone the script stops driving the pad at all, so the
        # player's own controller is free again (joypad.set only lasts a frame)
        frames = host.frames
        wait_until(lambda: host.frames > frames + 3)
        sets = host.calls.count("joypad_set")
        frames = host.frames
        wait_until(lambda: host.frames > frames + 5)
        assert host.calls.count("joypad_set") == sets
        assert host.pressed == ()
        assert host.error is None


def test_free_run_mode_lets_frames_pass() -> None:
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        assert host.paused
        adapter._call("mode", free_run=True)
        wait_until(lambda: not host.paused)
        start = host.frames
        wait_until(lambda: host.frames > start + 5)
        adapter._call("mode", free_run=False)
        wait_until(lambda: host.paused)
        adapter.disconnect()
        assert host.error is None


def test_script_survives_bad_requests_and_a_vanished_app() -> None:
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
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
        adapter.ping()  # still in step
        # the app vanishes without a bye: the script unpauses and waits again
        channel.close()
        wait_until(lambda: not host.paused)
        assert any(b"disconnected" in line for line in host.log)
        with pytest.raises(EmulatorDisconnectedError):
            adapter.ping()
        adapter.connect(timeout_s=5.0)
        adapter.disconnect()
        assert host.error is None


def test_screenshot_through_the_real_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        assert adapter.capabilities().screen_capture
        expected = host.game.render()
        frame = adapter.capture_frame()
        assert (frame.width, frame.height, frame.channels) == (expected.width, expected.height, 3)
        assert frame.pixels == bytes(v for v in expected.pixels for _ in range(3))
        assert frame.number == host.game.state.frame
        # On this PC the adapter names the file (T-825) and reads it itself.
        assert adapter.image_transfer == "file"
        assert [Path(p.decode()) for p in host.screenshots] == [Path(adapter.image_path)]
        assert not Path(adapter.image_path).exists()  # the adapter tidies up after itself
        host.write_screenshots = False
        with pytest.raises(EmulatorError, match="did not write a screenshot"):
            adapter.capture_frame()
        assert adapter.is_connected
        adapter.disconnect()
        assert host.error is None


def test_screenshot_over_the_socket_through_the_real_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The base64 route stays for an emulator on another PC."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0, image_transfer="socket")
        adapter.connect(timeout_s=5.0)
        frame = adapter.capture_frame()
        assert frame.number == host.game.state.frame
        # the script joins with "/", which Windows accepts; compare as paths
        assert [Path(p.decode()) for p in host.screenshots] == [tmp_path / "retro_ai_lab_frame.png"]
        assert list(tmp_path.iterdir()) == []  # the script tidies up after itself
        answer = adapter._call("step", frames=1, screenshot=True)
        assert base64.b64decode(answer["image"])[:8] == b"\x89PNG\r\n\x1a\n"
        assert "image_file" not in answer
        # A named file comes back as its path, with no base64 copy.
        target = tmp_path / "named.png"
        answer = adapter._call("step", frames=1, screenshot=True, image_file=str(target))
        assert answer["image_file"] == str(target) and "image" not in answer
        assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        for bad in ("notes.txt", "", 7):
            with pytest.raises(EmulatorError, match="image_file"):
                adapter._call("screenshot", image_file=bad)
        assert adapter.is_connected
        adapter.disconnect()
        assert host.error is None


def test_hello_leaves_out_a_domain_the_emulator_cannot_read() -> None:
    """T-832: the user's EmuHawk threw a .NET exception on one domain's first read."""
    with LuaScriptHost() as host:
        host.unreadable.add(b"WRAM")
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        assert [d["name"] for d in adapter.hello["domains"]] == [
            "RAM",
            "System Bus",
            "PRG ROM",
            "CHR VROM",
        ]
        assert [r.name for r in adapter.memory_regions()] == ["ram"]
        assert any(b"not offering memory domain 'WRAM'" in line for line in host.log)
        assert adapter.read_memory("ram", 0, 32) == host.game.ram
        adapter.disconnect()
        assert host.error is None


def test_a_read_that_fails_names_the_region() -> None:
    """T-832: a .NET exception on a read reaches the app with the region in it."""
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        host.unreadable.add(b"RAM")  # breaks after the handshake probed it
        with pytest.raises(EmulatorError, match=r"could not read 8 bytes at 4 of 'RAM': .*peek"):
            adapter.read_memory("ram", 4, 8)
        assert adapter.is_connected  # a refused read is not a lost link
        # T-833: the region is left out from now on, so the next plan skips it
        assert adapter.unreadable_regions == {"ram"}
        assert [r.name for r in adapter.memory_regions()] == ["wram"]
        assert adapter.read_memory("wram", 0, 4) == bytes(4)
        adapter.disconnect()
        assert host.error is None


def test_a_region_refused_mid_game_is_dropped_and_the_snapshots_go_on() -> None:
    """T-833: the user's EmuHawk refused 'CPU registers' on every step after hello."""
    from retro_ai_lab.core.events import EventBus
    from retro_ai_lab.services import ConnectionService
    from retro_ai_lab.services.observer import ObservationService

    with LuaScriptHost() as host:
        bus = EventBus()
        connection = ConnectionService(_registry(), bus)
        observation = ObservationService(connection, bus, ring_capacity=10)
        connection.connect("bizhawk", port=host.port, call_timeout_s=5.0, timeout_s=5.0)
        assert observation.snapshot_reads() == (("ram", 0, 32), ("wram", 0, 65536))
        host.unreadable.add(b"WRAM")
        reads = observation.snapshot_reads()
        with pytest.raises(EmulatorError, match="could not read 65536 bytes at 0 of 'WRAM'"):
            connection.call(lambda a: a.step(1, reads=reads))
        # the next step plans without it and the snapshot lands
        reads = observation.snapshot_reads()
        assert reads == (("ram", 0, 32),)
        result = connection.call(lambda a: a.step(1, reads=reads))
        event = observation.advance(1, snapshot=result.reads)
        assert event is not None and event.snapshot.get("ram") == host.game.ram
        assert observation.observe() is not None  # the unbundled path replans too
        connection.disconnect()
        assert host.error is None


def test_screenshot_messages_are_off_while_connected() -> None:
    """T-833: EmuHawk's '<path> saved.' message on every picture is silenced."""
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        assert host.messages_shown
        adapter.connect(timeout_s=5.0)
        assert not host.messages_shown
        adapter.disconnect()
        wait_until(lambda: host.messages_shown)
        assert host.error is None


def test_hold_mode_step_unpauses_runs_and_pauses_again() -> None:
    """T-831: a paused EmuHawk never wakes emu.frameadvance(), so a step unpauses first.

    The host runs EmuHawk's loop order (latch the pad, resume yielders, then
    a frame only when unpaused), which is how the user's first real run
    failed: with the old script every step timed out after 5 s.
    """
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        assert host.paused
        calls_before = len(host.calls)
        before = host.frames  # the game ran free until hello paused it
        reference = ToyRacer(0)
        reference.restore(host.game.snapshot())
        answer = adapter._call("step", frames=1, buttons={"Up": True})
        assert answer["frame"] == host.game.state.frame
        assert host.frames == before + 1 and host.paused
        reference.step(("up",))  # the pad reached the very first frame
        assert host.game.snapshot() == reference.snapshot()
        pad_calls = [c for c in host.calls[calls_before:] if c in ("pause", "unpause")]
        assert pad_calls == ["unpause", "pause"]
        assert host.pressed == ()  # nothing left armed for a frame that must not run
        time.sleep(0.2)
        assert host.frames == before + 1
        adapter.disconnect()
        assert host.error is None


def test_step_advances_exactly_and_keeps_the_hold() -> None:
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        assert adapter.capabilities().frame_advance
        assert host.paused
        before = host.frames
        reference = ToyRacer(0)
        reference.restore(host.game.snapshot())
        adapter.set_input({Button("up", "Up")})
        adapter.advance_frames(5)
        assert host.frames == before + 5
        for _ in range(5):  # the held button went into every one of those frames
            reference.step(("up",))
        assert host.game.snapshot() == reference.snapshot()
        assert host.paused
        # nothing moves while the app thinks
        time.sleep(0.2)
        assert host.frames == before + 5
        # a bundled step: buttons, reads and a screenshot in one round trip
        host.write_screenshots = True
        answer = adapter._call(
            "step",
            frames=2,
            buttons={"Up": False, "A": True},
            reads=[{"domain": "RAM", "addr": 0, "len": 8}],
            screenshot=True,
        )
        assert host.frames == before + 7
        assert host.pressed == () and host.last_pad is not None
        assert host.last_pad[0][b"A"] is True and host.last_pad[0][b"Up"] is False
        assert base64.b64decode(answer["reads"][0]) == bytes(host.game.ram[:8])
        assert base64.b64decode(answer["image"])[:8] == b"\x89PNG\r\n\x1a\n"
        assert answer["frame"] == host.game.state.frame
        for bad in ({"frames": 0}, {"frames": 3601}, {"frames": 1.5}, {"frames": "3"}):
            with pytest.raises(EmulatorError, match="whole number from 1 to 3600"):
                adapter._call("step", **bad)
        with pytest.raises(EmulatorError, match="'reads' must be a list"):
            adapter._call("step", frames=1, reads="RAM")
        with pytest.raises(EmulatorError, match="'screenshot' must be true or false"):
            adapter._call("step", frames=1, screenshot=1)
        with pytest.raises(EmulatorError, match="has no button 'Turbo'"):
            adapter._call("step", frames=1, buttons={"Turbo": True})
        assert host.frames == before + 7  # refused steps move nothing
        adapter.disconnect()
        assert host.error is None


def test_speed_and_save_states_through_the_real_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        caps = adapter.capabilities()
        assert caps.speed_control and caps.save_states
        adapter.set_speed(1.5)
        assert host.speed == 150
        adapter.set_speed(0.004)
        assert host.speed == 1
        adapter._call("speed", percent=249.6)
        assert host.speed == 250
        for bad in ("fast", 0, -1, True):
            with pytest.raises(EmulatorError, match="number above zero"):
                adapter._call("speed", percent=bad)
        adapter.set_input({Button("up", "Up")})
        adapter.advance_frames(3)
        adapter.save_state("lap_1")
        snapshot = host.game.snapshot()
        assert [Path(p.decode()) for p in host.states] == [tmp_path / "retro_ai_lab_lap_1.State"]
        adapter.advance_frames(10)
        assert host.game.snapshot() != snapshot
        adapter.load_state("lap_1")
        assert host.game.snapshot() == snapshot
        with pytest.raises(ConfigurationError, match=r"No save state in slot 'lap_2'"):
            adapter.load_state("lap_2")
        for bad in ("", "no/slashes", "..", 7, None):
            with pytest.raises(EmulatorError, match="letters, digits and underscores"):
                adapter._call("save", slot=bad)
            with pytest.raises(EmulatorError, match="letters, digits and underscores"):
                adapter._call("load", slot=bad)
        assert host.paused and adapter.is_connected
        adapter.disconnect()
        assert host.error is None
        # The app changed the speed, so the script puts 100% back when it leaves (T-836).
        wait_until(lambda: host.speed == 100)


def test_a_speed_the_app_never_touched_is_left_alone() -> None:
    with LuaScriptHost() as host:
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        adapter.disconnect()
        wait_until(lambda: not host.paused)
        assert host.speed is None and "speedmode" not in host.calls


json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False, width=32)
    | st.text(),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(), inner, max_size=4),
    max_leaves=12,
)


@settings(max_examples=150, deadline=None)
@given(json_values)
def test_lua_json_agrees_with_python(value: Any) -> None:
    internals = _INTERNALS
    text = json.dumps(value)
    decoded = internals.json.decode(text.encode("utf-8"))
    encoded = internals.json.encode(decoded)
    assert json.loads(encoded) == _lossy(value)


@pytest.mark.parametrize(
    "value",
    [
        "\x8d",  # a C1 control: U+008D is UTF-8 C2 8D
        "Í",  # U+00CD is C3 8D: byte 0x8D must not be escaped on its own (Windows %c did)
        '\x00\x1f\x7f"\\',
        "é€𝄞",
    ],
)
def test_lua_json_keeps_utf8_bytes_whole(value: str) -> None:
    internals = _INTERNALS
    encoded = internals.json.encode(internals.json.decode(json.dumps(value).encode("utf-8")))
    assert json.loads(encoded) == value


@settings(max_examples=100, deadline=None)
@given(st.binary(max_size=200))
def test_lua_base64_agrees_with_python(data: bytes) -> None:
    internals = _INTERNALS
    assert internals.base64.encode(data) == base64.b64encode(data)
    assert internals.base64.decode(base64.b64encode(data)) == data


_INTERNALS = load_internals()


def _lossy(value: Any) -> Any:
    """What survives a trip through Lua: a null inside a container vanishes.

    Lua tables can't hold nil, so the script's JSON drops them. The protocol
    only ever puts null in a top-level response field (``image``), never
    inside an array.
    """
    if isinstance(value, dict):
        return {k: _lossy(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_lossy(v) for v in value if v is not None]
    return value


def test_bundled_step_through_the_real_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-822: the adapter's step() is one request to the script, reads and screenshot included."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with LuaScriptHost() as host:
        host.write_screenshots = True
        adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
        adapter.connect(timeout_s=5.0)
        ram = next(r for r in adapter.memory_regions() if r.name == "ram")
        before = host.frames
        result = adapter.step(
            2, pressed={Button("up", "Up")}, reads=[("ram", 0, ram.size)], screenshot=True
        )
        assert host.frames == before + 2
        assert result.reads == (host.game.ram,)
        assert result.frame is not None and result.frame.number == host.frames
        adapter.disconnect()
