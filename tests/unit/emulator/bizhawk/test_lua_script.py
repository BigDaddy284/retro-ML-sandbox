"""T-111: the shipped Lua bridge script, run under lupa against stubs."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import EmulatorDisconnectedError
from retro_ai_lab.emulator.bizhawk import PROTOCOL_VERSION, BizHawkAdapter
from tests.unit.emulator.bizhawk.lua_host import (
    BIZHAWK_VERSION,
    ROM_HASH,
    ROM_NAME,
    SYSTEM,
    LuaScriptHost,
    load_internals,
)


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
            {"name": "System Bus", "size": 65536},
        ]
        assert hello["buttons"] == sorted(["Up", "Down", "Left", "Right", "A", "B", "Start"])
        assert isinstance(hello["frame"], int)
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
        assert host.calls[-1] == "unpause"
        # the emulator runs again, and the script takes a new connection
        wait_until(lambda: host.frames > frames_before)
        adapter.connect(timeout_s=5.0)
        assert adapter.is_connected and host.paused
        adapter.disconnect()
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
