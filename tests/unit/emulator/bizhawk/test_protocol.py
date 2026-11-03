"""T-111: BizHawk bridge framing."""

from __future__ import annotations

import contextlib
import json
import socket
import threading

import pytest

from retro_ai_lab.core import EmulatorDisconnectedError, EmulatorError, ProtocolError
from retro_ai_lab.emulator.bizhawk.protocol import (
    MAX_LINE,
    LineChannel,
    Link,
    decode_response,
    encode_request,
    open_link,
)


def test_request_and_response_round_trip() -> None:
    line = encode_request(3, "read", reads=[{"domain": "RAM", "addr": 0, "len": 4}])
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert json.loads(line) == {
        "id": 3,
        "cmd": "read",
        "reads": [{"domain": "RAM", "addr": 0, "len": 4}],
    }
    answer = decode_response(b'{"id":3,"ok":true,"frame":9,"reads":["AAEC"]}', 3)
    assert answer["frame"] == 9 and answer["reads"] == ["AAEC"]
    with pytest.raises(ProtocolError, match="can't be called"):
        encode_request(1, "x", id=2)


@pytest.mark.parametrize(
    ("line", "error", "match"),
    [
        (b"not json", ProtocolError, "not JSON"),
        (b"[1,2]", ProtocolError, "not an object"),
        (b'{"id":2,"ok":true}', ProtocolError, "answered request 2 instead of 1"),
        (b'{"id":1}', ProtocolError, "no 'ok'"),
        (b'{"id":1,"ok":false,"error":"Unknown domain."}', EmulatorError, "Unknown domain."),
        (b'{"id":1,"ok":false}', EmulatorError, "refused"),
    ],
)
def test_bad_responses_are_reported(line: bytes, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        decode_response(line, 1)


def test_channel_reassembles_lines_split_across_packets() -> None:
    ours, theirs = socket.socketpair()
    channel = LineChannel(ours)
    theirs.sendall(b'{"id":1,"ok":tr')
    theirs.sendall(b'ue}\n{"id":2,"ok":true}\n{"id"')
    assert channel.receive_line(1.0) == b'{"id":1,"ok":true}'
    assert channel.receive_line(1.0) == b'{"id":2,"ok":true}'
    theirs.sendall(b':3,"ok":true}\n')
    assert channel.receive_line(1.0) == b'{"id":3,"ok":true}'
    with pytest.raises(EmulatorError, match=r"did not answer within 0\.05 seconds"):
        channel.receive_line(0.05)
    theirs.close()
    with pytest.raises(EmulatorDisconnectedError):
        channel.receive_line(1.0)
    assert not channel.is_open
    channel.close()  # idempotent
    ours.close()


def test_channel_refuses_absurdly_long_lines() -> None:
    ours, theirs = socket.socketpair()
    channel = LineChannel(ours)

    def flood() -> None:
        with contextlib.suppress(OSError):
            theirs.sendall(b"x" * (MAX_LINE + 2))

    thread = threading.Thread(target=flood, daemon=True)
    thread.start()
    with pytest.raises(ProtocolError, match="far too long"):
        channel.receive_line(10.0)
    assert not channel.is_open
    theirs.close()
    thread.join(timeout=5.0)


def test_link_numbers_requests_and_closes_on_protocol_errors() -> None:
    ours, theirs = socket.socketpair()
    link = Link(LineChannel(ours), timeout_s=1.0)
    theirs.sendall(b'{"id":1,"ok":true}\n{"id":2,"ok":true,"x":1}\n')
    assert link.call("ping") == {"id": 1, "ok": True}
    assert link.call("ping")["x"] == 1
    assert theirs.recv(1024) == b'{"id":1,"cmd":"ping"}\n{"id":2,"cmd":"ping"}\n'
    theirs.sendall(b'{"id":99,"ok":true}\n')
    with pytest.raises(ProtocolError):
        link.call("ping")
    assert not link.is_open
    with pytest.raises(EmulatorDisconnectedError):
        link.call("ping")
    theirs.close()


def test_open_link_explains_a_missing_bridge() -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens here now
    with pytest.raises(EmulatorError, match="Couldn't find BizHawk on port") as info:
        open_link("127.0.0.1", port, timeout_s=1.0)
    assert "Lua Console" in str(info.value)
