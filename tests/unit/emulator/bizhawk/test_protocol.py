"""T-111: BizHawk bridge framing."""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading

import pytest

from retro_ai_lab.core import EmulatorDisconnectedError, EmulatorError, ProtocolError
from retro_ai_lab.emulator.bizhawk.protocol import (
    MAX_LINE,
    CallStats,
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


def test_a_late_answer_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """T-854: the answer to a call that timed out does not break the next call."""
    ours, theirs = socket.socketpair()
    link = Link(LineChannel(ours), timeout_s=0.2)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.emulator.bizhawk.protocol")
    with pytest.raises(EmulatorError, match="did not answer"):
        link.call("step", frames=300)  # nothing comes back in time
    assert link.is_open
    theirs.sendall(b'{"id":1,"ok":true}\n{"id":2,"ok":true,"x":1}\n')  # both arrive now
    assert link.call("ping")["x"] == 1 and link.is_open
    assert "Skipped the emulator's late answer to request 1." in caplog.text
    # A stream of late answers does not wait forever: the deadline covers the skipping.
    theirs.sendall(b'{"id":1,"ok":true}\n' * 3)
    with pytest.raises(EmulatorError, match="did not answer"):
        link.call("ping")
    assert link.is_open
    # An answer to a request not made yet is still a protocol error that closes the link.
    theirs.sendall(b'{"id":50,"ok":true}\n')
    with pytest.raises(ProtocolError):
        link.call("ping")
    assert not link.is_open
    theirs.close()


def test_open_link_explains_a_missing_bridge() -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens here now
    with pytest.raises(EmulatorError, match="Couldn't find BizHawk on port") as info:
        open_link("127.0.0.1", port, timeout_s=1.0)
    assert "Lua Console" in str(info.value)


def test_peer_closed_probes_without_sending() -> None:
    ours, theirs = socket.socketpair()
    channel = LineChannel(ours)
    assert not channel.peer_closed()  # open and idle
    theirs.sendall(b'{"id":1,"ok":true}\n')
    assert not channel.peer_closed()  # unread data is not a closed peer
    assert channel.receive_line(1.0) == b'{"id":1,"ok":true}'
    theirs.close()
    assert channel.peer_closed()
    channel.close()
    assert channel.peer_closed()  # our own closed channel counts too, quietly


# ------------------------------------------------------------- T-819 timing


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _timed_link(report_every_s: float = 5.0) -> tuple[Link, socket.socket, _Clock]:
    ours, theirs = socket.socketpair()
    clock = _Clock()
    link = Link(LineChannel(ours), timeout_s=1.0, report_every_s=report_every_s, clock=clock)
    return link, theirs, clock


def _answer(theirs: socket.socket, request_id: int) -> None:
    theirs.sendall(f'{{"id":{request_id},"ok":true}}\n'.encode())


def test_link_times_every_call_per_command() -> None:
    link, theirs, clock = _timed_link()
    assert link.summary() == "no calls" and link.stats == {}

    # The clock moves while the answer is awaited: patch receive to advance it.
    original = link._channel.receive_line
    steps = [0.03, 0.05, 0.02]

    def slow_receive(timeout_s: float) -> bytes:
        clock.now += steps.pop(0)
        return original(timeout_s)

    link._channel.receive_line = slow_receive  # type: ignore[method-assign]
    _answer(theirs, 1)
    link.call("screenshot")
    _answer(theirs, 2)
    link.call("screenshot")
    _answer(theirs, 3)
    link.call("step", frames=1)
    stats = link.stats
    assert stats["screenshot"].count == 2
    assert stats["screenshot"].average_s == pytest.approx(0.04)
    assert stats["screenshot"].max_s == pytest.approx(0.05)
    assert stats["step"].count == 1 and stats["step"].max_s == pytest.approx(0.02)
    assert link.summary() == (
        "screenshot \u00d72 avg 40 ms max 50 ms; step \u00d71 avg 20 ms max 20 ms"
    )
    theirs.close()


def test_a_failed_call_is_still_timed() -> None:
    link, theirs, clock = _timed_link()
    original = link._channel.receive_line

    def slow_receive(timeout_s: float) -> bytes:
        clock.now += 0.25
        return original(timeout_s)

    link._channel.receive_line = slow_receive  # type: ignore[method-assign]
    theirs.sendall(b'{"id":99,"ok":true}\n')  # the wrong id: a protocol error
    with pytest.raises(ProtocolError):
        link.call("ping")
    assert link.stats["ping"].count == 1
    assert link.stats["ping"].max_s == pytest.approx(0.25)
    theirs.close()


def test_link_reports_once_per_window_and_starts_afresh(
    caplog: pytest.LogCaptureFixture,
) -> None:
    link, theirs, clock = _timed_link(report_every_s=5.0)
    with caplog.at_level("INFO", logger="retro_ai_lab.emulator.bizhawk.protocol"):
        _answer(theirs, 1)
        link.call("screenshot")
        assert not caplog.records  # the window is still open
        clock.now += 5.0
        _answer(theirs, 2)
        link.call("step", frames=1)  # the window is over: this call reports
        assert [r.getMessage() for r in caplog.records] == [
            "Emulator link, last 5 s: screenshot \u00d71 avg 0 ms max 0 ms; "
            "step \u00d71 avg 0 ms max 0 ms"
        ]
        assert link.stats == {} and link.summary() == "no calls"
        _answer(theirs, 3)
        link.call("ping")
        assert len(caplog.records) == 1  # a new window has begun
        assert list(link.stats) == ["ping"]
        # T-830: the totals keep every call since the link opened.
        assert {name: stats.count for name, stats in link.totals.items()} == {
            "screenshot": 1,
            "step": 1,
            "ping": 1,
        }
        assert link.summary(lifetime=True) == (
            "screenshot \u00d71 avg 0 ms max 0 ms; step \u00d71 avg 0 ms max 0 ms; "
            "ping \u00d71 avg 0 ms max 0 ms"
        )
    theirs.close()


def test_an_idle_link_never_reports(caplog: pytest.LogCaptureFixture) -> None:
    link, theirs, clock = _timed_link(report_every_s=1.0)
    with caplog.at_level("INFO", logger="retro_ai_lab.emulator.bizhawk.protocol"):
        clock.now += 60.0
        assert link.summary() == "no calls"
    assert not caplog.records
    theirs.close()


def test_call_stats_describe_rounds_to_milliseconds() -> None:
    stats = CallStats()
    stats.add(0.0334)
    stats.add(0.0616)
    assert stats.describe("screenshot") == "screenshot \u00d72 avg 48 ms max 62 ms"


def test_link_notes_timings_the_emulator_reports() -> None:
    """T-890: a timing from the script is kept under its own name, in window and lifetime."""
    link, theirs, _clock = _timed_link()
    _answer(theirs, 1)
    link.call("step", frames=1)
    link.note("step:latch", 0.016)
    link.note("step:latch", 0.018)
    assert link.stats["step:latch"].count == 2
    assert link.stats["step:latch"].average_s == pytest.approx(0.017)
    assert "step:latch \u00d72 avg 17 ms max 18 ms" in link.summary()
    assert link.totals["step:latch"].count == 2
