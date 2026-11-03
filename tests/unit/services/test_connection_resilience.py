"""T-011: the service notices silent link loss and survives bad disconnects."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

import pytest

from retro_ai_lab.core import EmulatorDisconnectedError, EmulatorError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.services import (
    LINK_LOST,
    ConnectionService,
    ConnectionState,
    EmulatorDisconnected,
)
from retro_ai_lab.services import connection as connection_module


@pytest.fixture
def bus() -> EventBus:
    return EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))


@pytest.fixture
def events(bus: EventBus) -> list[Any]:
    seen: list[Any] = []
    bus.subscribe(EmulatorDisconnected, seen.append)
    return seen


@pytest.fixture
def service(bus: EventBus) -> ConnectionService:
    return ConnectionService(default_registry(), bus)


def fake(service: ConnectionService) -> FakeEmulatorAdapter:
    a = service.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    return a


# --------------------------------------------------------------- check


def test_check_is_false_and_silent_when_not_connected(
    service: ConnectionService, events: list[Any]
) -> None:
    assert service.check() is False
    assert events == []


def test_check_is_true_and_silent_while_link_is_up(
    service: ConnectionService, events: list[Any]
) -> None:
    service.connect("fake")
    assert service.check() is True
    assert service.check() is True
    assert events == []
    assert service.is_connected


def test_check_reports_silent_drop_exactly_once(
    service: ConnectionService, events: list[Any]
) -> None:
    service.connect("fake")
    fake(service).drop_link()  # nothing was being sent
    assert service.check() is False
    assert service.state is ConnectionState.DISCONNECTED
    assert service.adapter is None
    assert events == [EmulatorDisconnected("fake", False, LINK_LOST)]
    assert service.check() is False
    assert len(events) == 1


def test_reconnect_after_silent_drop(service: ConnectionService) -> None:
    service.connect("fake", seed=9)
    fake(service).drop_link()
    service.check()
    session = service.connect("fake", seed=9)
    assert session.game_info.rom_hash == "toyracer-9"
    assert service.is_connected


# ---------------------------------------------------------- disconnect


def test_disconnect_returns_reason(service: ConnectionService, events: list[Any]) -> None:
    assert service.disconnect() == ""  # not connected
    service.connect("fake")
    assert service.disconnect() == "Disconnected."
    assert events == [EmulatorDisconnected("fake", True, "Disconnected.")]


def test_backend_error_during_disconnect_still_disconnects(
    service: ConnectionService, events: list[Any], caplog: pytest.LogCaptureFixture
) -> None:
    service.connect("fake")
    adapter = fake(service)

    def broken() -> None:
        raise EmulatorError("socket already closed")

    adapter._do_disconnect = broken  # type: ignore[method-assign]
    with caplog.at_level("WARNING"):
        reason = service.disconnect()
    assert service.state is ConnectionState.DISCONNECTED
    assert service.adapter is None
    assert not adapter.is_connected
    assert "socket already closed" in reason
    assert events == [EmulatorDisconnected("fake", True, reason)]
    assert "failed while disconnecting" in caplog.text


def test_unexpected_exception_during_disconnect_propagates_but_state_is_clean(
    service: ConnectionService, events: list[Any]
) -> None:
    service.connect("fake")
    adapter = fake(service)

    def broken() -> None:
        raise RuntimeError("bug in adapter")

    adapter._do_disconnect = broken  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="bug in adapter"):
        service.disconnect()
    assert service.state is ConnectionState.DISCONNECTED
    assert service.adapter is None
    assert len(events) == 1 and events[0].expected


# ----------------------------------------------------------- T-818 threads


def test_two_workers_dropping_at_once_report_the_loss_once(
    service: ConnectionService, events: list[Any]
) -> None:
    """The frame pump and the play clock may both hit the dead link together."""
    service.connect("fake")
    adapter = fake(service)
    start = threading.Barrier(2)

    def op(_: Any) -> None:
        start.wait(5)
        adapter.mark_disconnected()
        raise EmulatorDisconnectedError("The emulator went away.")

    def worker() -> None:
        with contextlib.suppress(EmulatorDisconnectedError):
            service.call(op)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert service.state is ConnectionState.DISCONNECTED
    assert [e for e in events if isinstance(e, EmulatorDisconnected)] == [
        EmulatorDisconnected("fake", False, "The emulator went away.")
    ]


def test_a_worker_cannot_turn_a_clean_disconnect_into_a_lost_link(
    service: ConnectionService, events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-820: Disconnect was pressed while a worker was mid-call.

    The worker's next call fails because the adapter is closing; it must find
    the disconnect already recorded rather than announce a lost link itself.
    """
    service.connect("fake")
    adapter = fake(service)
    capture_entered = threading.Event()
    capture_gate = threading.Event()
    closing = threading.Event()
    original_capture = adapter._do_capture_frame

    def slow_capture() -> Any:
        capture_entered.set()
        capture_gate.wait(5)
        return original_capture()

    def broken_disconnect() -> None:
        closing.set()
        raise EmulatorError("socket already closed")

    adapter._do_capture_frame = slow_capture  # type: ignore[method-assign]
    adapter._do_disconnect = broken_disconnect  # type: ignore[method-assign]
    # The service is slow between the backend's failure and its own bookkeeping.
    monkeypatch.setattr(connection_module.log, "warning", lambda *a, **k: time.sleep(0.1))

    worker_errors: list[Exception] = []

    def worker() -> None:
        service.call(lambda a: a.capture_frame())  # holds the adapter until the gate opens
        assert closing.wait(5)
        try:
            service.call(lambda a: a.capture_frame())  # waits for the adapter, then fails
        except EmulatorDisconnectedError as exc:
            worker_errors.append(exc)

    reasons: list[str] = []
    worker_thread = threading.Thread(target=worker)
    worker_thread.start()
    assert capture_entered.wait(5)
    disconnector = threading.Thread(target=lambda: reasons.append(service.disconnect()))
    disconnector.start()  # blocks on the adapter until the worker's capture ends
    time.sleep(0.05)
    capture_gate.set()
    worker_thread.join(5)
    disconnector.join(5)

    assert len(worker_errors) == 1
    assert reasons == ["Disconnected, although the emulator complained: socket already closed"]
    assert events == [EmulatorDisconnected("fake", True, reasons[0])]
    assert service.state is ConnectionState.DISCONNECTED
