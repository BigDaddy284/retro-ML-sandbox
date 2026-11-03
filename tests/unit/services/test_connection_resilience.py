"""T-011: the service notices silent link loss and survives bad disconnects."""

from __future__ import annotations

from typing import Any

import pytest

from retro_ai_lab.core import EmulatorError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.services import (
    LINK_LOST,
    ConnectionService,
    ConnectionState,
    EmulatorDisconnected,
)


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
