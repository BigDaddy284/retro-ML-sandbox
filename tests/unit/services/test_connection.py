"""T-006: connection service state machine and events."""

from __future__ import annotations

from typing import Any

import pytest

from retro_ai_lab.core import (
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
    GameInfo,
)
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.emulator.registry import AdapterRegistry, default_registry
from retro_ai_lab.services import (
    ConnectionFailed,
    ConnectionService,
    ConnectionState,
    ConnectionStateChanged,
    EmulatorConnected,
    EmulatorDisconnected,
    Session,
)


@pytest.fixture
def bus() -> EventBus:
    return EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))


@pytest.fixture
def events(bus: EventBus) -> list[Any]:
    seen: list[Any] = []
    for t in (ConnectionStateChanged, EmulatorConnected, EmulatorDisconnected, ConnectionFailed):
        bus.subscribe(t, seen.append)
    return seen


@pytest.fixture
def service(bus: EventBus) -> ConnectionService:
    return ConnectionService(default_registry(), bus)


def states(events: list[Any]) -> list[tuple[ConnectionState, ConnectionState]]:
    return [(e.old, e.new) for e in events if isinstance(e, ConnectionStateChanged)]


# ------------------------------------------------------------- queries


def test_initial_state(service: ConnectionService) -> None:
    assert service.state is ConnectionState.DISCONNECTED
    assert not service.is_connected
    assert service.adapter is None
    assert service.session is None
    assert service.available_adapters() == (("fake", "Practice console (Toy Racer)"),)


# ------------------------------------------------------------- connect


def test_connect_happy_path(service: ConnectionService, events: list[Any]) -> None:
    session = service.connect("fake", seed=5)
    assert isinstance(session, Session)
    assert session == Session(
        adapter_name="fake",
        display_name="Practice console (Toy Racer)",
        game_info=GameInfo("TOY", "Toy Racer", rom_hash="toyracer-5"),
        capabilities=Capabilities.all(),
        buttons=BUTTONS,
    )
    assert service.is_connected
    assert service.session == session
    assert isinstance(service.adapter, FakeEmulatorAdapter)
    assert states(events) == [
        (ConnectionState.DISCONNECTED, ConnectionState.CONNECTING),
        (ConnectionState.CONNECTING, ConnectionState.CONNECTED),
    ]
    connected = [e for e in events if isinstance(e, EmulatorConnected)]
    assert connected == [EmulatorConnected("fake", session.game_info, Capabilities.all(), BUTTONS)]
    # Events arrive in a sensible order: CONNECTED state before the connected event.
    assert isinstance(events[-1], EmulatorConnected)
    assert isinstance(events[-2], ConnectionStateChanged)


def test_connect_passes_timeout(service: ConnectionService) -> None:
    with pytest.raises(EmulatorError, match=r"within 0\.5 seconds"):
        service.connect("fake", timeout_s=0.5, connect_delay_s=1.0)
    service.connect("fake", timeout_s=2.0, connect_delay_s=1.0)
    assert service.is_connected


def test_unknown_adapter_leaves_state_untouched(
    service: ConnectionService, events: list[Any]
) -> None:
    with pytest.raises(ConfigurationError, match="No emulator adapter"):
        service.connect("nope")
    assert service.state is ConnectionState.DISCONNECTED
    assert events == []


def test_bad_options_leave_state_untouched(service: ConnectionService, events: list[Any]) -> None:
    with pytest.raises(ConfigurationError, match="Bad options"):
        service.connect("fake", colour="red")
    assert service.state is ConnectionState.DISCONNECTED
    assert events == []


def test_failed_connect_returns_to_disconnected_and_reports(
    service: ConnectionService, events: list[Any]
) -> None:
    with pytest.raises(EmulatorError, match="Nothing is listening"):
        service.connect("fake", fail_connect=EmulatorError("Nothing is listening."))
    assert service.state is ConnectionState.DISCONNECTED
    assert service.adapter is None
    assert states(events) == [
        (ConnectionState.DISCONNECTED, ConnectionState.CONNECTING),
        (ConnectionState.CONNECTING, ConnectionState.DISCONNECTED),
    ]
    assert events[-1] == ConnectionFailed("fake", "Nothing is listening.")
    assert not any(isinstance(e, EmulatorConnected) for e in events)
    # And we can try again.
    service.connect("fake")
    assert service.is_connected


def test_failed_connect_with_empty_message_uses_class_name(
    service: ConnectionService, events: list[Any]
) -> None:
    with pytest.raises(OSError):
        service.connect("fake", fail_connect=OSError())
    assert events[-1] == ConnectionFailed("fake", "OSError")


def test_double_connect_is_rejected(service: ConnectionService, events: list[Any]) -> None:
    service.connect("fake")
    n = len(events)
    with pytest.raises(ConfigurationError, match="Already connected"):
        service.connect("fake")
    assert service.is_connected
    assert len(events) == n


# ---------------------------------------------------------- disconnect


def test_disconnect_happy_path(service: ConnectionService, events: list[Any]) -> None:
    service.connect("fake")
    adapter = service.adapter
    assert adapter is not None
    service.disconnect()
    assert service.state is ConnectionState.DISCONNECTED
    assert service.adapter is None
    assert service.session is None
    assert not adapter.is_connected
    assert states(events)[-1] == (ConnectionState.CONNECTED, ConnectionState.DISCONNECTED)
    assert events[-1] == EmulatorDisconnected("fake", True, "Disconnected.")


def test_disconnect_when_idle_is_a_silent_noop(
    service: ConnectionService, events: list[Any]
) -> None:
    service.disconnect()
    assert events == []
    service.connect("fake")
    service.disconnect()
    n = len(events)
    service.disconnect()
    assert len(events) == n


def test_reconnect_after_disconnect(service: ConnectionService) -> None:
    service.connect("fake", seed=1)
    service.disconnect()
    session = service.connect("fake", seed=2)
    assert session.game_info.rom_hash == "toyracer-2"


# ---------------------------------------------------------------- call


def test_call_runs_operation_on_live_adapter(service: ConnectionService) -> None:
    service.connect("fake")
    frame = service.call(lambda a: a.capture_frame())
    assert frame.number == 0
    service.call(lambda a: a.advance_frames(4))
    assert service.call(lambda a: a.capture_frame()).number == 4


def test_call_without_connection_raises(service: ConnectionService, events: list[Any]) -> None:
    with pytest.raises(EmulatorDisconnectedError):
        service.call(lambda a: a.capture_frame())
    assert events == []


def test_call_detects_dropped_link_once(service: ConnectionService, events: list[Any]) -> None:
    service.connect("fake", drop_after_ops=1)
    service.call(lambda a: a.capture_frame())
    with pytest.raises(EmulatorDisconnectedError, match="stopped responding"):
        service.call(lambda a: a.capture_frame())
    assert service.state is ConnectionState.DISCONNECTED
    assert service.adapter is None
    assert events[-1] == EmulatorDisconnected("fake", False, "The emulator stopped responding.")
    assert states(events)[-1] == (ConnectionState.CONNECTED, ConnectionState.DISCONNECTED)
    n = len(events)
    with pytest.raises(EmulatorDisconnectedError):
        service.call(lambda a: a.capture_frame())
    assert len(events) == n  # reported exactly once


def test_call_passes_other_errors_through_without_disconnecting(
    service: ConnectionService, events: list[Any]
) -> None:
    service.connect("fake")
    n = len(events)
    with pytest.raises(ConfigurationError):
        service.call(lambda a: a.advance_frames(0))
    assert service.is_connected
    assert len(events) == n


# ------------------------------------------------------ custom registry


def test_service_uses_given_registry(bus: EventBus) -> None:
    reg = AdapterRegistry()
    service = ConnectionService(reg, bus)
    assert service.available_adapters() == ()
    with pytest.raises(ConfigurationError):
        service.connect("fake")
