"""T-202: the observation service snapshots memory as frames advance."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import Capabilities, EmulatorError, Frame
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.services import ConnectionService
from retro_ai_lab.services.observer import MemoryObserved, ObservationService

UP = next(b for b in BUTTONS if b.name == "up")


@pytest.fixture
def bus() -> EventBus:
    return EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))


@pytest.fixture
def connection(bus: EventBus) -> ConnectionService:
    return ConnectionService(default_registry(), bus)


@pytest.fixture
def observer(connection: ConnectionService, bus: EventBus) -> ObservationService:
    return ObservationService(connection, bus, ring_capacity=10)


@pytest.fixture
def events(bus: EventBus) -> list[MemoryObserved]:
    seen: list[MemoryObserved] = []
    bus.subscribe(MemoryObserved, seen.append)
    return seen


def fake(connection: ConnectionService) -> FakeEmulatorAdapter:
    a = connection.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    return a


def test_idle_until_connected(observer: ObservationService) -> None:
    assert not observer.enabled
    assert observer.observe() is None
    assert observer.advance() is None
    assert observer.frame == 1  # frames are still counted
    assert observer.latest is None


def test_observes_on_connect_and_each_advance(
    observer: ObservationService, connection: ConnectionService, events: list[MemoryObserved]
) -> None:
    connection.connect("fake", seed=1)
    assert observer.enabled and observer.frame == 0
    first = observer.observe()
    assert first is not None and first.diff is None
    assert first.snapshot.frame == 0
    assert first.snapshot.get("ram") == fake(connection).game.ram

    connection.call(lambda a: a.set_input([UP]))
    connection.call(lambda a: a.advance_frames(1))
    second = observer.advance(1)
    assert second is not None and second.diff is not None
    assert second.snapshot.frame == 1
    assert LAYOUT.speed in second.diff.offsets("ram")
    assert len(observer.ring) == 2
    assert observer.tracker.observed == 2
    assert events == [first, second]


def test_ring_and_tracker_reset_on_reconnect(
    observer: ObservationService, connection: ConnectionService
) -> None:
    connection.connect("fake")
    observer.observe()
    observer.advance(3)
    assert len(observer.ring) == 2 and observer.frame == 3
    connection.disconnect()
    assert not observer.enabled
    assert observer.observe() is None
    assert len(observer.ring) == 2  # kept until the next connect, for review
    connection.connect("fake")
    assert len(observer.ring) == 0 and observer.frame == 0
    assert observer.tracker.observed == 0


def test_emulator_without_memory_read(
    observer: ObservationService, connection: ConnectionService
) -> None:
    connection.connect("fake", capabilities=Capabilities(screen_capture=True, frame_advance=True))
    assert not observer.enabled
    assert observer.advance() is None


def test_dropped_link_is_quiet_and_disables(
    observer: ObservationService, connection: ConnectionService, events: list[MemoryObserved]
) -> None:
    connection.connect("fake", drop_after_ops=0)
    assert observer.observe() is None
    assert not connection.is_connected
    assert not observer.enabled
    assert observer.errors == 0
    assert events == []


def test_transient_read_error_is_counted(
    observer: ObservationService, connection: ConnectionService
) -> None:
    connection.connect("fake")
    adapter = fake(connection)
    original = adapter._do_read_memory
    calls = {"n": 0}

    def flaky(region: str, offset: int, length: int) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmulatorError("bus glitch")
        return original(region, offset, length)

    adapter._do_read_memory = flaky  # type: ignore[method-assign]
    assert observer.observe() is None
    assert observer.errors == 1
    assert observer.observe() is not None
    assert observer.enabled


def test_ring_capacity_is_honoured(
    observer: ObservationService, connection: ConnectionService
) -> None:
    connection.connect("fake")
    for _ in range(15):
        observer.advance()
    assert len(observer.ring) == 10
    assert observer.ring[0].frame == 6
    assert observer.latest is not None and observer.latest.frame == 15


def test_advance_rejects_negative(observer: ObservationService) -> None:
    with pytest.raises(ValueError):
        observer.advance(-1)


# ------------------------------------------------------------- screen (T-203)


def test_observe_frame_tracks_screen_and_publishes(
    observer: ObservationService, connection: ConnectionService, bus: EventBus
) -> None:
    from retro_ai_lab.services.observer import ScreenObserved

    seen: list[ScreenObserved] = []
    bus.subscribe(ScreenObserved, seen.append)
    frame = fake_frame()
    assert observer.observe_frame(frame) is None  # not connected: ignored
    connection.connect("fake", seed=2)
    first = observer.observe_frame(connection.call(lambda a: a.capture_frame()))
    assert first is not None and first.diff is None
    connection.call(lambda a: a.set_input([UP]))
    connection.call(lambda a: a.advance_frames(5))
    second = observer.observe_frame(connection.call(lambda a: a.capture_frame()))
    assert second is not None and second.diff is not None
    assert observer.frames_seen == 2 and observer.screen.transitions == 1
    assert seen == [first, second]
    connection.disconnect()
    connection.connect("fake")
    assert observer.frames_seen == 0 and observer.screen.observed == 0


def fake_frame() -> Frame:
    return Frame(4, 4, 1, bytes(16))
