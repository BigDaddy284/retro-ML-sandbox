"""T-202: the observation service snapshots memory as frames advance."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import Capabilities, ConfigurationError, EmulatorError, Frame
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


def test_the_observation_service_forwards_expect_jump(
    observer: ObservationService, connection: ConnectionService
) -> None:
    """T-935: a restore announced to the service is skipped by its tracker."""
    connection.connect("fake")
    observer.observe()
    observer.advance(4)
    counted = observer.tracker.transitions
    connection.call(lambda a: a.save_state("t935"))
    observer.advance(4)
    observer.expect_jump()
    assert observer.tracker.jump_expected
    connection.call(lambda a: a.load_state("t935"))
    assert observer.observe() is not None
    assert observer.tracker.transitions == counted + 1 and not observer.tracker.jump_expected


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


# ------------------------------------------------------------- inputs (T-204)


def test_record_input_tags_the_current_frame_and_publishes(
    observer: ObservationService, connection: ConnectionService, bus: EventBus
) -> None:
    from retro_ai_lab.services.observer import InputRecorded

    seen: list[InputRecorded] = []
    bus.subscribe(InputRecorded, seen.append)
    assert observer.record_input(["up"]) is None  # not connected
    connection.connect("fake")
    first = observer.record_input(["up"])
    assert first is not None and first.event.frame == 0 and first.event.source == "human"
    observer.advance(4)
    assert observer.record_input(["up"]) is None  # unchanged
    second = observer.record_input(["up", "a"], "ai")
    assert second is not None and second.event.frame == 4
    assert observer.inputs.held_at(2) == {"up"}
    assert seen == [first, second]
    with pytest.raises(ConfigurationError):
        observer.record_input(["up"], "ghost")
    connection.disconnect()
    connection.connect("fake")
    assert len(observer.inputs) == 0


# ------------------------------------------------- T-823 prefetched snapshots


def test_snapshot_reads_name_every_region(
    observer: ObservationService, connection: ConnectionService
) -> None:
    assert observer.snapshot_reads() == ()  # not connected
    connection.connect("fake")
    assert observer.snapshot_reads() == (("ram", 0, len(fake(connection).game.ram)),)
    connection.disconnect()
    assert observer.snapshot_reads() == ()
    connection.connect("fake", capabilities=Capabilities(frame_advance=True))
    assert observer.snapshot_reads() == ()  # no memory to read


def test_advance_with_prefetched_bytes_reads_nothing(
    observer: ObservationService, connection: ConnectionService, events: list[MemoryObserved]
) -> None:
    connection.connect("fake", seed=1)
    adapter = fake(connection)
    reads = observer.snapshot_reads()
    adapter.set_input([UP])
    result = adapter.step(2, reads=reads)
    original = adapter._do_read_many
    adapter._do_read_many = lambda r: (_ for _ in ()).throw(AssertionError("read"))  # type: ignore[method-assign]
    event = observer.advance(2, snapshot=result.reads)
    adapter._do_read_many = original  # type: ignore[method-assign]
    assert event is not None and event.snapshot.frame == 2
    assert event.snapshot.get("ram") == adapter.game.ram
    assert observer.latest == event.snapshot and events == [event]
    assert len(observer.ring) == 1


def test_prefetched_bytes_with_the_wrong_shape_are_rejected(
    observer: ObservationService, connection: ConnectionService
) -> None:
    connection.connect("fake")
    with pytest.raises(ConfigurationError, match="planned"):
        observer.advance(1, snapshot=[b"", b""])
    assert observer.latest is None and observer.errors == 0


def test_a_jump_back_marks_only_the_next_snapshot(
    observer: ObservationService, connection: ConnectionService, events: list[MemoryObserved]
) -> None:
    """T-1004: the app put the game back to an earlier moment of the go, and says so."""
    connection.connect("fake")
    observer.expect_jump(back=True)  # before the first snapshot there is nothing to mark
    observer.observe()
    observer.advance(4)
    assert [e.went_back for e in events] == [False, False]
    connection.call(lambda a: a.save_state("t1004"))
    observer.advance(4)
    counted = observer.tracker.transitions
    # A plain restore (a checkpoint, a practise spot) marks nothing.
    observer.expect_jump()
    connection.call(lambda a: a.load_state("t1004"))
    observer.observe()
    assert events[-1].went_back is False
    observer.advance(4)
    # A jump back marks the next snapshot, is skipped by the byte statistics like any
    # restore, and marks no snapshot after it.
    observer.expect_jump(back=True)
    assert observer.tracker.jump_expected
    connection.call(lambda a: a.load_state("t1004"))
    marked = observer.observe()
    assert marked is not None and marked.went_back is True and events[-1] is marked
    # One step was played since the count was taken; neither jump is counted.
    assert observer.tracker.transitions == counted + 1
    observer.advance(4)
    assert events[-1].went_back is False
    # A plain restore asked for after a jump back takes the mark off again.
    observer.expect_jump(back=True)
    observer.expect_jump()
    observer.observe()
    assert events[-1].went_back is False


def test_the_first_snapshot_after_a_saved_state_is_loaded_is_marked(
    observer: ObservationService, connection: ConnectionService, events: list[MemoryObserved]
) -> None:
    """T-1021: whoever loaded it, the values it shows were put there by the app."""
    connection.connect("fake")
    adapter = fake(connection)
    assert adapter.states_loaded == 0
    observer.observe()
    observer.advance(4)
    adapter.save_state("t1021")
    observer.advance(4)
    assert adapter.states_loaded == 0  # saving is not loading
    assert [e.put_back for e in events] == [False, False, False]
    # A load nobody told the observer about (the checkpoint at the start of a go, an
    # experiment putting the game back): the next snapshot is marked, and only that one.
    adapter.load_state("t1021")
    assert adapter.states_loaded == 1
    marked = observer.observe()
    assert marked is not None and marked.put_back and not marked.went_back
    observer.advance(4)
    assert [e.put_back for e in events[-2:]] == [True, False]
    # Several loads between two snapshots (a search's tries) are one mark.
    for _ in range(3):
        adapter.load_state("t1021")
    assert adapter.states_loaded == 4
    observer.advance(4)
    observer.advance(4)
    assert [e.put_back for e in events[-2:]] == [True, False]
    # A load that did not load is not counted, and marks nothing.
    with pytest.raises(ConfigurationError):
        adapter.load_state("never_saved_t1021")
    assert adapter.states_loaded == 4
    observer.advance(4)
    assert events[-1].put_back is False
    # A jump the observer was told of is marked too, with what it was told.
    observer.expect_jump(back=True)
    adapter.load_state("t1021")
    back = observer.observe()
    assert back is not None and back.put_back and back.went_back
    observer.expect_jump()  # told of a jump, but nothing was loaded: no mark
    observer.advance(4)
    assert (events[-1].put_back, events[-1].went_back) == (False, False)
    # Another connection starts its own count: what was loaded before is not its doing.
    connection.disconnect()
    connection.connect("fake")
    first = observer.observe()
    assert first is not None and first.put_back is False
    assert fake(connection).states_loaded == 0
