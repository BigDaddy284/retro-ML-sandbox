"""T-1063: a starting spot saved again in the state it was saved in is the same spot.

The harness counts the saves of its checkpoint (``taken``), and whatever was worked
out from a spot was kept by that count: the way the AI remembers, above all. So a
pinned spot, which is put back and saved again at every Start, was a new spot each
time and the way was forgotten. The harness now also counts the spots (``spots``): a
spot is told by the game's memory as it is read after the save.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retro_ai_lab.core import (
    EmulatorDisconnectedError,
    EmulatorError,
    MemoryRegion,
    ProtocolError,
)
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.services import ConnectionService
from retro_ai_lab.services.experiments import ExperimentHarness, constant_plan

Reads = Sequence[tuple[str, int, int]]


@pytest.fixture
def connection() -> ConnectionService:
    return ConnectionService(default_registry(), EventBus())


@pytest.fixture
def harness(connection: ConnectionService) -> ExperimentHarness:
    connection.connect("fake", seed=5)
    return ExperimentHarness(connection)


def game(connection: ConnectionService) -> FakeEmulatorAdapter:
    a = connection.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    return a


def reads_counted(connection: ConnectionService, monkeypatch: pytest.MonkeyPatch) -> list[Reads]:
    """Every read of memory the adapter is asked for from here on, in the list."""
    adapter = game(connection)
    real = adapter.read_many
    asked: list[Reads] = []

    def read_many(reads: Reads) -> tuple[bytes, ...]:
        asked.append(tuple(reads))
        return real(reads)

    monkeypatch.setattr(adapter, "read_many", read_many)
    return asked


def memory_is(
    connection: ConnectionService, monkeypatch: pytest.MonkeyPatch, **regions: bytes
) -> None:
    """The emulator offers these regions with these bytes, whatever the game does."""
    adapter = game(connection)
    offered = tuple(MemoryRegion(name, len(data)) for name, data in regions.items())
    monkeypatch.setattr(adapter, "memory_regions", lambda: offered)
    monkeypatch.setattr(
        adapter, "read_many", lambda reads: tuple(regions[name] for name, _, _ in reads)
    )


# ----------------------------------------------------------- the same spot


def test_the_same_state_saved_again_is_the_same_spot(
    harness: ExperimentHarness, connection: ConnectionService
) -> None:
    assert (harness.spots, harness.taken) == (0, 0)
    harness.checkpoint()
    assert (harness.spots, harness.taken) == (1, 1) and harness.has_checkpoint
    harness.checkpoint()  # nothing has moved
    assert (harness.spots, harness.taken) == (1, 2)  # the saves are counted as they were
    # Played on from it and put back: the game is in the state it was saved in.
    game(connection).advance_frames(7)
    harness.restore()
    harness.checkpoint()
    assert harness.spots == 1 and harness.has_checkpoint
    # A trial from it puts it back first, and leaves the game somewhere else.
    harness.run(constant_plan(["up"], 2))
    harness.restore()
    harness.checkpoint()
    assert (harness.spots, harness.taken) == (1, 4)


def test_a_save_after_the_game_has_moved_is_another_spot(
    harness: ExperimentHarness, connection: ConnectionService
) -> None:
    harness.checkpoint()
    game(connection).advance_frames(1)
    harness.checkpoint()
    assert harness.spots == 2
    harness.checkpoint()
    assert harness.spots == 2  # and that one saved again is still the second
    game(connection).advance_frames(3)
    harness.checkpoint()
    assert harness.spots == 3


def test_a_spot_is_compared_with_the_save_before_it_only(
    harness: ExperimentHarness, connection: ConnectionService
) -> None:
    """Back at the first spot after another, the count goes up again: it says the spot
    changed, and never goes down. What was worked out from the first was dropped when
    the second was taken."""
    adapter = game(connection)
    adapter.save_state("first")
    harness.checkpoint()
    adapter.advance_frames(5)
    harness.checkpoint()
    assert harness.spots == 2
    adapter.load_state("first")
    harness.checkpoint()
    assert harness.spots == 3
    harness.checkpoint()
    assert harness.spots == 3


def test_each_harness_counts_its_own_spots(connection: ConnectionService) -> None:
    connection.connect("fake", seed=5)
    one, other = ExperimentHarness(connection, slot="one"), ExperimentHarness(connection, slot="b")
    one.checkpoint()
    other.checkpoint()
    one.checkpoint()
    assert (one.spots, other.spots) == (1, 1)
    game(connection).advance_frames(2)
    other.checkpoint()
    assert (one.spots, other.spots) == (1, 2)


# ----------------------------------------------------------- what it is told by


def test_one_read_of_memory_for_each_save_and_none_otherwise(
    harness: ExperimentHarness, connection: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = reads_counted(connection, monkeypatch)
    regions = game(connection).memory_regions()
    whole = tuple((r.name, 0, r.size) for r in regions)
    harness.checkpoint()
    assert asked == [whole]  # every region the emulator offers, whole, in one request
    harness.restore()
    assert asked == [whole]
    harness.checkpoint()
    assert asked == [whole, whole]
    assert not harness.trials and harness.spots == 1


def test_any_byte_of_any_region_tells(
    harness: ExperimentHarness, connection: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory_is(connection, monkeypatch, ram=b"\x01\x02\x03", vram=b"\x09")
    harness.checkpoint()
    harness.checkpoint()
    assert harness.spots == 1
    memory_is(connection, monkeypatch, ram=b"\x01\x02\x03", vram=b"\x08")
    harness.checkpoint()
    assert harness.spots == 2
    memory_is(connection, monkeypatch, ram=b"\x01\x02\x04", vram=b"\x08")
    harness.checkpoint()
    assert harness.spots == 3
    memory_is(connection, monkeypatch, vram=b"\x08", ram=b"\x01\x02\x04")
    harness.checkpoint()
    assert harness.spots == 3  # the order the regions are offered in is not the spot


def test_the_same_bytes_in_other_regions_are_another_spot(
    harness: ExperimentHarness, connection: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory_is(connection, monkeypatch, a=b"12", b=b"3")
    harness.checkpoint()
    memory_is(connection, monkeypatch, a=b"1", b=b"23")  # split differently
    harness.checkpoint()
    assert harness.spots == 2
    memory_is(connection, monkeypatch, a=b"1", c=b"23")  # under another name
    harness.checkpoint()
    assert harness.spots == 3
    memory_is(connection, monkeypatch, a=b"1")  # a region gone
    harness.checkpoint()
    assert harness.spots == 4
    memory_is(connection, monkeypatch, a=b"1", c=b"\x00")  # and one come
    harness.checkpoint()
    assert harness.spots == 5
    assert harness.taken == 5


# ----------------------------------------------------------- where it cannot tell


def test_with_no_memory_to_read_every_save_is_a_new_spot(
    harness: ExperimentHarness, connection: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(game(connection), "memory_regions", lambda: ())
    for count in (1, 2, 3):
        harness.checkpoint()  # the game has not moved
        assert harness.spots == count and harness.has_checkpoint
    harness.restore()


@pytest.mark.parametrize("error", [ProtocolError("The emulator refused."), EmulatorError("No.")])
def test_a_read_the_emulator_refuses_is_a_new_spot_and_the_save_stands(
    harness: ExperimentHarness,
    connection: ConnectionService,
    monkeypatch: pytest.MonkeyPatch,
    error: EmulatorError,
) -> None:
    adapter = game(connection)
    harness.checkpoint()
    with monkeypatch.context() as patch:

        def refuse(reads: Reads) -> tuple[bytes, ...]:
            raise error

        patch.setattr(adapter, "read_many", refuse)
        harness.checkpoint()  # the same state, and nothing to tell it by
        assert harness.spots == 2 and harness.has_checkpoint
        harness.checkpoint()
        assert harness.spots == 3
    # Read again, it is a spot not told before: counted once, and then known.
    harness.checkpoint()
    assert harness.spots == 4
    harness.checkpoint()
    assert harness.spots == 4
    harness.restore()


def test_a_link_that_drops_at_the_read_shows_at_the_next_call(
    harness: ExperimentHarness, connection: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = game(connection)
    harness.checkpoint()
    saved: list[str] = []
    save = adapter.save_state
    monkeypatch.setattr(adapter, "save_state", lambda slot: (saved.append(slot), save(slot))[1])

    def drop(reads: Reads) -> tuple[bytes, ...]:
        adapter.mark_disconnected()
        raise EmulatorDisconnectedError("The emulator stopped responding.")

    monkeypatch.setattr(adapter, "read_many", drop)
    harness.checkpoint()  # the save was made before the link went
    assert saved == [harness.slot]
    # It may be another spot for all the harness can tell: counted.
    assert harness.spots == 2 and harness.has_checkpoint
    assert not harness.available()
    with pytest.raises(EmulatorDisconnectedError):
        harness.restore()
    with pytest.raises(EmulatorDisconnectedError):
        harness.checkpoint()
    assert harness.spots == 2 and saved == [harness.slot]


def test_a_save_that_fails_counts_nothing(
    harness: ExperimentHarness, connection: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = game(connection)
    harness.checkpoint()
    asked = reads_counted(connection, monkeypatch)

    def fail(slot: str) -> None:
        raise ProtocolError("The emulator could not save.")

    with monkeypatch.context() as patch:
        patch.setattr(adapter, "save_state", fail)
        with pytest.raises(ProtocolError):
            harness.checkpoint()
    assert harness.spots == 1 and asked == []
    adapter.advance_frames(2)
    harness.restore()
    harness.checkpoint()
    assert harness.spots == 1  # and the spot it had is still the one it is told by
