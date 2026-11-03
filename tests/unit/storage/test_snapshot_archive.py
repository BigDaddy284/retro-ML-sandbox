"""T-201: SQLite snapshot archive."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.storage import SnapshotArchive


@pytest.fixture
def archive(tmp_path: Path) -> SnapshotArchive:
    a = SnapshotArchive(tmp_path / "snapshots.sqlite3")
    yield a  # type: ignore[misc]
    a.close()


def snap(frame: int, *, ram: bytes = b"\x00\x01", extra: bytes | None = None) -> MemorySnapshot:
    regions = {"ram": ram}
    if extra is not None:
        regions["sram"] = extra
    return MemorySnapshot(frame, regions)


def test_sessions_lifecycle(archive: SnapshotArchive) -> None:
    assert archive.sessions() == []
    s1 = archive.new_session("morning", "Toy Racer")
    s2 = archive.new_session("evening", "Toy Racer")
    assert archive.sessions() == [(s1, "morning", "Toy Racer"), (s2, "evening", "Toy Racer")]
    archive.delete_session(s1)
    assert archive.sessions() == [(s2, "evening", "Toy Racer")]
    with pytest.raises(ConfigurationError):
        archive.new_session("", "x")
    with pytest.raises(ConfigurationError):
        archive.new_session("x", "  ")


def test_append_count_iterate_and_latest(archive: SnapshotArchive) -> None:
    sid = archive.new_session("s", "g")
    assert archive.count(sid) == 0
    assert archive.latest(sid) is None
    assert list(archive.iter_session(sid)) == []
    assert archive.append(sid, snap(0, extra=b"")) == 0
    assert archive.append(sid, snap(1, ram=b"\xff\x00", extra=b"\x7f")) == 1
    assert archive.append(sid, snap(5, ram=bytes(2), extra=b"")) == 2
    assert archive.count(sid) == 3
    stored = list(archive.iter_session(sid))
    assert stored == [
        snap(0, extra=b""),
        snap(1, ram=b"\xff\x00", extra=b"\x7f"),
        snap(5, ram=bytes(2), extra=b""),
    ]
    assert archive.latest(sid) == stored[-1]
    assert archive.region_history(sid, "ram") == [(0, b"\x00\x01"), (1, b"\xff\x00"), (5, bytes(2))]
    assert archive.region_history(sid, "nope") == []


def test_sessions_are_isolated(archive: SnapshotArchive) -> None:
    a = archive.new_session("a", "g")
    b = archive.new_session("b", "g")
    archive.append(a, snap(0))
    archive.append(b, snap(10))
    archive.append(b, snap(11))
    assert archive.count(a) == 1 and archive.count(b) == 2
    assert [s.frame for s in archive.iter_session(b)] == [10, 11]
    archive.delete_session(b)
    assert archive.count(b) == 0
    assert archive.count(a) == 1


def test_append_to_unknown_session_is_an_error(archive: SnapshotArchive) -> None:
    with pytest.raises(ConfigurationError, match="No session 99"):
        archive.append(99, snap(0))


def test_data_survives_reopen_and_sequence_continues(tmp_path: Path) -> None:
    path = tmp_path / "a.sqlite3"
    with SnapshotArchive(path) as archive:
        sid = archive.new_session("s", "g")
        archive.append(sid, snap(0, ram=bytes(range(256))))
        archive.append(sid, snap(1))
    with SnapshotArchive(path) as reopened:
        assert reopened.sessions() == [(sid, "s", "g")]
        assert reopened.count(sid) == 2
        first = next(iter(reopened.iter_session(sid)))
        assert first.get("ram") == bytes(range(256))
        assert reopened.append(sid, snap(2)) == 2
        assert reopened.count(sid) == 3


def test_binary_integrity_with_awkward_bytes(archive: SnapshotArchive) -> None:
    sid = archive.new_session("s", "g")
    awkward = b"\x00" * 10 + b"\xff" * 10 + bytes(range(256)) + b"'\"\\;--"
    archive.append(sid, snap(3, ram=awkward))
    latest = archive.latest(sid)
    assert latest is not None and latest.get("ram") == awkward


def test_directory_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="folder"):
        SnapshotArchive(tmp_path)


def test_unwritable_path_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Can't open"):
        SnapshotArchive(tmp_path / "missing_dir" / "x.sqlite3")


@pytest.mark.slow
def test_a_thousand_snapshots_store_quickly(archive: SnapshotArchive) -> None:
    sid = archive.new_session("s", "g")
    payload = bytes(range(256)) * 32  # 8 KiB, bigger than any retro work RAM we start with
    start = time.perf_counter()
    for i in range(1000):
        archive.append(sid, MemorySnapshot(i, {"ram": payload}))
    elapsed = time.perf_counter() - start
    assert archive.count(sid) == 1000
    assert elapsed < 10, f"{elapsed:.1f}s for 1000 snapshots"


# ------------------------------------------------------------- inputs (T-204)


def test_inputs_are_stored_per_session_in_order(archive: SnapshotArchive) -> None:
    from retro_ai_lab.observation.inputs import InputEvent

    s1 = archive.new_session("one", "toy")
    s2 = archive.new_session("two", "toy")
    assert archive.inputs(s1) == []
    e1 = InputEvent(0, frozenset({"up"}))
    e2 = InputEvent(7, frozenset({"up", "a"}), "ai")
    assert archive.append_input(s1, e1) == 0
    assert archive.append_input(s1, e2) == 1
    assert archive.append_input(s2, InputEvent(3, frozenset())) == 0
    assert archive.inputs(s1) == [e1, e2]
    assert archive.inputs(s2) == [InputEvent(3, frozenset())]
    archive.delete_session(s1)
    assert archive.inputs(s1) == []
    with pytest.raises(ConfigurationError):
        archive.append_input(999, e1)
    with pytest.raises(ConfigurationError):
        archive.append_input(s2, "up")  # type: ignore[arg-type]
