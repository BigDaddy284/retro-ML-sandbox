"""T-201: memory snapshots, capture and the in-memory ring."""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import Capabilities, ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, LAYOUT, RAM_SIZE
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.observation import MemoryCapture, MemorySnapshot, SnapshotRing
from retro_ai_lab.services import ConnectionService

# ---------------------------------------------------------------- snapshot


def test_snapshot_basics() -> None:
    s = MemorySnapshot(5, {"ram": b"\x01\x02\x03", "vram": b"\xff"})
    assert s.frame == 5
    assert s.region_names == ("ram", "vram")
    assert s.size == 4
    assert s.get("ram") == b"\x01\x02\x03"
    assert s.get("ram", 1) == b"\x02\x03"
    assert s.get("ram", 1, 1) == b"\x02"
    assert s.get("ram", 3, 0) == b""
    assert s.byte("vram", 0) == 0xFF


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame": -1, "regions": {"ram": b""}},
        {"frame": 1.0, "regions": {"ram": b""}},
        {"frame": True, "regions": {"ram": b""}},
        {"frame": 0, "regions": {}},
        {"frame": 0, "regions": {"RAM": b""}},
        {"frame": 0, "regions": {"1ram": b""}},
        {"frame": 0, "regions": {"ram": "00"}},
        {"frame": 0, "regions": {"ram": bytearray(1)}},
        {"frame": 0, "regions": [("ram", b"")]},
    ],
)
def test_snapshot_rejects_bad_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        MemorySnapshot(**kwargs)


@pytest.mark.parametrize(
    ("region", "offset", "length"),
    [("x", 0, 1), ("ram", -1, 1), ("ram", 0, 4), ("ram", 3, 1), ("ram", 0, -1)],
)
def test_snapshot_get_rejects_bad_ranges(region: str, offset: int, length: int) -> None:
    s = MemorySnapshot(0, {"ram": b"abc"})
    with pytest.raises(ConfigurationError):
        s.get(region, offset, length)


def test_snapshot_is_immutable_and_compares_by_value() -> None:
    a = MemorySnapshot(1, {"ram": b"ab"})
    b = MemorySnapshot(1, {"ram": b"ab"})
    c = MemorySnapshot(2, {"ram": b"ab"})
    d = MemorySnapshot(1, {"ram": b"ac"})
    assert a == b and hash(a) == hash(b)
    assert a != c and a != d
    assert a != "snapshot"
    with pytest.raises(TypeError):
        a.regions["ram"] = b"zz"  # type: ignore[index]
    with pytest.raises(AttributeError):
        a.frame = 3  # type: ignore[misc]


def test_snapshot_copies_its_input_mapping() -> None:
    source = {"ram": b"ab"}
    s = MemorySnapshot(0, source)
    source["ram"] = b"zz"
    assert s.get("ram") == b"ab"


@settings(max_examples=30)
@given(frame=st.integers(0, 10**6), data=st.binary(max_size=64), extra=st.binary(max_size=8))
def test_snapshot_json_round_trip(frame: int, data: bytes, extra: bytes) -> None:
    s = MemorySnapshot(frame, {"ram": data, "sram": extra})
    back = MemorySnapshot.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back == s


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"frame": 0},
        {"frame": 0, "regions": {"ram": "00"}, "x": 1},
        {"frame": 0, "regions": "00"},
        {"frame": 0, "regions": {"ram": b"\x00"}},
        {"frame": 0, "regions": {"ram": "zz"}},
        "nope",
    ],
)
def test_snapshot_from_dict_rejects_bad_shapes(data: Any) -> None:
    with pytest.raises(ConfigurationError):
        MemorySnapshot.from_dict(data)


# ----------------------------------------------------------------- capture


@pytest.fixture
def service() -> ConnectionService:
    return ConnectionService(default_registry(), EventBus())


def fake(service: ConnectionService) -> FakeEmulatorAdapter:
    a = service.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    return a


def test_capture_reads_every_region_exactly(service: ConnectionService) -> None:
    service.connect("fake", seed=2)
    cap = MemoryCapture(service)
    assert [r.name for r in cap.available_regions()] == ["ram"]
    snap = cap.capture(frame=0)
    assert snap == MemorySnapshot(0, {"ram": fake(service).game.ram})
    assert snap.size == RAM_SIZE
    assert cap.captures == 1
    up = next(b for b in BUTTONS if b.name == "up")
    service.call(lambda a: a.set_input([up]))
    service.call(lambda a: a.advance_frames(3))
    later = cap.capture(frame=3)
    assert later.frame == 3
    assert later.byte("ram", LAYOUT.speed) == 3
    assert later.get("ram") == fake(service).game.ram
    assert cap.captures == 2


def test_capture_subset_and_unknown_regions(service: ConnectionService) -> None:
    service.connect("fake")
    assert MemoryCapture(service, ["ram"]).capture(0).region_names == ("ram",)
    with pytest.raises(ConfigurationError, match="Unknown memory regions"):
        MemoryCapture(service, ["vram"]).capture(0)


def test_capture_without_memory_capability(service: ConnectionService) -> None:
    service.connect("fake", capabilities=Capabilities(screen_capture=True))
    with pytest.raises(UnsupportedCapabilityError):
        MemoryCapture(service).capture(0)


def test_capture_without_connection(service: ConnectionService) -> None:
    from retro_ai_lab.core import EmulatorDisconnectedError

    with pytest.raises(EmulatorDisconnectedError):
        MemoryCapture(service).capture(0)


# -------------------------------------------------------------------- ring


def test_ring_keeps_the_newest_in_order() -> None:
    ring = SnapshotRing(capacity=3)
    assert ring.capacity == 3
    assert ring.latest is None and ring.previous() is None and len(ring) == 0
    for i in range(5):
        ring.append(MemorySnapshot(i, {"ram": bytes([i])}))
    assert len(ring) == 3
    assert [s.frame for s in ring] == [2, 3, 4]
    assert ring.latest is not None and ring.latest.frame == 4
    assert ring.previous(1) is not None and ring.previous(1).frame == 3
    assert ring.previous(0) is ring.latest
    assert ring.previous(3) is None
    assert ring.previous(-1) is None
    assert ring[0].frame == 2
    ring.clear()
    assert len(ring) == 0 and ring.latest is None


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True, "3"])
def test_ring_rejects_bad_capacity(capacity: Any) -> None:
    with pytest.raises(ConfigurationError):
        SnapshotRing(capacity)


def test_ring_rejects_non_snapshots() -> None:
    with pytest.raises(ConfigurationError):
        SnapshotRing().append("snap")  # type: ignore[arg-type]


def test_ring_default_capacity_is_ten_seconds_at_sixty_fps() -> None:
    assert SnapshotRing().capacity == 600


def test_capture_is_one_request_on_a_socket_backend() -> None:
    """T-822: a snapshot of every region is one read_many, not one read per region."""
    from retro_ai_lab.emulator.bizhawk.simulator import BizHawkSimulator

    with BizHawkSimulator() as sim:
        service = ConnectionService(default_registry(), EventBus())
        service.connect("bizhawk", port=sim.port, call_timeout_s=2.0, timeout_s=2.0)
        cap = MemoryCapture(service)
        before = len(sim.requests)
        snapshot = cap.capture(7)
        assert len(sim.requests) == before + 1 and sim.requests[-1]["cmd"] == "read"
        assert snapshot.frame == 7 and snapshot.get("ram") == sim.game.ram
        service.disconnect()
