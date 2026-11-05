"""T-202: byte diffs and change statistics."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT, RAM_SIZE
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.observation.diff import (
    ByteChange,
    ByteStats,
    ChangeTracker,
    SnapshotDiff,
    diff_snapshots,
)


def snap(frame: int, ram: bytes, **more: bytes) -> MemorySnapshot:
    return MemorySnapshot(frame, {"ram": ram, **more})


# ------------------------------------------------------------- diffing


def test_identical_snapshots_give_empty_diff() -> None:
    d = diff_snapshots(snap(0, b"abc"), snap(1, b"abc"))
    assert d.is_empty and d.changed_count == 0
    assert (d.before_frame, d.after_frame) == (0, 1)
    assert d.offsets("ram") == ()
    assert d.by_region() == {}


def test_diff_lists_every_changed_byte_in_order() -> None:
    d = diff_snapshots(
        snap(3, b"\x00\x05\xff\x10", x=b"\x01"), snap(4, b"\x00\x06\x00\x10", x=b"\x00")
    )
    assert d.changes == (
        ByteChange("ram", 1, 5, 6),
        ByteChange("ram", 2, 255, 0),
        ByteChange("x", 0, 1, 0),
    )
    assert d.changes[0].delta == 1
    assert d.changes[1].delta == -255
    assert d.offsets("ram") == (1, 2)
    assert d.by_region() == {"ram": d.changes[:2], "x": d.changes[2:]}


def test_diff_rejects_mismatched_regions_or_sizes() -> None:
    with pytest.raises(ConfigurationError, match="different regions"):
        diff_snapshots(snap(0, b"a"), snap(1, b"a", x=b"b"))
    with pytest.raises(ConfigurationError, match="changed size"):
        diff_snapshots(snap(0, b"ab"), snap(1, b"abc"))


@settings(max_examples=40)
@given(a=st.binary(min_size=1, max_size=64), data=st.data())
def test_diff_matches_a_naive_comparison(a: bytes, data: st.DataObject) -> None:
    b = data.draw(st.binary(min_size=len(a), max_size=len(a)))
    d = diff_snapshots(snap(0, a), snap(1, b))
    expected = tuple(ByteChange("ram", i, a[i], b[i]) for i in range(len(a)) if a[i] != b[i])
    assert d.changes == expected


def test_snapshot_diff_is_immutable() -> None:
    d = SnapshotDiff(0, 1, ())
    with pytest.raises(AttributeError):
        d.changes = (ByteChange("ram", 0, 0, 1),)  # type: ignore[misc]


# ------------------------------------------------------------- tracker


def test_tracker_seeds_on_first_snapshot_and_diffs_after() -> None:
    t = ChangeTracker()
    assert t.observe(snap(0, b"\x05\x00")) is None
    assert t.observed == 1 and t.transitions == 0
    st0 = t.stats("ram", 0)
    assert (st0.samples, st0.changes, st0.minimum, st0.maximum, st0.last) == (0, 0, 5, 5, 5)
    d = t.observe(snap(1, b"\x06\x00"))
    assert d is not None and d.changes == (ByteChange("ram", 0, 5, 6),)
    assert t.transitions == 1
    st0 = t.stats("ram", 0)
    assert (st0.samples, st0.changes, st0.ups, st0.downs) == (1, 1, 1, 0)
    assert st0.monotonic_up and not st0.monotonic_down and not st0.constant
    assert (st0.minimum, st0.maximum, st0.last, st0.distinct) == (5, 6, 6, {5, 6})
    st1 = t.stats("ram", 1)
    assert st1.constant and st1.change_rate == 0.0
    assert st1.to_dict()["distinct"] == 1


def test_tracker_counts_ups_downs_and_rates() -> None:
    t = ChangeTracker()
    for v in (10, 11, 12, 5, 5, 6):
        t.observe(snap(0, bytes([v])))
    st = t.stats("ram", 0)
    assert st.samples == 5 and st.changes == 4
    assert st.ups == 3 and st.downs == 1
    assert st.change_rate == 0.8
    assert not st.monotonic_up and not st.monotonic_down
    assert (st.minimum, st.maximum) == (5, 12)
    assert st.to_dict()["change_rate"] == 0.8


def test_tracker_unknown_byte_and_bad_input() -> None:
    t = ChangeTracker()
    with pytest.raises(ConfigurationError):
        t.stats("ram", 0)
    with pytest.raises(ConfigurationError):
        t.observe("snap")  # type: ignore[arg-type]


def test_tracker_reset() -> None:
    t = ChangeTracker()
    t.observe(snap(0, b"\x00"))
    t.observe(snap(1, b"\x01"))
    t.reset()
    assert t.observed == 0 and t.all_stats() == {}
    assert t.observe(snap(0, b"\x00")) is None


def test_classify_separates_constant_noise_and_state() -> None:
    t = ChangeTracker()
    # byte0 constant, byte1 noise (changes every frame), byte2 state (changes sometimes)
    for i in range(20):
        t.observe(snap(i, bytes([7, i, i // 5])))
    groups = t.classify()
    assert groups["constant"] == (("ram", 0),)
    assert groups["noise"] == (("ram", 1),)
    assert groups["state"] == (("ram", 2),)
    assert groups["unknown"] == ()


def test_classify_reports_unknown_before_enough_transitions() -> None:
    t = ChangeTracker()
    t.observe(snap(0, b"\x00"))
    assert t.classify()["unknown"] == (("ram", 0),)
    t.observe(snap(1, b"\x01"))
    assert t.classify(min_transitions=1)["unknown"] == ()
    assert t.classify()["unknown"] == (("ram", 0),)


@pytest.mark.parametrize("rate", [0, -0.1, 1.5])
def test_classify_rejects_bad_noise_rate(rate: float) -> None:
    with pytest.raises(ConfigurationError):
        ChangeTracker().classify(noise_rate=rate)


def test_changed_in_all_intersects_diffs() -> None:
    t = ChangeTracker()
    d1 = diff_snapshots(snap(0, b"\x00\x00\x00"), snap(1, b"\x01\x01\x00"))
    d2 = diff_snapshots(snap(1, b"\x01\x01\x00"), snap(2, b"\x02\x01\x01"))
    assert t.changed_in_all([d1, d2]) == {("ram", 0)}
    assert t.changed_in_all([]) == set()
    assert t.changed_in_all([d1, SnapshotDiff(0, 1, ())]) == set()


# ------------------------------------------------- on the real toy game


def test_tracker_finds_the_toy_racers_noise_timer_and_state_bytes() -> None:
    """The point of the module: tell signal from noise on a game it has never seen."""
    game = ToyRacer(seed=3)
    t = ChangeTracker()
    t.observe(MemorySnapshot(0, {"ram": game.ram}))
    for i in range(1, 120):
        game.step(("up",) if i % 40 < 30 else ("down", "left"))
        t.observe(MemorySnapshot(i, {"ram": game.ram}))
    groups = t.classify()
    noise = set(groups["noise"])
    state = set(groups["state"])
    constant = set(groups["constant"])
    # The four noise bytes and the free-running timer change every frame.
    for off in range(LAYOUT.noise, LAYOUT.noise + 4):
        assert ("ram", off) in noise
    assert ("ram", LAYOUT.timer) in noise
    assert ("ram", LAYOUT.frame) in noise  # low byte of the frame counter
    # Header bytes never change.
    for off in range(LAYOUT.header, RAM_SIZE):
        assert ("ram", off) in constant
    # Real game state changes some of the time.
    for off in (LAYOUT.position, LAYOUT.speed, LAYOUT.lane, LAYOUT.score, LAYOUT.input):
        assert ("ram", off) in state, off
    # Score only ever goes up in this run; speed goes both ways.
    assert t.stats("ram", LAYOUT.score + 1).monotonic_up or t.stats("ram", LAYOUT.score).ups > 0
    assert not t.stats("ram", LAYOUT.speed).monotonic_up


def test_tracker_records_the_largest_hop_either_way() -> None:
    """T-911: the largest step a byte took, up or down, with roll-overs read as short hops."""
    t = ChangeTracker()
    for i, v in enumerate([10, 14, 14, 11, 250, 3, 3, 1]):
        t.observe(MemorySnapshot(i, {"ram": bytes([v])}))
    st = t.stats("ram", 0)
    # 10->14 is 4; 14->11 is 3; 11->250 is a 17 hop down past zero; 250->3 is 9 up; 3->1 is 2.
    assert st.max_hop == 17
    assert st.max_step == 239  # unchanged: the plain upward step (T-302 keeps its meaning)
    assert st.to_dict()["max_hop"] == 17
    assert ByteStats().max_hop == 0
