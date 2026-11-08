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
    SMALL_HOP,
    SMOOTH_SHARE,
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


def test_tracker_skips_an_expected_jump() -> None:
    """T-935: the snapshot after a restore is moved past, not counted."""
    t = ChangeTracker()
    t.expect_jump()  # nothing to skip before the first snapshot
    assert not t.jump_expected
    for v in (3, 4, 5):
        t.observe(snap(0, bytes([v])))
    st = t.stats("ram", 0)
    assert (st.samples, st.changes, st.ups, st.downs, st.last) == (2, 2, 2, 0, 5)
    t.expect_jump()
    assert t.jump_expected
    d = t.observe(snap(3, b"\x03"))  # put back to the start
    assert d is not None and d.changes == (ByteChange("ram", 0, 5, 3),)
    assert not t.jump_expected
    st = t.stats("ram", 0)  # a fresh reading each time (T-945): the counts live in arrays
    assert (st.samples, st.changes, st.ups, st.downs, st.last) == (2, 2, 2, 0, 3)
    assert st.monotonic_up and t.transitions == 2 and t.observed == 4
    t.observe(snap(4, b"\x04"))  # counted again, from the restored value
    st = t.stats("ram", 0)
    assert (st.samples, st.changes, st.ups, st.downs, st.last) == (3, 3, 3, 0, 4)
    assert t.transitions == 3
    t.expect_jump()
    t.reset()
    assert not t.jump_expected


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


def test_smooth_takes_three_small_hops_in_four() -> None:
    """T-918: a thing that comes and goes makes one big hop each way per visit."""
    smooth = ChangeTracker()
    rough = ChangeTracker()
    for i, (a, b) in enumerate(
        [(0, 0), (200, 200), (197, 100), (194, 194), (0, 0)]  # 3 of 4 small; 2 of 4 small
    ):
        smooth.observe(MemorySnapshot(i, {"ram": bytes([a])}))
        rough.observe(MemorySnapshot(i, {"ram": bytes([b])}))
    assert smooth.stats("ram", 0).small_hops == 2 and smooth.stats("ram", 0).changes == 4
    assert not smooth.stats("ram", 0).smooth  # 2 of 4 is under three in four
    for i, value in enumerate((3, 6, 9, 12), start=5):
        smooth.observe(MemorySnapshot(i, {"ram": bytes([value])}))
    assert smooth.stats("ram", 0).smooth  # 6 of 8: exactly three in four
    assert not rough.stats("ram", 0).smooth
    assert SMOOTH_SHARE == 0.75


def test_tracker_records_the_smallest_hop_too() -> None:
    """T-921: the smallest step either way, so a byte always stepping by four shows it."""
    t = ChangeTracker()
    for i, v in enumerate([10, 14, 14, 11, 250, 3, 3, 1]):
        t.observe(MemorySnapshot(i, {"ram": bytes([v])}))
    st = t.stats("ram", 0)
    assert st.min_hop == 2 and st.max_hop == 17 and st.to_dict()["min_hop"] == 2
    assert ByteStats().min_hop == 0
    steady = ChangeTracker()
    for i in range(100):
        steady.observe(MemorySnapshot(i, {"ram": bytes([(i * 4) & 0xFF])}))
    assert steady.stats("ram", 0).min_hop == steady.stats("ram", 0).max_hop == 4


def test_tracker_counts_repeated_hops() -> None:
    """T-924: how often a hop was the same as the change before, and which hop that was."""
    t = ChangeTracker()
    for i, v in enumerate([10, 14, 18, 18, 22, 100, 104, 108, 109]):
        t.observe(MemorySnapshot(i, {"ram": bytes([v])}))
    st = t.stats("ram", 0)
    # hops: 4, 4, 4, 78, 4, 4, 1 -> the same as the one before three times
    assert st.changes == 7 and st.same_hops == 3 and st.usual_hop == 4 and st.last_hop == 1
    d = st.to_dict()
    assert d["same_hops"] == 3 and d["usual_hop"] == 4
    assert ByteStats().same_hops == 0 and ByteStats().usual_hop == 0
    once = ChangeTracker()
    for i, v in enumerate([3, 9]):
        once.observe(MemorySnapshot(i, {"ram": bytes([v])}))
    assert once.stats("ram", 0).same_hops == 0 and once.stats("ram", 0).last_hop == 6


def test_tracker_counts_small_and_unit_hops() -> None:
    """T-914: how many changes were small hops (at most SMALL_HOP) and exact ones."""
    t = ChangeTracker()
    for i, v in enumerate([10, 11, 11, 14, 60, 61, 0, 255]):
        t.observe(MemorySnapshot(i, {"ram": bytes([v])}))
    st = t.stats("ram", 0)
    # +1, +3, +46, +1, -61, -1 (a roll-over, read as one step down)
    assert st.changes == 6 and st.small_hops == 4 and st.unit_hops == 3
    assert st.to_dict()["small_hops"] == 4 and st.to_dict()["unit_hops"] == 3
    assert ByteStats().small_hops == 0 and ByteStats().unit_hops == 0
    assert SMALL_HOP == 8


# ---------------------------------------------------- the arrays agree (T-945)


def _count_by_hand(streams: list[bytes], jumps: set[int]) -> dict[int, ByteStats]:
    """The tracker's counting as T-202..T-935 wrote it, one byte at a time, for one region."""
    stats = {
        i: ByteStats(minimum=v, maximum=v, last=v, distinct={v}) for i, v in enumerate(streams[0])
    }
    for n, data in enumerate(streams[1:], start=1):
        wrapped: set[int] = set()
        for offset, value in enumerate(data):
            st = stats[offset]
            if n not in jumps:
                st.samples += 1
                assert st.last is not None
                if value != st.last:
                    st.changes += 1
                    delta = value - st.last
                    hop = min(abs(delta), 256 - abs(delta))
                    st.max_hop = max(st.max_hop, hop)
                    st.min_hop = hop if st.changes == 1 else min(st.min_hop, hop)
                    if hop <= SMALL_HOP:
                        st.small_hops += 1
                    if hop == 1:
                        st.unit_hops += 1
                    if st.changes > 1 and hop == st.last_hop:
                        st.same_hops += 1
                        st.usual_hop = hop
                    st.last_hop = hop
                    if delta > 0:
                        st.ups += 1
                        st.max_step = max(st.max_step, delta)
                        if delta == 1:
                            st.unit_steps += 1
                    else:
                        st.downs += 1
                        if delta < -128:
                            st.wraps += 1
                            wrapped.add(offset)
                            if delta + 256 == 1:
                                st.unit_steps += 1
                    if offset - 1 in wrapped:
                        st.carry_ins += 1
            st.last = value
            st.minimum = min(st.minimum, value)
            st.maximum = max(st.maximum, value)
            st.distinct.add(value)
    return stats


@settings(max_examples=60, deadline=None)
@given(
    width=st.integers(min_value=1, max_value=6),
    steps=st.integers(min_value=1, max_value=12),
    data=st.data(),
)
def test_the_arrays_count_exactly_as_the_byte_loop_did(
    width: int, steps: int, data: st.DataObject
) -> None:
    """T-945: the tracker counts every byte of a region in numpy; every statistic of every
    byte, restores included, must equal what the one-byte-at-a-time loop counted."""
    small = st.integers(min_value=-3, max_value=3)
    streams = [data.draw(st.binary(min_size=width, max_size=width))]
    jumps: set[int] = set()
    for n in range(1, steps + 1):
        kind = data.draw(st.sampled_from(["walk", "wrap", "random", "same", "jump"]))
        prev = streams[-1]
        if kind == "walk":
            nxt = bytes((b + data.draw(small)) % 256 for b in prev)
        elif kind == "wrap":  # a counter rolling over, with the byte above carrying
            nxt = bytes(
                [(prev[0] + 1) % 256, *((b + (1 if prev[0] == 255 else 0)) % 256 for b in prev[1:])]
            )
        elif kind == "random":
            nxt = data.draw(st.binary(min_size=width, max_size=width))
        elif kind == "same":
            nxt = prev
        else:
            nxt = data.draw(st.binary(min_size=width, max_size=width))
            jumps.add(n)
        streams.append(nxt)
    t = ChangeTracker()
    for n, stream in enumerate(streams):
        if n in jumps:
            t.expect_jump()
        t.observe(snap(n, stream))
    expected = _count_by_hand(streams, jumps)
    for offset in range(width):
        assert t.stats("ram", offset) == expected[offset], f"byte {offset}"
    assert t.transitions == steps - len(jumps)
    assert set(t.all_stats()) == {("ram", i) for i in range(width)}
    assert t.all_stats()[("ram", 0)] == expected[0]
