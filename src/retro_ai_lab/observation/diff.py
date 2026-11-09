"""Memory change detection: which bytes changed between two snapshots.

The Game Understanding Engine (Phase 3) builds on three questions this module
answers cheaply and exactly:

* which bytes changed between two frames (:func:`diff_snapshots`);
* how each byte has behaved over a run (:class:`ChangeTracker`): how often it
  changes, whether it only counts up/down, and its value range;
* which bytes look like *noise* (change nearly every frame) versus *state*
  (change sometimes) versus *constants* (never).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot


@dataclass(frozen=True, slots=True)
class ByteChange:
    region: str
    offset: int
    old: int
    new: int

    @property
    def delta(self) -> int:
        return self.new - self.old


@dataclass(frozen=True, slots=True)
class SnapshotDiff:
    """Byte-level differences from ``before`` to ``after``."""

    before_frame: int
    after_frame: int
    changes: tuple[ByteChange, ...]

    @property
    def changed_count(self) -> int:
        return len(self.changes)

    @property
    def is_empty(self) -> bool:
        return not self.changes

    def offsets(self, region: str) -> tuple[int, ...]:
        return tuple(c.offset for c in self.changes if c.region == region)

    def by_region(self) -> dict[str, tuple[ByteChange, ...]]:
        out: dict[str, list[ByteChange]] = {}
        for c in self.changes:
            out.setdefault(c.region, []).append(c)
        return {k: tuple(v) for k, v in out.items()}


def diff_snapshots(before: MemorySnapshot, after: MemorySnapshot) -> SnapshotDiff:
    """Exact byte diff. Both snapshots must cover the same regions and sizes."""
    if before.region_names != after.region_names:
        raise ConfigurationError(
            f"Snapshots cover different regions: {before.region_names} vs {after.region_names}."
        )
    changes: list[ByteChange] = []
    for name in before.region_names:
        a = before.regions[name]
        b = after.regions[name]
        if len(a) != len(b):
            raise ConfigurationError(
                f"Region {name!r} changed size between snapshots ({len(a)} vs {len(b)})."
            )
        if a == b:
            continue
        # The bytes that differ are found in one array comparison (T-945); only
        # those become objects.
        differ = np.flatnonzero(
            np.frombuffer(a, dtype=np.uint8) != np.frombuffer(b, dtype=np.uint8)
        )
        changes.extend(ByteChange(name, i, a[i], b[i]) for i in differ.tolist())
    return SnapshotDiff(before.frame, after.frame, tuple(changes))


SMALL_HOP = 8  # a thing on screen moves by at most this many between observations (T-911)
# The share of a byte's changes that must be small hops for it to be "smooth" (T-918):
# a thing that appears and is cleared makes one big hop each way per visit, so a
# short visit of six small steps is still three small hops in four.
SMOOTH_SHARE = 0.75


@dataclass(slots=True)
class ByteStats:
    """How one byte has behaved across every observed transition."""

    samples: int = 0  # transitions observed
    changes: int = 0  # transitions where the value changed
    ups: int = 0  # changed and increased
    downs: int = 0  # changed and decreased
    minimum: int = 255
    maximum: int = 0
    last: int | None = None
    distinct: set[int] = field(default_factory=set)
    wraps: int = 0  # went down by more than 128 (255 -> 0 style roll-over)
    carry_ins: int = 0  # changed in the same transition as the byte below wrapped
    unit_steps: int = 0  # changed by exactly +1 (counting a wrap as +1)
    max_step: int = 0  # largest upward step seen
    max_hop: int = 0  # largest step either way, a roll-over read as the short way round (T-911)
    min_hop: int = 0  # smallest step either way, 0 until the byte has changed (T-921)
    small_hops: int = 0  # changes by at most SMALL_HOP either way (T-914)
    unit_hops: int = 0  # changes by exactly one either way, roll-overs included (T-914)
    last_hop: int = 0  # the hop of the latest change (T-924)
    same_hops: int = 0  # changes whose hop equalled the previous change's hop (T-924)
    usual_hop: int = 0  # the hop last seen in two changes running (T-924)

    @property
    def change_rate(self) -> float:
        return self.changes / self.samples if self.samples else 0.0

    @property
    def monotonic_up(self) -> bool:
        return self.changes > 0 and self.downs == 0

    @property
    def monotonic_down(self) -> bool:
        return self.changes > 0 and self.ups == 0

    @property
    def constant(self) -> bool:
        return self.samples > 0 and self.changes == 0

    @property
    def resets(self) -> int:
        """Drops that were not roll-overs (a value being set back, e.g. on restart)."""
        return self.downs - self.wraps

    @property
    def monotonic_mod_wrap(self) -> bool:
        """Only ever went up, apart from roll-overs (counter-like)."""
        return self.changes > 0 and self.downs == self.wraps

    @property
    def smooth(self) -> bool:
        """Most changes were small hops: something moving, not noise (T-914, T-918)."""
        return self.changes > 0 and self.small_hops >= SMOOTH_SHARE * self.changes

    @property
    def mostly_up(self) -> bool:
        """Goes up apart from roll-overs and the odd reset (score-like)."""
        return self.ups > 0 and self.resets <= max(2.0, self.ups * 0.15)

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "changes": self.changes,
            "ups": self.ups,
            "downs": self.downs,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "distinct": len(self.distinct),
            "change_rate": round(self.change_rate, 4),
            "wraps": self.wraps,
            "carry_ins": self.carry_ins,
            "unit_steps": self.unit_steps,
            "max_step": self.max_step,
            "max_hop": self.max_hop,
            "min_hop": self.min_hop,
            "small_hops": self.small_hops,
            "unit_hops": self.unit_hops,
            "same_hops": self.same_hops,
            "usual_hop": self.usual_hop,
        }


Counts = NDArray[np.int64]


def _values(data: bytes) -> Counts:
    return np.frombuffer(data, dtype=np.uint8).astype(np.int64)


class _RegionStats:
    """Every byte's statistics of one region, one array per field (T-945).

    The user's log had the app's step at 140 ms against 30 ms of emulator; the
    tracker walked every byte of a 2 KB region in Python on every step, about
    5 ms of pure work at that scale and more under the interpreter lock. The
    same counting over arrays costs a fraction of a millisecond, and
    :class:`ByteStats` is built from the arrays when someone asks for a byte.
    """

    __slots__ = (
        "carry_ins",
        "changes",
        "downs",
        "last",
        "last_hop",
        "max_hop",
        "max_step",
        "maximum",
        "min_hop",
        "minimum",
        "same_hops",
        "samples",
        "seen",
        "size",
        "small_hops",
        "unit_hops",
        "unit_steps",
        "ups",
        "usual_hop",
        "wraps",
    )
    size: int
    samples: Counts
    changes: Counts
    ups: Counts
    downs: Counts
    minimum: Counts
    maximum: Counts
    last: Counts
    seen: NDArray[np.bool_]
    wraps: Counts
    carry_ins: Counts
    unit_steps: Counts
    max_step: Counts
    max_hop: Counts
    min_hop: Counts
    small_hops: Counts
    unit_hops: Counts
    last_hop: Counts
    same_hops: Counts
    usual_hop: Counts

    def __init__(self, data: bytes) -> None:
        values = _values(data)
        self.size = len(data)
        zeros = np.zeros(self.size, dtype=np.int64)
        self.samples = zeros.copy()
        self.changes = zeros.copy()
        self.ups = zeros.copy()
        self.downs = zeros.copy()
        self.wraps = zeros.copy()
        self.carry_ins = zeros.copy()
        self.unit_steps = zeros.copy()
        self.max_step = zeros.copy()
        self.max_hop = zeros.copy()
        self.min_hop = zeros.copy()
        self.small_hops = zeros.copy()
        self.unit_hops = zeros.copy()
        self.last_hop = zeros.copy()
        self.same_hops = zeros.copy()
        self.usual_hop = zeros.copy()
        self.minimum = values.copy()
        self.maximum = values.copy()
        self.last = values
        self.seen = np.zeros((self.size, 256), dtype=bool)  # the distinct values of each byte
        self._note(values)

    def _note(self, values: Counts) -> None:
        self.last = values
        np.minimum(self.minimum, values, out=self.minimum)
        np.maximum(self.maximum, values, out=self.maximum)
        self.seen[np.arange(self.size), values] = True

    def move_on(self, data: bytes) -> None:
        """Take the values without counting a transition (a restore, T-935)."""
        self._note(_values(data))

    def count(self, data: bytes) -> None:
        """Count the transition from the last values to ``data``."""
        values = _values(data)
        delta = values - self.last
        changed = delta != 0
        away = np.abs(delta)
        hop = np.minimum(away, 256 - away)
        self.samples += 1
        self.changes += changed
        first = changed & (self.changes == 1)
        later = changed & (self.changes > 1)
        np.maximum(self.max_hop, np.where(changed, hop, 0), out=self.max_hop)
        self.min_hop = np.where(
            first, hop, np.where(later, np.minimum(self.min_hop, hop), self.min_hop)
        )
        self.small_hops += changed & (hop <= SMALL_HOP)
        self.unit_hops += changed & (hop == 1)
        same = later & (hop == self.last_hop)
        self.same_hops += same
        self.usual_hop = np.where(same, hop, self.usual_hop)
        self.last_hop = np.where(changed, hop, self.last_hop)
        up = delta > 0
        self.ups += up
        np.maximum(self.max_step, np.where(up, delta, 0), out=self.max_step)
        down = delta < 0
        self.downs += down
        wrapped = delta < -128  # 255 -> 0 style roll-over
        self.wraps += wrapped
        self.unit_steps += (delta == 1) | (delta == -255)  # +1, a roll-over counted as +1
        carry = np.zeros(self.size, dtype=bool)
        carry[1:] = wrapped[:-1]  # the byte below rolled over in the same transition
        self.carry_ins += changed & carry
        self._note(values)

    def all_stats(self) -> list[ByteStats]:
        """Every byte's statistics, built from the arrays in one pass (T-957).

        The same objects :meth:`stats` builds one at a time; the review asked
        for all 2,048 at once and paid 20 ms in per-byte numpy calls.
        """
        rows, cols = np.nonzero(self.seen)
        starts = np.searchsorted(rows, np.arange(self.size + 1))
        values = cols.tolist()
        distinct = [set(values[starts[i] : starts[i + 1]]) for i in range(self.size)]
        columns = zip(
            self.samples.tolist(),
            self.changes.tolist(),
            self.ups.tolist(),
            self.downs.tolist(),
            self.minimum.tolist(),
            self.maximum.tolist(),
            self.last.tolist(),
            self.wraps.tolist(),
            self.carry_ins.tolist(),
            self.unit_steps.tolist(),
            self.max_step.tolist(),
            self.max_hop.tolist(),
            self.min_hop.tolist(),
            self.small_hops.tolist(),
            self.unit_hops.tolist(),
            self.last_hop.tolist(),
            self.same_hops.tolist(),
            self.usual_hop.tolist(),
            distinct,
            strict=True,
        )
        return [
            ByteStats(
                samples=samples,
                changes=changes,
                ups=ups,
                downs=downs,
                minimum=minimum,
                maximum=maximum,
                last=last,
                distinct=seen,
                wraps=wraps,
                carry_ins=carry_ins,
                unit_steps=unit_steps,
                max_step=max_step,
                max_hop=max_hop,
                min_hop=min_hop,
                small_hops=small_hops,
                unit_hops=unit_hops,
                last_hop=last_hop,
                same_hops=same_hops,
                usual_hop=usual_hop,
            )
            for (
                samples,
                changes,
                ups,
                downs,
                minimum,
                maximum,
                last,
                wraps,
                carry_ins,
                unit_steps,
                max_step,
                max_hop,
                min_hop,
                small_hops,
                unit_hops,
                last_hop,
                same_hops,
                usual_hop,
                seen,
            ) in columns
        ]

    def stats(self, offset: int) -> ByteStats:
        return ByteStats(
            samples=int(self.samples[offset]),
            changes=int(self.changes[offset]),
            ups=int(self.ups[offset]),
            downs=int(self.downs[offset]),
            minimum=int(self.minimum[offset]),
            maximum=int(self.maximum[offset]),
            last=int(self.last[offset]),
            distinct=set(np.flatnonzero(self.seen[offset]).tolist()),
            wraps=int(self.wraps[offset]),
            carry_ins=int(self.carry_ins[offset]),
            unit_steps=int(self.unit_steps[offset]),
            max_step=int(self.max_step[offset]),
            max_hop=int(self.max_hop[offset]),
            min_hop=int(self.min_hop[offset]),
            small_hops=int(self.small_hops[offset]),
            unit_hops=int(self.unit_hops[offset]),
            last_hop=int(self.last_hop[offset]),
            same_hops=int(self.same_hops[offset]),
            usual_hop=int(self.usual_hop[offset]),
        )


class ChangeTracker:
    """Accumulates per-byte statistics from a stream of snapshots."""

    def __init__(self) -> None:
        self._regions: dict[str, _RegionStats] = {}
        self._previous: MemorySnapshot | None = None
        self._observed = 0
        self._jumps = 0  # snapshots moved past without counting (restores, T-935)
        self._jump_expected = False

    @property
    def observed(self) -> int:
        """Snapshots fed so far."""
        return self._observed

    @property
    def transitions(self) -> int:
        """Transitions counted in the statistics (an expected jump is not one)."""
        return max(0, self._observed - 1 - self._jumps)

    @property
    def jump_expected(self) -> bool:
        return self._jump_expected

    def reset(self) -> None:
        self._regions.clear()
        self._previous = None
        self._observed = 0
        self._jumps = 0
        self._jump_expected = False

    def expect_jump(self) -> None:
        """The next snapshot follows a restore (T-935): move on to it without counting.

        Putting the game back to a saved state makes every byte the game had
        changed since jump back at once. That is the AI's doing, not the game's,
        so the next ``observe`` returns the diff but adds no sample, change, up,
        down, hop or wrap to any byte; the one after it counts from the restored
        values. Before the first snapshot there is nothing to skip.
        """
        self._jump_expected = self._previous is not None

    def observe(self, snapshot: MemorySnapshot) -> SnapshotDiff | None:
        """Feed the next snapshot; returns the diff from the previous one, if any."""
        if not isinstance(snapshot, MemorySnapshot):
            raise ConfigurationError("ChangeTracker.observe needs a MemorySnapshot.")
        self._observed += 1
        if self._previous is None:
            self._regions = {name: _RegionStats(data) for name, data in snapshot.regions.items()}
            self._previous = snapshot
            return None
        diff = diff_snapshots(self._previous, snapshot)  # checks the regions and sizes
        if self._jump_expected:
            self._jump_expected = False
            self._jumps += 1
            for name, data in snapshot.regions.items():
                self._regions[name].move_on(data)
        else:
            for name, data in snapshot.regions.items():
                self._regions[name].count(data)
        self._previous = snapshot
        return diff

    # ------------------------------------------------------------ queries

    def stats(self, region: str, offset: int) -> ByteStats:
        """This byte's statistics so far, built from the arrays (a fresh object each time)."""
        found = self._regions.get(region)
        if found is None or not 0 <= offset < found.size:
            raise ConfigurationError(f"No statistics for {region}[{offset}].")
        return found.stats(offset)

    def all_stats(self) -> Mapping[tuple[str, int], ByteStats]:
        return {
            (name, offset): stats
            for name, region in self._regions.items()
            for offset, stats in enumerate(region.all_stats())
        }

    def classify(
        self,
        *,
        noise_rate: float = 0.9,
        min_transitions: int = 2,
    ) -> dict[str, tuple[tuple[str, int], ...]]:
        """Split bytes into ``constant``, ``noise`` and ``state`` groups.

        A byte that changes in at least ``noise_rate`` of transitions is
        noise; one that never changes is constant; the rest is state and the
        interesting part for discovery. Needs ``min_transitions`` to say
        anything; before that every byte is reported as ``unknown``.
        """
        if not 0 < noise_rate <= 1:
            raise ConfigurationError("noise_rate must be in (0, 1].")
        groups: dict[str, list[tuple[str, int]]] = {
            "constant": [],
            "noise": [],
            "state": [],
            "unknown": [],
        }
        for name, region in self._regions.items():
            samples = region.samples
            changes = region.changes
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.where(samples > 0, changes / np.maximum(samples, 1), 0.0)
            unknown = samples < min_transitions
            constant = ~unknown & (samples > 0) & (changes == 0)
            noise = ~unknown & ~constant & (rate >= noise_rate)
            state = ~unknown & ~constant & ~noise
            for group, mask in (
                ("unknown", unknown),
                ("constant", constant),
                ("noise", noise),
                ("state", state),
            ):
                groups[group].extend((name, offset) for offset in np.flatnonzero(mask).tolist())
        return {k: tuple(v) for k, v in groups.items()}

    def changed_in_all(self, diffs: Iterable[SnapshotDiff]) -> set[tuple[str, int]]:
        """Bytes that changed in *every* given diff (an intersection helper)."""
        result: set[tuple[str, int]] | None = None
        for d in diffs:
            keys = {(c.region, c.offset) for c in d.changes}
            result = keys if result is None else result & keys
        return result or set()


__all__ = [
    "SMALL_HOP",
    "SMOOTH_SHARE",
    "ByteChange",
    "ByteStats",
    "ChangeTracker",
    "SnapshotDiff",
    "diff_snapshots",
]
