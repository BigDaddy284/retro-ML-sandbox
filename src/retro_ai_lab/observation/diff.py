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
        # Fast path on the common case of few changes: compare in chunks.
        for i in range(len(a)):
            if a[i] != b[i]:
                changes.append(ByteChange(name, i, a[i], b[i]))
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
        }


class ChangeTracker:
    """Accumulates per-byte statistics from a stream of snapshots."""

    def __init__(self) -> None:
        self._stats: dict[tuple[str, int], ByteStats] = {}
        self._previous: MemorySnapshot | None = None
        self._observed = 0

    @property
    def observed(self) -> int:
        """Snapshots fed so far."""
        return self._observed

    @property
    def transitions(self) -> int:
        return max(0, self._observed - 1)

    def reset(self) -> None:
        self._stats.clear()
        self._previous = None
        self._observed = 0

    def observe(self, snapshot: MemorySnapshot) -> SnapshotDiff | None:
        """Feed the next snapshot; returns the diff from the previous one, if any."""
        if not isinstance(snapshot, MemorySnapshot):
            raise ConfigurationError("ChangeTracker.observe needs a MemorySnapshot.")
        self._observed += 1
        if self._previous is None:
            self._seed(snapshot)
            self._previous = snapshot
            return None
        diff = diff_snapshots(self._previous, snapshot)
        changed = {(c.region, c.offset) for c in diff.changes}
        wrapped: set[tuple[str, int]] = set()
        for name, data in snapshot.regions.items():
            for offset, value in enumerate(data):
                st = self._stats[(name, offset)]
                st.samples += 1
                if (name, offset) in changed:
                    st.changes += 1
                    assert st.last is not None
                    delta = value - st.last
                    hop = min(abs(delta), 256 - abs(delta))
                    st.max_hop = max(st.max_hop, hop)
                    st.min_hop = hop if st.changes == 1 else min(st.min_hop, hop)
                    if hop <= SMALL_HOP:
                        st.small_hops += 1
                    if hop == 1:
                        st.unit_hops += 1
                    if delta > 0:
                        st.ups += 1
                        st.max_step = max(st.max_step, delta)
                        if delta == 1:
                            st.unit_steps += 1
                    else:
                        st.downs += 1
                        if delta < -128:
                            st.wraps += 1
                            wrapped.add((name, offset))
                            if delta + 256 == 1:
                                st.unit_steps += 1
                    if (name, offset - 1) in wrapped:
                        st.carry_ins += 1
                st.last = value
                st.minimum = min(st.minimum, value)
                st.maximum = max(st.maximum, value)
                st.distinct.add(value)
        self._previous = snapshot
        return diff

    def _seed(self, snapshot: MemorySnapshot) -> None:
        for name, data in snapshot.regions.items():
            for offset, value in enumerate(data):
                self._stats[(name, offset)] = ByteStats(
                    minimum=value, maximum=value, last=value, distinct={value}
                )

    # ------------------------------------------------------------ queries

    def stats(self, region: str, offset: int) -> ByteStats:
        try:
            return self._stats[(region, offset)]
        except KeyError:
            raise ConfigurationError(f"No statistics for {region}[{offset}].") from None

    def all_stats(self) -> Mapping[tuple[str, int], ByteStats]:
        return dict(self._stats)

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
        for key, st in self._stats.items():
            if st.samples < min_transitions:
                groups["unknown"].append(key)
            elif st.constant:
                groups["constant"].append(key)
            elif st.change_rate >= noise_rate:
                groups["noise"].append(key)
            else:
                groups["state"].append(key)
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
