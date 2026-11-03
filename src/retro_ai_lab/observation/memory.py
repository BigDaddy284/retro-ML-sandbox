"""Memory snapshots: what the emulator's RAM looked like at one moment.

A :class:`MemorySnapshot` is immutable and compares by value, so later layers
(change detection, variable discovery) can diff and hash them freely.
:class:`MemoryCapture` reads every readable region through the connection
service; :class:`SnapshotRing` keeps the most recent ones in memory.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from retro_ai_lab.core import ConfigurationError, MemoryRegion
from retro_ai_lab.emulator.base import EmulatorAdapter

if TYPE_CHECKING:
    from retro_ai_lab.services import ConnectionService


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """RAM contents of one or more regions at frame ``frame``."""

    frame: int
    regions: Mapping[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise ConfigurationError(
                f"Snapshot frame must be a non-negative integer, got {self.frame!r}."
            )
        if not isinstance(self.regions, Mapping) or not self.regions:
            raise ConfigurationError("Snapshot needs at least one memory region.")
        clean: dict[str, bytes] = {}
        for name, data in self.regions.items():
            if not isinstance(name, str) or not name.isidentifier() or not name.islower():
                raise ConfigurationError(f"Bad region name {name!r} in snapshot.")
            if not isinstance(data, bytes):
                raise ConfigurationError(f"Region {name!r} data must be bytes.")
            clean[name] = data
        object.__setattr__(self, "regions", MappingProxyType(clean))

    # ------------------------------------------------------------ access

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple(self.regions)

    @property
    def size(self) -> int:
        return sum(len(d) for d in self.regions.values())

    def get(self, region: str, offset: int = 0, length: int | None = None) -> bytes:
        try:
            data = self.regions[region]
        except KeyError:
            raise ConfigurationError(f"No region {region!r} in snapshot.") from None
        if length is None:
            length = len(data) - offset
        if offset < 0 or length < 0 or offset + length > len(data):
            raise ConfigurationError(
                f"Read of {length} bytes at {offset} is outside region {region!r} "
                f"({len(data)} bytes)."
            )
        return data[offset : offset + length]

    def byte(self, region: str, offset: int) -> int:
        return self.get(region, offset, 1)[0]

    def __hash__(self) -> int:
        return hash((self.frame, tuple(self.regions.items())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemorySnapshot):
            return NotImplemented
        return self.frame == other.frame and dict(self.regions) == dict(other.regions)

    # ---------------------------------------------------------- storage

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "regions": {n: d.hex() for n, d in self.regions.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemorySnapshot:
        if not isinstance(data, Mapping) or set(data) != {"frame", "regions"}:
            raise ConfigurationError("A snapshot dict needs exactly 'frame' and 'regions'.")
        regions = data["regions"]
        if not isinstance(regions, Mapping):
            raise ConfigurationError("Snapshot 'regions' must be a mapping.")
        decoded: dict[str, bytes] = {}
        for name, hexed in regions.items():
            if not isinstance(hexed, str):
                raise ConfigurationError(f"Region {name!r} must be a hex string.")
            try:
                decoded[name] = bytes.fromhex(hexed)
            except ValueError as exc:
                raise ConfigurationError(f"Region {name!r} is not valid hex.") from exc
        return cls(data["frame"], decoded)


class MemoryCapture:
    """Reads memory regions from the live adapter into snapshots."""

    def __init__(self, connection: ConnectionService, regions: Sequence[str] | None = None):
        self._connection = connection
        self._wanted = tuple(regions) if regions is not None else None
        self._captures = 0
        self._plan: tuple[tuple[str, int, int], ...] | None = None

    @property
    def captures(self) -> int:
        return self._captures

    def available_regions(self) -> tuple[MemoryRegion, ...]:
        return tuple(self._connection.call(lambda a: a.memory_regions()))

    def capture(self, frame: int) -> MemorySnapshot:
        """Snapshot the wanted regions, tagging the result with ``frame``."""
        snapshot = self._connection.call(lambda a: self._read_all(a, frame))
        self._captures += 1
        return snapshot

    def reads(self) -> tuple[tuple[str, int, int], ...]:
        """The ``(region, offset, length)`` reads one snapshot needs (T-823).

        Hand them to ``EmulatorAdapter.step(reads=...)`` and give the bytes
        back to :meth:`from_reads`, so a frame step and its snapshot are one
        request. The plan is made once per capture object: an emulator's
        regions do not change while it is connected.
        """
        if self._plan is None:
            self._plan = self._connection.call(self._make_plan)
        return self._plan

    def from_reads(self, frame: int, chunks: Sequence[bytes]) -> MemorySnapshot:
        """A snapshot from bytes fetched elsewhere, in :meth:`reads` order."""
        names = [name for name, _, _ in self.reads()]
        if len(chunks) != len(names):
            raise ConfigurationError(
                f"Got {len(chunks)} region(s) of bytes for {len(names)} planned."
            )
        self._captures += 1
        return MemorySnapshot(frame, dict(zip(names, chunks, strict=True)))

    def _make_plan(self, adapter: EmulatorAdapter) -> tuple[tuple[str, int, int], ...]:
        regions = {r.name: r for r in adapter.memory_regions()}
        names = self._wanted if self._wanted is not None else tuple(regions)
        unknown = [n for n in names if n not in regions]
        if unknown:
            raise ConfigurationError(
                f"Unknown memory regions {unknown}; available: {sorted(regions)}."
            )
        if not names:
            raise ConfigurationError("This emulator exposes no memory regions.")
        return tuple((n, 0, regions[n].size) for n in names)

    def _read_all(self, adapter: EmulatorAdapter, frame: int) -> MemorySnapshot:
        if self._plan is None:
            self._plan = self._make_plan(adapter)
        chunks = adapter.read_many(list(self._plan))  # one request
        names = [n for n, _, _ in self._plan]
        return MemorySnapshot(frame, dict(zip(names, chunks, strict=True)))


class SnapshotRing:
    """The most recent ``capacity`` snapshots, oldest first."""

    def __init__(self, capacity: int = 600) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ConfigurationError("Ring capacity must be a positive integer.")
        self._items: deque[MemorySnapshot] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._items.maxlen or 0

    def append(self, snapshot: MemorySnapshot) -> None:
        if not isinstance(snapshot, MemorySnapshot):
            raise ConfigurationError("Only MemorySnapshot objects can be stored.")
        self._items.append(snapshot)

    @property
    def latest(self) -> MemorySnapshot | None:
        return self._items[-1] if self._items else None

    def previous(self, n: int = 1) -> MemorySnapshot | None:
        """The snapshot ``n`` places before the latest (``n=0`` is the latest)."""
        if n < 0 or n >= len(self._items):
            return None
        return self._items[-1 - n]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[MemorySnapshot]:
        return iter(self._items)

    def __getitem__(self, index: int) -> MemorySnapshot:
        return self._items[index]


__all__ = ["MemoryCapture", "MemorySnapshot", "SnapshotRing"]
