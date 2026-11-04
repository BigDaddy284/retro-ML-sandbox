"""A value search over all of the game's memory, like a cheat finder (T-884).

A person who knows a number on the screen (3 lives, 12 coins) can find the byte
that holds it: start a search over every byte the emulator shares, play a
little, then keep only the bytes that went up, went down, changed, stayed the
same or now equal a number. A few scans usually leave one byte.

Each scan compares the bytes with the snapshot of the previous scan, so "went
up" means "went up since I last pressed a scan button". Values are one byte
(0 to 255); the app only reads the game's memory, it never writes to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot

MAX_BYTE = 255
RANGE_TEXT = "Type a number from 0 to 255, like 12, or in hex like 0x0C."


class ScanKind(Enum):
    EXACT = "exact"
    UP = "up"
    DOWN = "down"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class Match:
    """One byte still in the search: where it is, its value now and at the scan before."""

    region: str
    offset: int
    now: int
    before: int


def parse_number(text: str) -> int:
    """Read ``12``, ``0x0C``, ``$0C`` or ``0Ch`` as a byte value."""
    raw = text.strip()
    digits, base = raw, 10
    lowered = raw.lower()
    if lowered.startswith("0x"):
        digits, base = raw[2:], 16
    elif raw.startswith("$"):
        digits, base = raw[1:], 16
    elif lowered.endswith("h") and len(raw) > 1:
        digits, base = raw[:-1], 16
    allowed = "0123456789abcdefABCDEF" if base == 16 else "0123456789"
    if not digits or any(c not in allowed for c in digits):
        raise ConfigurationError(f"That is not a number. {RANGE_TEXT}")
    value = int(digits, base)
    _check_byte(value)
    return value


def _check_byte(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_BYTE:
        raise ConfigurationError(f"A byte holds 0 to 255. {RANGE_TEXT}")


class ValueSearch:
    """The bytes still in the running, and their values at the last scan."""

    def __init__(self) -> None:
        self._offsets: dict[str, np.ndarray] = {}  # region -> candidate offsets
        self._values: dict[str, np.ndarray] = {}  # region -> their bytes at the last scan
        self._before: dict[str, np.ndarray] = {}  # region -> their bytes at the scan before
        self._sizes: dict[str, int] = {}
        self._frame: int | None = None

    @property
    def started(self) -> bool:
        return self._frame is not None

    @property
    def frame(self) -> int | None:
        """The frame of the snapshot the last scan used."""
        return self._frame

    @property
    def count(self) -> int:
        return int(sum(len(o) for o in self._offsets.values()))

    def reset(self) -> None:
        self._offsets, self._values, self._before = {}, {}, {}
        self._sizes, self._frame = {}, None

    def start(self, snapshot: MemorySnapshot, value: int | None = None) -> None:
        """Begin again with every byte, or only those equal to ``value``."""
        if value is not None:
            _check_byte(value)
        offsets: dict[str, np.ndarray] = {}
        values: dict[str, np.ndarray] = {}
        for region, data in snapshot.regions.items():
            current = np.frombuffer(data, dtype=np.uint8)
            if value is None:
                keep = np.arange(len(current), dtype=np.int64)
            else:
                keep = np.flatnonzero(current == value)
            offsets[region] = keep
            values[region] = current[keep].copy()
        self._offsets, self._values, self._before = offsets, values, values
        self._sizes = {r: len(d) for r, d in snapshot.regions.items()}
        self._frame = snapshot.frame

    def refine(self, snapshot: MemorySnapshot, kind: ScanKind, value: int | None = None) -> None:
        """Keep the bytes that pass ``kind`` against the previous scan."""
        if not self.started:
            raise ConfigurationError("Press New search first.")
        if kind is ScanKind.EXACT:
            if value is None:
                raise ConfigurationError(f"Is exactly needs a number. {RANGE_TEXT}")
            _check_byte(value)
        sizes = {r: len(d) for r, d in snapshot.regions.items()}
        if sizes != self._sizes:
            raise ConfigurationError(
                "The game's memory changed shape (another game?). Press New search."
            )
        offsets: dict[str, np.ndarray] = {}
        values: dict[str, np.ndarray] = {}
        before_kept: dict[str, np.ndarray] = {}
        for region, data in snapshot.regions.items():
            where = self._offsets[region]
            current = np.frombuffer(data, dtype=np.uint8)[where]
            before = self._values[region]
            if kind is ScanKind.EXACT:
                keep = current == value
            elif kind is ScanKind.UP:
                keep = current > before
            elif kind is ScanKind.DOWN:
                keep = current < before
            elif kind is ScanKind.CHANGED:
                keep = current != before
            else:
                keep = current == before
            offsets[region] = where[keep]
            values[region] = current[keep].copy()
            before_kept[region] = before[keep]
        self._offsets, self._values, self._before = offsets, values, before_kept
        self._frame = snapshot.frame

    def matches(self, limit: int | None = None) -> tuple[Match, ...]:
        """The bytes still in the search, by region then offset."""
        out: list[Match] = []
        for region in sorted(self._offsets):
            offsets, now, before = self._offsets[region], self._values[region], self._before[region]
            for i in range(len(offsets)):
                if limit is not None and len(out) >= limit:
                    return tuple(out)
                out.append(Match(region, int(offsets[i]), int(now[i]), int(before[i])))
        return tuple(out)


__all__ = ["RANGE_TEXT", "Match", "ScanKind", "ValueSearch", "parse_number"]
