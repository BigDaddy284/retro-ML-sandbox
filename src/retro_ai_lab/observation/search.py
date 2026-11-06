"""A value search over all of the game's memory, like a cheat finder (T-884).

A person who knows a number on the screen (3 lives, 12 coins) can find the byte
that holds it: start a search over every byte the emulator shares, play a
little, then keep only the bytes that went up, went down, changed, stayed the
same or now equal a number. A few scans usually leave one byte.

Each scan compares the bytes with the snapshot of the previous scan, so "went
up" means "went up since I last pressed a scan button". The rows are bytes,
but the number typed can be anything a game shows (T-936): a score of 300 is
looked for in every way games hold numbers (binary over one to four bytes in
either byte order, packed BCD in either byte order, one digit per byte in
either order) and every byte taking part is kept. The app only reads the
game's memory, it never writes to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot

MAX_BYTE = 255
MAX_VALUE = 99_999_999  # eight digits: four packed-BCD bytes, the most any 8-bit score shows
RANGE_TEXT = f"Type the number you see, like 12 or 300 (up to {MAX_VALUE:,}), or in hex like 0x0C."


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
    _check_value(value)
    return value


def _check_value(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_VALUE:
        raise ConfigurationError(f"The search takes 0 to {MAX_VALUE:,}. {RANGE_TEXT}")


def patterns(value: int) -> tuple[tuple[int, ...], ...]:
    """Every byte sequence a game might hold ``value`` as (T-936), each listed once.

    Binary in as many bytes as the number needs, low byte first and high byte
    first; packed BCD (two decimal digits per byte), either byte order; and one
    decimal digit per byte, most-significant first or last, for numbers of two
    digits or more. A one-byte number is its own binary form.
    """
    _check_value(value)
    out: list[tuple[int, ...]] = []

    def add(p: tuple[int, ...]) -> None:
        if p not in out:
            out.append(p)

    width = max(1, (value.bit_length() + 7) // 8)
    little = tuple((value >> (8 * i)) & 0xFF for i in range(width))
    add(little)
    add(tuple(reversed(little)))
    digits = str(value)
    if len(digits) % 2:
        digits = "0" + digits
    packed = tuple(int(digits[i : i + 2], 16) for i in range(0, len(digits), 2))
    add(packed)
    add(tuple(reversed(packed)))
    if value >= 10:
        unpacked = tuple(int(d) for d in str(value))
        add(unpacked)
        add(tuple(reversed(unpacked)))
    return tuple(out)


def holding(data: np.ndarray, value: int) -> np.ndarray:
    """A mask over ``data``: True at every byte that takes part in ``value`` (T-936)."""
    mask = np.zeros(len(data), dtype=bool)
    for pattern in patterns(value):
        k = len(pattern)
        if k > len(data):
            continue
        hit = np.ones(len(data) - k + 1, dtype=bool)
        for j, byte in enumerate(pattern):
            hit &= data[j : len(data) - k + 1 + j] == byte
        for start in np.flatnonzero(hit):
            mask[start : start + k] = True
    return mask


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
            _check_value(value)
        offsets: dict[str, np.ndarray] = {}
        values: dict[str, np.ndarray] = {}
        for region, data in snapshot.regions.items():
            current = np.frombuffer(data, dtype=np.uint8)
            if value is None:
                keep = np.arange(len(current), dtype=np.int64)
            else:
                keep = np.flatnonzero(holding(current, value))
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
            _check_value(value)
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
            whole = np.frombuffer(data, dtype=np.uint8)
            current = whole[where]
            before = self._values[region]
            if kind is ScanKind.EXACT:
                assert value is not None
                keep = holding(whole, value)[where]
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


__all__ = [
    "MAX_VALUE",
    "RANGE_TEXT",
    "Match",
    "ScanKind",
    "ValueSearch",
    "holding",
    "parse_number",
    "patterns",
]
