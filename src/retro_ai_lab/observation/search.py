"""A value search over all of the game's memory, like a cheat finder (T-884).

A person who knows a number on the screen (3 lives, 12 coins) can find where
it lives: start a search over every byte the emulator shares, play a little,
then keep only the values that went up, went down, changed, stayed the same
or now equal a number. A few scans usually leave one.

Each scan compares with the snapshot of the previous scan, so "went up" means
"went up since I last pressed a scan button". The candidates are values, not
bytes (T-941): a typed number is looked for in every way games hold numbers
(binary over one to four bytes in either byte order, packed BCD in either
byte order, one decimal digit per byte in either order), and each place found
is kept as a span of bytes with its layout, extended over the zero bytes
beyond its high end, because a 600 shown as 000600 is six digits, not three.
Every later scan compares the decoded number of each span, so a score that
rolls from 800 to 1,000 still "went up". A search from every byte starts with
one-byte values, as a cheat finder does. The app only reads the game's
memory, it never writes to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot
from retro_ai_lab.understanding.knowledge import (
    MAX_BINARY,
    MAX_DIGITS,
    Encoding,
    address_text,
)

MAX_BYTE = 255
MAX_VALUE = 99_999_999  # eight digits: four packed-BCD bytes, the most any 8-bit score shows
RANGE_TEXT = f"Type the number you see, like 12 or 300 (up to {MAX_VALUE:,}), or in hex like 0x0C."


Spans = dict[tuple[Encoding, int], np.ndarray]  # (layout, length) -> start offsets
HIGH_FIRST = frozenset({Encoding.U16BE, Encoding.UINT_BE, Encoding.BCD_BE, Encoding.DIGITS_BE})
BINARY = frozenset(
    {Encoding.U8, Encoding.U16LE, Encoding.U16BE, Encoding.UINT_LE, Encoding.UINT_BE}
)
PACKED = frozenset({Encoding.BCD, Encoding.BCD_LE, Encoding.BCD_BE})
DIGITS = frozenset({Encoding.DIGITS_LE, Encoding.DIGITS_BE})


class ScanKind(Enum):
    EXACT = "exact"
    UP = "up"
    DOWN = "down"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class Match:
    """One value still in the search: where it is, how it is held, its number now and before."""

    region: str
    offset: int
    now: int
    before: int
    encoding: Encoding = Encoding.U8
    length: int = 1
    readings: int = 1  # readings of the same number (this one and longer ones) this row stands for

    @property
    def low_end(self) -> int:
        """The byte holding the number's lowest part: the one a shorter reading shares."""
        return self.offset + self.length - 1 if self.encoding in HIGH_FIRST else self.offset

    @property
    def address(self) -> str:
        return address_text(self.region, self.offset, self.length)

    @property
    def held_as(self) -> str:
        return self.encoding.describe(self.length)


@dataclass(frozen=True, slots=True)
class Form:
    """One way a game might hold a number: the bytes, their layout, and which way it grows."""

    encoding: Encoding
    bytes: tuple[int, ...]
    high_end_first: bool  # the high part sits before the bytes (big-endian, digits as read)

    @property
    def length(self) -> int:
        return len(self.bytes)


def typed_digits(text: str) -> int | None:
    """How many decimal digits were typed, zeros and all: ``000600`` is six (T-941).

    None for hex, or for a plain number with no leading zero, which says nothing
    about how many digits the game shows.
    """
    raw = text.strip()
    if not raw.isdigit() or len(raw) < 2 or raw[0] != "0":
        return None
    return len(raw)


def parse_number(text: str) -> int:
    """Read ``12``, ``0x0C``, ``$0C`` or ``0Ch`` as a number."""
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


def _binary(length: int, high_first: bool) -> Encoding:
    if length == 1:
        return Encoding.U8
    if length == 2:
        return Encoding.U16BE if high_first else Encoding.U16LE
    return Encoding.UINT_BE if high_first else Encoding.UINT_LE


def _packed(length: int, high_first: bool) -> Encoding:
    if length == 1:
        return Encoding.BCD
    return Encoding.BCD_BE if high_first else Encoding.BCD_LE


def sized_encoding(encoding: Encoding, length: int, high_first: bool) -> Encoding:
    """The layout of ``encoding``'s family at ``length`` bytes, the high part first or
    last (T-941): a one-byte 5 grows into either order."""
    if encoding in BINARY:
        return _binary(length, high_first)
    if encoding in PACKED:
        return _packed(length, high_first)
    return encoding


def forms(value: int, digits: int | None = None) -> tuple[Form, ...]:
    """Every way a game might hold ``value``, each layout once (T-936, T-941).

    Binary in as many bytes as the number needs, low byte first and high byte
    first; packed BCD (two decimal digits per byte), either byte order; and one
    decimal digit per byte, lowest first or highest first. ``digits`` is how
    many decimal digits the game shows, when the person typed them (000600):
    the decimal forms then take that many, zeros and all.
    """
    _check_value(value)
    if digits is not None and not len(str(value)) <= digits <= MAX_DIGITS:
        raise ConfigurationError(f"A number shows 1 to {MAX_DIGITS} digits. {RANGE_TEXT}")
    out: list[Form] = []

    def add(encoding: Encoding, data: tuple[int, ...], high_end_first: bool) -> None:
        form = Form(encoding, data, high_end_first)
        if form not in out:
            out.append(form)

    width = max(1, (value.bit_length() + 7) // 8)
    little = tuple((value >> (8 * i)) & 0xFF for i in range(width))
    add(_binary(width, False), little, False)
    add(_binary(width, True), tuple(reversed(little)), True)
    text = str(value) if digits is None else f"{value:0{digits}d}"
    if len(text) % 2:
        text = "0" + text
    packed = tuple(int(text[i : i + 2], 16) for i in range(0, len(text), 2))
    add(_packed(len(packed), True), packed, True)
    add(_packed(len(packed), False), tuple(reversed(packed)), False)
    unpacked = tuple(int(d) for d in (str(value) if digits is None else f"{value:0{digits}d}"))
    add(Encoding.DIGITS_BE, unpacked, True)
    add(Encoding.DIGITS_LE, tuple(reversed(unpacked)), False)
    return tuple(out)


def patterns(value: int) -> tuple[tuple[int, ...], ...]:
    """The byte sequences of ``forms(value)``, each listed once (T-936)."""
    out: list[tuple[int, ...]] = []
    for form in forms(value):
        if form.bytes not in out:
            out.append(form.bytes)
    return tuple(out)


def spans(data: np.ndarray, value: int, digits: int | None = None) -> Spans:
    """Every span of ``data`` holding ``value`` in some layout, by (layout, length) (T-941).

    A found number is also read with the zero bytes beyond its high end, up to
    the layout's longest form: 600 at the end of 000600 is six digits as much
    as three, and the scans decide between them. The same bytes read the same
    way in two layouts (a 5 is one byte in binary, BCD or digits) are one span.
    """
    found: dict[tuple[Encoding, int], set[int]] = {}
    seen: set[tuple[int, ...]] = set()
    size = len(data)
    for form in forms(value, digits):
        k = form.length
        if k > size:
            continue
        hit = np.ones(size - k + 1, dtype=bool)
        for j, byte in enumerate(form.bytes):
            hit &= data[j : size - k + 1 + j] == byte
        starts = np.flatnonzero(hit)
        if len(starts) == 0:
            continue
        if form.bytes not in seen:
            seen.add(form.bytes)
            found.setdefault((form.encoding, k), set()).update(starts.tolist())
        longest = MAX_DIGITS if form.encoding in DIGITS else MAX_BINARY
        for n in range(k + 1, longest + 1):
            if form.high_end_first:
                starts = starts[starts >= 1] - 1
                starts = starts[data[starts] == 0]
            else:
                starts = starts[starts + n <= size]
                starts = starts[data[starts + n - 1] == 0]
            if len(starts) == 0:
                break
            layout = sized_encoding(form.encoding, n, form.high_end_first)
            found.setdefault((layout, n), set()).update(starts.tolist())
    return {key: np.array(sorted(offsets), dtype=np.int64) for key, offsets in found.items()}


def holding(data: np.ndarray, value: int) -> np.ndarray:
    """A mask over ``data``: True at every byte that takes part in ``value`` (T-936)."""
    mask = np.zeros(len(data), dtype=bool)
    for (_, k), starts in spans(data, value).items():
        for start in starts:
            mask[start : start + k] = True
    return mask


def read_spans(
    data: np.ndarray, encoding: Encoding, length: int, starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the spans at ``starts``: (values, valid), vectorised (T-941).

    A span whose bytes are not digits (over 9, or a nibble over 9 in BCD) is not
    valid: it is not that kind of number any more.
    """
    columns = [data[starts + j].astype(np.int64) for j in range(length)]
    high_first = encoding in (Encoding.U16BE, Encoding.UINT_BE, Encoding.BCD_BE, Encoding.DIGITS_BE)
    if not high_first:
        columns = columns[::-1]
    values = np.zeros(len(starts), dtype=np.int64)
    valid = np.ones(len(starts), dtype=bool)
    if encoding in (
        Encoding.U8,
        Encoding.U16LE,
        Encoding.U16BE,
        Encoding.UINT_LE,
        Encoding.UINT_BE,
    ):
        for column in columns:
            values = values * 256 + column
    elif encoding in PACKED:
        for column in columns:
            hi, lo = column >> 4, column & 0xF
            valid &= (hi <= 9) & (lo <= 9)
            values = values * 100 + hi * 10 + lo
    elif encoding in DIGITS:
        for column in columns:
            valid &= column <= 9
            values = values * 10 + column
    elif encoding is Encoding.S8:
        values = np.where(columns[0] > 127, columns[0] - 256, columns[0])
    else:
        values = (columns[0] != 0).astype(np.int64)
    return values, valid


class ValueSearch:
    """The values still in the running, with the memory of the last two scans."""

    def __init__(self) -> None:
        self._spans: dict[str, Spans] = {}  # region -> candidates by (layout, length)
        self._now: dict[str, np.ndarray] = {}  # region -> the bytes at the last scan
        self._before: dict[str, np.ndarray] = {}  # region -> the bytes at the scan before
        self._frame: int | None = None

    @property
    def started(self) -> bool:
        return self._frame is not None

    @property
    def frame(self) -> int | None:
        """The frame of the snapshot the last scan used."""
        return self._frame

    @property
    def total(self) -> int:
        """Every reading still in the running, the longer readings of a number included."""
        return int(sum(len(o) for by in self._spans.values() for o in by.values()))

    @property
    def count(self) -> int:
        """The values still in the running, a number's longer readings counted with it."""
        if all(set(by) <= {(Encoding.U8, 1)} for by in self._spans.values()):
            return self.total
        return sum(len(self._rows(region)) for region in self._spans)

    def reset(self) -> None:
        self._spans, self._now, self._before, self._frame = {}, {}, {}, None

    def start(
        self, snapshot: MemorySnapshot, value: int | None = None, digits: int | None = None
    ) -> None:
        """Begin again with every byte, or with every value equal to ``value``."""
        if value is not None:
            forms(value, digits)  # checks both
        found: dict[str, Spans] = {}
        now: dict[str, np.ndarray] = {}
        for region, data in snapshot.regions.items():
            current = np.frombuffer(data, dtype=np.uint8).copy()
            if value is None:
                found[region] = {(Encoding.U8, 1): np.arange(len(current), dtype=np.int64)}
            else:
                found[region] = spans(current, value, digits)
            now[region] = current
        self._spans, self._now, self._before = found, now, now
        self._frame = snapshot.frame

    def refine(
        self,
        snapshot: MemorySnapshot,
        kind: ScanKind,
        value: int | None = None,
        digits: int | None = None,
    ) -> None:
        """Keep the values that pass ``kind`` against the previous scan."""
        if not self.started:
            raise ConfigurationError("Press New search first.")
        if kind is ScanKind.EXACT:
            if value is None:
                raise ConfigurationError(f"Is exactly needs a number. {RANGE_TEXT}")
            forms(value, digits)  # checks both
        sizes = {r: len(d) for r, d in snapshot.regions.items()}
        if sizes != {r: len(d) for r, d in self._now.items()}:
            raise ConfigurationError(
                "The game's memory changed shape (another game?). Press New search."
            )
        kept: dict[str, Spans] = {}
        now: dict[str, np.ndarray] = {}
        for region, data in snapshot.regions.items():
            whole = np.frombuffer(data, dtype=np.uint8).copy()
            now[region] = whole
            if kind is ScanKind.EXACT:
                assert value is not None
                kept[region] = self._exact(region, whole, value, digits)
                continue
            by: Spans = {}
            for (encoding, length), starts in self._spans[region].items():
                current, valid = read_spans(whole, encoding, length, starts)
                before, _ = read_spans(self._now[region], encoding, length, starts)
                if kind is ScanKind.UP:
                    keep = current > before
                elif kind is ScanKind.DOWN:
                    keep = current < before
                elif kind is ScanKind.CHANGED:
                    keep = current != before
                else:
                    keep = current == before
                keep &= valid
                if keep.any():
                    by[(encoding, length)] = starts[keep]
            kept[region] = by
        self._spans, self._before, self._now = kept, self._now, now
        self._frame = snapshot.frame

    def _exact(self, region: str, whole: np.ndarray, value: int, digits: int | None) -> Spans:
        """The spans holding ``value`` now whose bytes were all still in the running."""
        covered = np.zeros(len(whole), dtype=bool)
        for (_, length), starts in self._spans[region].items():
            for j in range(length):
                covered[starts + j] = True
        out: Spans = {}
        for (encoding, length), starts in spans(whole, value, digits).items():
            inside = np.ones(len(starts), dtype=bool)
            for j in range(length):
                inside &= covered[starts + j]
            if inside.any():
                out[(encoding, length)] = starts[inside]
        return out

    def _rows(self, region: str) -> list[Match]:
        """The region's values, one row per number: its shortest reading, with the count of
        longer readings (the same low byte, the same number now) it stands for."""
        rows: list[Match] = []
        for (encoding, length), starts in self._spans[region].items():
            if len(starts) == 0:
                continue
            current, _ = read_spans(self._now[region], encoding, length, starts)
            before, _ = read_spans(self._before[region], encoding, length, starts)
            rows.extend(
                Match(region, int(starts[i]), int(current[i]), int(before[i]), encoding, length)
                for i in range(len(starts))
            )
        rows.sort(key=lambda m: (m.low_end, m.now, m.length, m.encoding.value))
        out: list[Match] = []
        for row in rows:
            last = out[-1] if out else None
            if last is not None and (last.low_end, last.now) == (row.low_end, row.now):
                out[-1] = Match(
                    last.region,
                    last.offset,
                    last.now,
                    last.before,
                    last.encoding,
                    last.length,
                    last.readings + 1,
                )
            else:
                out.append(row)
        out.sort(key=lambda m: (m.offset, m.length))
        return out

    def matches(self, limit: int | None = None) -> tuple[Match, ...]:
        """The values still in the search, by region then offset: one row per number, its
        shortest reading, with ``readings`` saying how many longer readings it stands for."""
        out: list[Match] = []
        for region in sorted(self._spans):
            for row in self._rows(region):
                if limit is not None and len(out) >= limit:
                    return tuple(out)
                out.append(row)
        return tuple(out)


__all__ = [
    "MAX_VALUE",
    "RANGE_TEXT",
    "Form",
    "Match",
    "ScanKind",
    "ValueSearch",
    "forms",
    "holding",
    "parse_number",
    "patterns",
    "read_spans",
    "sized_encoding",
    "spans",
    "typed_digits",
]
