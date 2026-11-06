"""T-884: the value search over all memory."""

from __future__ import annotations

import time

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.observation.search import (
    MAX_VALUE,
    Match,
    ScanKind,
    ValueSearch,
    holding,
    parse_number,
    patterns,
)


def snap(frame: int, ram: list[int], **more: list[int]) -> MemorySnapshot:
    regions = {"ram": bytes(ram), **{k: bytes(v) for k, v in more.items()}}
    return MemorySnapshot(frame, regions)


def test_start_keeps_every_byte_or_the_ones_equal() -> None:
    s = ValueSearch()
    assert not s.started and s.count == 0 and s.matches() == ()
    s.start(snap(1, [5, 7, 5], wram=[5, 0]))
    assert s.started and s.count == 5 and s.frame == 1
    s.start(snap(2, [5, 7, 5], wram=[5, 0]), value=5)
    assert s.count == 3
    assert s.matches() == (
        Match("ram", 0, 5, 5),
        Match("ram", 2, 5, 5),
        Match("wram", 0, 5, 5),
    )


@pytest.mark.parametrize(
    ("kind", "value", "kept"),
    [
        (ScanKind.EXACT, 9, [1, 3]),
        (ScanKind.UP, None, [1, 3]),
        (ScanKind.DOWN, None, [2]),
        (ScanKind.CHANGED, None, [1, 2, 3]),
        (ScanKind.UNCHANGED, None, [0, 4]),
    ],
)
def test_each_scan_kind(kind: ScanKind, value: int | None, kept: list[int]) -> None:
    s = ValueSearch()
    s.start(snap(0, [1, 2, 3, 4, 5]))
    s.refine(snap(4, [1, 9, 0, 9, 5]), kind, value)
    assert [m.offset for m in s.matches()] == kept
    assert s.frame == 4


def test_scans_chain_against_the_previous_scan() -> None:
    s = ValueSearch()
    s.start(snap(0, [3, 3, 3, 3]))
    s.refine(snap(1, [3, 4, 4, 2]), ScanKind.UP)  # 1 and 2
    assert [m.offset for m in s.matches()] == [1, 2]
    s.refine(snap(2, [9, 5, 4, 9]), ScanKind.UP)  # against 4, 4: only 1
    assert s.matches() == (Match("ram", 1, 5, 4),)
    s.refine(snap(3, [0, 5, 0, 0]), ScanKind.UNCHANGED)
    assert s.matches() == (Match("ram", 1, 5, 5),)


def test_matches_are_limited_and_reset_empties() -> None:
    s = ValueSearch()
    s.start(snap(0, list(range(100))))
    assert len(s.matches(limit=10)) == 10 and s.count == 100
    assert s.matches(limit=10)[-1].offset == 9
    s.reset()
    assert not s.started and s.count == 0 and s.matches() == ()


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("12", 12),
        (" 0 ", 0),
        ("255", 255),
        ("0x0C", 12),
        ("0XFF", 255),
        ("$0c", 12),
        ("0Ch", 12),
        ("300", 300),
        ("99999999", MAX_VALUE),
        ("0x1FFFF", 0x1FFFF),
    ],
)
def test_parse_number_reads_decimal_and_hex(text: str, value: int) -> None:
    assert parse_number(text) == value


@pytest.mark.parametrize(
    "text", ["", "  ", "abc", "-1", "100000000", "0x5F5E100", "0x", "$", "1.5"]
)
def test_parse_number_refuses_in_plain_words(text: str) -> None:
    with pytest.raises(ConfigurationError) as info:
        parse_number(text)
    assert "99,999,999" in str(info.value) or "number" in str(info.value)


def test_patterns_list_every_way_a_game_holds_a_number() -> None:
    """T-936: a score of 300 is two bytes, in whichever way this game stores numbers."""
    assert patterns(5) == ((5,),)
    assert patterns(12) == ((12,), (0x12,), (1, 2), (2, 1))
    three_hundred = patterns(300)
    for form in ((0x2C, 0x01), (0x01, 0x2C), (0x03, 0x00), (0x00, 0x03), (3, 0, 0), (0, 0, 3)):
        assert form in three_hundred
    assert (0x70, 0x11, 0x01) in patterns(70000) and (0x01, 0x11, 0x70) in patterns(70000)
    assert (0x07, 0x00, 0x00) in patterns(70000)
    for value in (0, 7, 12, 255, 256, 300, 65535, 70000, MAX_VALUE):
        forms = patterns(value)
        assert len(set(forms)) == len(forms)
        assert all(0 <= b <= 255 for form in forms for b in form)
    with pytest.raises(ConfigurationError):
        patterns(MAX_VALUE + 1)


def test_a_big_number_is_found_across_bytes() -> None:
    """T-936: searching 300 keeps every byte that takes part in a 300."""
    ram = [0] * 40
    ram[10:12] = [0x2C, 0x01]  # binary, low byte first
    ram[20:22] = [0x03, 0x00]  # packed BCD, high byte first
    ram[30:33] = [3, 0, 0]  # one digit per byte
    ram[39] = 0x2C  # a 300 cannot start on the last byte
    data = np.frombuffer(bytes(ram), dtype=np.uint8)
    found = set(np.flatnonzero(holding(data, 300)).tolist())
    assert {10, 11, 20, 21, 30, 31, 32} <= found and 39 not in found
    # The zeros around 0x03 at 20 also read as "00 03" and "0 0 3", which is right: they
    # are kept until a scan says otherwise.
    assert not holding(np.frombuffer(bytes([1, 2, 3, 4]), dtype=np.uint8), 300).any()
    assert not holding(np.frombuffer(b"\x2c", dtype=np.uint8), 300).any()
    s = ValueSearch()
    s.start(snap(0, ram), value=300)
    assert {m.offset for m in s.matches()} == found
    assert all(m.region == "ram" for m in s.matches())


def test_is_exactly_with_a_big_number() -> None:
    """T-936: "Is exactly 300" keeps the candidates still part of a 300; bytes are as before."""
    s = ValueSearch()
    s.start(snap(0, [0x2C, 0x01, 0x2C, 0x01, 7, 3]))
    s.refine(snap(1, [0x2C, 0x01, 0x2D, 0x01, 7, 3]), ScanKind.EXACT, 300)
    assert [m.offset for m in s.matches()] == [0, 1]
    s.refine(snap(2, [0x2C, 0x01, 0x2D, 0x01, 7, 3]), ScanKind.UNCHANGED)
    assert [m.offset for m in s.matches()] == [0, 1]
    t = ValueSearch()
    t.start(snap(0, [3, 1, 3]))
    t.refine(snap(1, [3, 1, 3]), ScanKind.EXACT, 3)
    assert [m.offset for m in t.matches()] == [0, 2]


def test_errors_leave_the_search_as_it_was() -> None:
    s = ValueSearch()
    with pytest.raises(ConfigurationError, match="New search"):
        s.refine(snap(0, [1]), ScanKind.UP)
    s.start(snap(0, [1, 2, 3]))
    with pytest.raises(ConfigurationError, match="number"):
        s.refine(snap(1, [2, 2, 2]), ScanKind.EXACT)
    with pytest.raises(ConfigurationError, match="99,999,999"):
        s.refine(snap(1, [2, 2, 2]), ScanKind.EXACT, MAX_VALUE + 1)
    with pytest.raises(ConfigurationError, match="99,999,999"):
        s.start(snap(1, [2, 2, 2]), value=-1)
    with pytest.raises(ConfigurationError, match="memory changed"):
        s.refine(snap(1, [2, 2]), ScanKind.UP)
    with pytest.raises(ConfigurationError, match="memory changed"):
        s.refine(MemorySnapshot(1, {"wram": b"\x00\x00\x00"}), ScanKind.UP)
    assert s.count == 3 and s.frame == 0
    assert [m.now for m in s.matches()] == [1, 2, 3]


def test_a_big_region_scans_quickly() -> None:
    size = 128 * 1024
    a = snap(0, [i % 256 for i in range(size)])
    b = snap(1, [(i + (i % 3 == 0)) % 256 for i in range(size)])
    s = ValueSearch()
    started = time.perf_counter()
    s.start(a)
    s.refine(b, ScanKind.CHANGED)
    elapsed = time.perf_counter() - started
    assert s.count == len([i for i in range(size) if i % 3 == 0])
    assert elapsed < 0.05
