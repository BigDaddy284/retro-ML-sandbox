"""T-884: the value search over all memory."""

from __future__ import annotations

import time

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.observation.search import Match, ScanKind, ValueSearch, parse_number


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
    [("12", 12), (" 0 ", 0), ("255", 255), ("0x0C", 12), ("0XFF", 255), ("$0c", 12), ("0Ch", 12)],
)
def test_parse_number_reads_decimal_and_hex(text: str, value: int) -> None:
    assert parse_number(text) == value


@pytest.mark.parametrize("text", ["", "  ", "abc", "-1", "256", "0x100", "0x", "$", "1.5"])
def test_parse_number_refuses_in_plain_words(text: str) -> None:
    with pytest.raises(ConfigurationError) as info:
        parse_number(text)
    assert "0 to 255" in str(info.value) or "number" in str(info.value)


def test_errors_leave_the_search_as_it_was() -> None:
    s = ValueSearch()
    with pytest.raises(ConfigurationError, match="New search"):
        s.refine(snap(0, [1]), ScanKind.UP)
    s.start(snap(0, [1, 2, 3]))
    with pytest.raises(ConfigurationError, match="number"):
        s.refine(snap(1, [2, 2, 2]), ScanKind.EXACT)
    with pytest.raises(ConfigurationError, match="0 to 255"):
        s.refine(snap(1, [2, 2, 2]), ScanKind.EXACT, 300)
    with pytest.raises(ConfigurationError, match="0 to 255"):
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
