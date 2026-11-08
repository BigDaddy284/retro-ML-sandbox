"""T-884: the value search over all memory."""

from __future__ import annotations

import time

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.observation.search import (
    MAX_VALUE,
    Form,
    Match,
    ScanKind,
    ValueSearch,
    forms,
    holding,
    parse_number,
    patterns,
    read_spans,
    spans,
    typed_digits,
)
from retro_ai_lab.understanding.knowledge import Encoding


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
    # wram's 5 is followed by a zero, so it also reads as a longer number (T-941):
    # the row stands for those readings too.
    assert s.matches() == (
        Match("ram", 0, 5, 5),
        Match("ram", 2, 5, 5),
        Match("wram", 0, 5, 5, Encoding.U8, 1, 4),
    )
    assert s.total == 6


@pytest.mark.parametrize(
    ("kind", "value", "kept"),
    [
        (ScanKind.EXACT, 9, [1, 3]),  # the 9 at 1 also reads over the 0 after it: one row
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
    assert s.count == len(kept) and s.frame == 4


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
    covered = {o for m in s.matches() for o in range(m.offset, m.offset + m.length)}
    assert {10, 11, 20, 21, 30, 31, 32} <= covered <= found
    assert all(m.region == "ram" for m in s.matches())
    by_address = {m.address: m for m in s.matches()}
    assert by_address["ram[10..11]"].held_as == "2 bytes, low first"
    assert by_address["ram[30..32]"].held_as == "3 digits, highest first"
    assert all(m.now == 300 for m in s.matches())


def test_is_exactly_with_a_big_number() -> None:
    """T-936: "Is exactly 300" keeps the candidates still part of a 300; bytes are as before."""
    s = ValueSearch()
    s.start(snap(0, [0x2C, 0x01, 0x2C, 0x01, 7, 3]))
    s.refine(snap(1, [0x2C, 0x01, 0x2D, 0x01, 7, 3]), ScanKind.EXACT, 300)
    assert s.matches() == (Match("ram", 0, 300, 300, Encoding.U16LE, 2),)
    s.refine(snap(2, [0x2C, 0x01, 0x2D, 0x01, 7, 3]), ScanKind.UNCHANGED)
    assert s.matches() == (Match("ram", 0, 300, 300, Encoding.U16LE, 2),)
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
    """The scan is array work, not a Python loop over the bytes.

    T-950: a fixed 50 ms bound on one run tripped twice on slow CI runners (64 ms,
    65 ms) while the scan itself was fine: a first run pays for page faults and
    whatever else the runner is doing. Now the best of three runs counts, and the
    bound is 50 ms or a plain Python pass over the same bytes timed in the same
    process, whichever is longer, so a slow machine slows the yardstick too.
    """
    size = 128 * 1024
    a = snap(0, [i % 256 for i in range(size)])
    b = snap(1, [(i + (i % 3 == 0)) % 256 for i in range(size)])
    runs = []
    for _ in range(3):
        s = ValueSearch()
        started = time.perf_counter()
        s.start(a)
        s.refine(b, ScanKind.CHANGED)
        runs.append(time.perf_counter() - started)
        assert s.count == len([i for i in range(size) if i % 3 == 0])
    started = time.perf_counter()
    ram_a, ram_b = a.regions["ram"], b.regions["ram"]
    assert len([i for i in range(size) if ram_a[i] != ram_b[i]]) == s.count
    python_pass = time.perf_counter() - started
    assert min(runs) < max(0.05, python_pass), (runs, python_pass)


def test_forms_name_how_each_pattern_is_held() -> None:
    """T-941: every pattern comes with its layout, and typed zeros fix the digit count."""
    assert forms(5) == (
        Form(Encoding.U8, (5,), False),
        Form(Encoding.U8, (5,), True),
        Form(Encoding.BCD, (5,), True),
        Form(Encoding.BCD, (5,), False),
        Form(Encoding.DIGITS_BE, (5,), True),
        Form(Encoding.DIGITS_LE, (5,), False),
    )
    by_layout = {f.encoding: f.bytes for f in forms(300)}
    assert by_layout[Encoding.U16LE] == (0x2C, 0x01) and by_layout[Encoding.U16BE] == (0x01, 0x2C)
    assert by_layout[Encoding.BCD_BE] == (0x03, 0x00) and by_layout[Encoding.BCD_LE] == (0x00, 0x03)
    assert by_layout[Encoding.DIGITS_BE] == (3, 0, 0) and by_layout[Encoding.DIGITS_LE] == (0, 0, 3)
    six = {f.encoding: f.bytes for f in forms(600, digits=6)}
    assert six[Encoding.DIGITS_BE] == (0, 0, 0, 6, 0, 0) and six[Encoding.BCD_BE] == (0, 6, 0)
    assert six[Encoding.U16LE] == (0x58, 0x02)
    assert typed_digits("000600") == 6 and typed_digits("0600") == 4
    assert typed_digits("600") is None and typed_digits("0x0C") is None
    assert typed_digits("0") is None and typed_digits(" 007 ") == 3
    with pytest.raises(ConfigurationError, match="digits"):
        forms(600, digits=2)
    with pytest.raises(ConfigurationError, match="digits"):
        forms(600, digits=9)
    assert patterns(300) == tuple(dict.fromkeys(f.bytes for f in forms(300)))


def test_a_score_over_six_digit_bytes_is_one_value() -> None:
    """T-941: the user's score, one digit per byte, rolled from 800 to 1,000 and the byte
    the search had kept went back to 0; the search now follows the whole number."""
    ram = [0] * 24
    ram[10:16] = [0, 0, 0, 6, 0, 0]  # 000600, highest digit first
    s = ValueSearch()
    s.start(snap(0, ram), value=600)
    rows = {m.address: m for m in s.matches()}
    assert rows["ram[13..15]"].held_as == "3 digits, highest first"
    assert rows["ram[13..15]"].readings == 6  # 3 to 8 digits, over the zeros before
    assert rows["ram[13..15]"].now == 600 and rows["ram[13..15]"].low_end == 15
    ram[10:16] = [0, 0, 1, 0, 0, 0]  # 001000
    s.refine(snap(1, ram), ScanKind.UP)
    rows = {m.address: m for m in s.matches()}
    assert "ram[13..15]" not in rows  # 0 did not go up
    assert rows["ram[12..15]"].now == 1000 and rows["ram[12..15]"].before == 600
    assert rows["ram[12..15]"].held_as == "4 digits, highest first"
    s.refine(snap(2, ram), ScanKind.EXACT, 1000)
    rows = {m.address: m for m in s.matches()}
    assert rows["ram[12..15]"].now == 1000 and all(m.now == 1000 for m in rows.values())
    # Typing the number as the game shows it, zeros and all, names the digit count.
    t = ValueSearch()
    t.start(snap(2, ram), value=1000, digits=6)
    rows = {m.address: m for m in t.matches()}
    assert rows["ram[10..15]"].held_as == "6 digits, highest first" and "ram[12..15]" not in rows
    ram[10:16] = [0, 0, 1, 2, 5, 0]
    t.refine(snap(3, ram), ScanKind.UP)
    rows = {m.address: m for m in t.matches()}
    assert rows["ram[10..15]"].now == 1250 and rows["ram[10..15]"].before == 1000
    # The same bytes read lowest digit first went up too (001000 to 052100 that way round);
    # the data cannot tell yet, and "Is exactly 1250" would.
    assert rows["ram[9..14]"].held_as == "6 digits, lowest first"
    t.refine(snap(4, ram), ScanKind.EXACT, 1250, digits=6)
    assert [m.address for m in t.matches()] == ["ram[10..15]"]


def test_a_span_that_stops_being_digits_is_dropped() -> None:
    """T-941: bytes that no longer read as digits (or BCD) are not that number."""
    s = ValueSearch()
    s.start(snap(0, [1, 2, 0x12, 0, 0]), value=12)
    held = {m.address: m.held_as for m in s.matches()}
    assert held == {"ram[0..1]": "2 digits, highest first", "ram[2]": "packed decimal, 1 byte"}
    s.refine(snap(1, [1, 11, 0x1A, 0, 0]), ScanKind.CHANGED)
    assert s.matches() == ()
    s.start(snap(2, [1, 2, 0x12, 0, 0]), value=12)
    s.refine(snap(3, [1, 3, 0x13, 0, 0]), ScanKind.UP)
    assert [(m.address, m.now) for m in s.matches()] == [("ram[0..1]", 13), ("ram[2]", 13)]


def test_is_exactly_keeps_spans_over_bytes_still_running() -> None:
    """T-941: from every byte, "Is exactly 300" finds the two-byte 300 whose bytes are both
    still in the running, and not one whose byte a scan already threw out."""
    s = ValueSearch()
    s.start(snap(0, [0x2C, 0x01, 9, 0x2C, 0x01]))
    s.refine(snap(1, [0x2C, 0x01, 9, 0x2C, 0x02]), ScanKind.UNCHANGED)  # byte 4 goes
    s.refine(snap(2, [0x2C, 0x01, 9, 0x2C, 0x01]), ScanKind.EXACT, 300)
    assert s.matches() == (Match("ram", 0, 300, 300, Encoding.U16LE, 2),)


def test_read_spans_decodes_every_layout_at_once() -> None:
    data = np.frombuffer(bytes([0x12, 0x34, 1, 2, 3, 0xA1]), dtype=np.uint8)
    starts = np.array([0], dtype=np.int64)
    assert read_spans(data, Encoding.U16LE, 2, starts)[0].tolist() == [0x3412]
    assert read_spans(data, Encoding.U16BE, 2, starts)[0].tolist() == [0x1234]
    assert read_spans(data, Encoding.BCD_BE, 2, starts)[0].tolist() == [1234]
    assert read_spans(data, Encoding.BCD_LE, 2, starts)[0].tolist() == [3412]
    two = np.array([2], dtype=np.int64)
    assert read_spans(data, Encoding.DIGITS_BE, 3, two)[0].tolist() == [123]
    assert read_spans(data, Encoding.DIGITS_LE, 3, two)[0].tolist() == [321]
    _values, valid = read_spans(data, Encoding.DIGITS_BE, 4, two)
    assert not valid[0]
    _values, valid = read_spans(data, Encoding.BCD_LE, 2, np.array([4], dtype=np.int64))
    assert not valid[0]
    assert read_spans(data, Encoding.UINT_BE, 4, starts)[0].tolist() == [0x12340102]
    found = spans(data, 0x1234)
    assert (Encoding.U16BE, 2) in found and found[(Encoding.U16BE, 2)].tolist() == [0]
