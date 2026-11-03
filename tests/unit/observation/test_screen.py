"""T-203: pixel diffs and per-cell screen activity."""

from __future__ import annotations

import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.observation.screen import (
    FrameDiff,
    ScreenTracker,
    cell_bounds,
    diff_frames,
    to_array,
)


def grey(w: int, h: int, fill: int = 0, number: int = 0) -> Frame:
    return Frame(w, h, 1, bytes([fill]) * (w * h), number=number)


def with_pixels(frame: Frame, points: dict[tuple[int, int], int]) -> Frame:
    arr = to_array(frame).copy()
    for (x, y), v in points.items():
        arr[y, x, :] = v
    return Frame(frame.width, frame.height, frame.channels, arr.tobytes(), number=frame.number + 1)


# ------------------------------------------------------------ helpers


def test_to_array_shape_and_values() -> None:
    f = Frame(2, 2, 3, bytes(range(12)))
    a = to_array(f)
    assert a.shape == (2, 2, 3)
    assert a[1, 0].tolist() == [6, 7, 8]
    assert f.pixel(0, 1) == (6, 7, 8)


@pytest.mark.parametrize(
    ("size", "parts", "expected"),
    [
        (10, 2, [(0, 5), (5, 10)]),
        (10, 3, [(0, 3), (3, 6), (6, 10)]),
        (7, 7, [(i, i + 1) for i in range(7)]),
    ],
)
def test_cell_bounds_cover_the_span(size: int, parts: int, expected: list[tuple[int, int]]) -> None:
    b = cell_bounds(size, parts)
    assert b == expected
    assert b[0][0] == 0 and b[-1][1] == size
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))


# --------------------------------------------------------------- diff


def test_identical_frames_give_empty_diff() -> None:
    d = diff_frames(grey(16, 12, number=3), grey(16, 12, number=4))
    assert d.is_empty and d.changed_pixels == 0 and d.bbox is None
    assert d.fraction_changed == 0.0
    assert (d.before_number, d.after_number) == (3, 4)
    assert d.rows == 6 and d.columns == 8
    assert d.changed_cells() == ()


def test_diff_counts_pixels_bbox_and_cells() -> None:
    before = grey(16, 12)
    after = with_pixels(before, {(0, 0): 9, (15, 11): 9, (3, 1): 9})
    d = diff_frames(before, after, grid=(4, 3))
    assert d.changed_pixels == 3
    assert d.bbox == (0, 0, 15, 11)
    assert d.fraction_changed == pytest.approx(3 / 192)
    assert d.cells == ((2, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1))
    assert d.changed_cells() == ((0, 0), (3, 2))


def test_threshold_ignores_small_changes() -> None:
    before = grey(8, 8, 100)
    after = with_pixels(before, {(1, 1): 103, (2, 2): 120})
    assert diff_frames(before, after).changed_pixels == 2
    d = diff_frames(before, after, threshold=5)
    assert d.changed_pixels == 1 and d.bbox == (2, 2, 2, 2)
    assert diff_frames(before, after, threshold=50).is_empty


def test_multichannel_pixel_changes_once_even_if_all_channels_change() -> None:
    before = Frame(4, 2, 3, bytes(24))
    arr = to_array(before).copy()
    arr[1, 2] = (1, 2, 3)
    after = Frame(4, 2, 3, arr.tobytes())
    d = diff_frames(before, after, grid=(4, 2))
    assert d.changed_pixels == 1 and d.bbox == (2, 1, 2, 1)
    assert d.cells == ((0, 0, 0, 0), (0, 0, 1, 0))


def test_diff_rejects_shape_mismatch_bad_threshold_and_bad_grid() -> None:
    with pytest.raises(ConfigurationError, match="shape"):
        diff_frames(grey(4, 4), grey(4, 5))
    with pytest.raises(ConfigurationError, match="shape"):
        diff_frames(grey(4, 4), Frame(4, 4, 3, bytes(48)))
    for t in (-1, 256):
        with pytest.raises(ConfigurationError, match="threshold"):
            diff_frames(grey(4, 4), grey(4, 4), threshold=t)
    with pytest.raises(ConfigurationError, match="grid"):
        diff_frames(grey(4, 4), grey(4, 4), grid=(0, 2))
    with pytest.raises(ConfigurationError, match="finer"):
        diff_frames(grey(4, 4), grey(4, 4), grid=(5, 2))


@settings(max_examples=30, deadline=None)
@given(
    w=st.integers(2, 12),
    h=st.integers(2, 12),
    data=st.data(),
)
def test_diff_matches_naive_count_and_bbox(w: int, h: int, data: st.DataObject) -> None:
    a = data.draw(st.binary(min_size=w * h, max_size=w * h))
    b = data.draw(st.binary(min_size=w * h, max_size=w * h))
    d = diff_frames(Frame(w, h, 1, a), Frame(w, h, 1, b), grid=(2, 2))
    changed = [(x, y) for y in range(h) for x in range(w) if a[y * w + x] != b[y * w + x]]
    assert d.changed_pixels == len(changed)
    assert sum(sum(r) for r in d.cells) == len(changed)
    if changed:
        xs = [x for x, _ in changed]
        ys = [y for _, y in changed]
        assert d.bbox == (min(xs), min(ys), max(xs), max(ys))
    else:
        assert d.bbox is None


def test_frame_diff_is_immutable() -> None:
    d = FrameDiff(0, 1, 2, 2, 0, None, ((0, 0), (0, 0)))
    with pytest.raises(AttributeError):
        d.changed_pixels = 3  # type: ignore[misc]


# ------------------------------------------------------------ tracker


def test_tracker_seeds_then_accumulates_activity() -> None:
    t = ScreenTracker(grid=(2, 2))
    f0 = grey(8, 8)
    assert t.observe(f0) is None
    assert t.observed == 1 and t.transitions == 0
    assert t.activity(0, 0) == 0.0
    f1 = with_pixels(f0, {(0, 0): 1})  # top-left cell
    f2 = with_pixels(f1, {(7, 7): 1, (1, 1): 1})  # bottom-right and top-left
    d1 = t.observe(f1)
    d2 = t.observe(f2)
    assert d1 is not None and d1.changed_cells() == ((0, 0),)
    assert d2 is not None and set(d2.changed_cells()) == {(0, 0), (1, 1)}
    assert t.transitions == 2
    assert t.activity(0, 0) == 1.0
    assert t.activity(1, 1) == 0.5
    assert t.activity(1, 0) == 0.0
    assert t.activity_map() == ((1.0, 0.0), (0.0, 0.5))


def test_tracker_classify() -> None:
    t = ScreenTracker(grid=(2, 1))
    f = grey(4, 2)
    t.observe(f)
    for i in range(10):
        f = with_pixels(f, {(0, 0): i + 1} if i < 2 else {(0, 0): i + 1, (3, 1): i + 1})
        t.observe(f)
    # left cell changed every time (busy); right cell 8 of 10 (busy); none static
    assert t.classify() == {"static": (), "active": (), "busy": ((0, 0), (1, 0))}
    assert t.classify(busy_above=0.9) == {"static": (), "active": ((1, 0),), "busy": ((0, 0),)}
    with pytest.raises(ConfigurationError):
        t.classify(static_below=0.6, busy_above=0.5)


def test_tracker_resets_on_resolution_change_and_reset() -> None:
    t = ScreenTracker(grid=(1, 1))
    t.observe(grey(4, 4))
    t.observe(with_pixels(grey(4, 4), {(0, 0): 1}))
    assert t.transitions == 1 and t.activity(0, 0) == 1.0
    assert t.observe(grey(8, 8)) is None  # new size: start over, no exception
    assert t.observed == 1 and t.activity(0, 0) == 0.0
    t.reset()
    assert t.observed == 0


def test_tracker_rejects_bad_input() -> None:
    with pytest.raises(ConfigurationError):
        ScreenTracker(threshold=300)
    with pytest.raises(ConfigurationError):
        ScreenTracker(grid=(0, 1))
    t = ScreenTracker()
    with pytest.raises(ConfigurationError):
        t.observe(b"frame")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="outside"):
        t.activity(8, 0)


# ------------------------------------------------- on the real toy game


def test_tracker_sees_the_racer_move_and_the_border_stay_still() -> None:
    game = ToyRacer(seed=1)
    t = ScreenTracker(grid=(4, 4))
    t.observe(game.render())
    diffs = []
    for i in range(60):
        game.step(("up", "left") if i % 20 < 10 else ("up", "right"))
        d = t.observe(game.render())
        assert d is not None
        diffs.append(d)
    assert any(not d.is_empty for d in diffs), "the screen never changed while driving"
    amap = t.activity_map()
    assert max(max(r) for r in amap) > 0.0
    groups = t.classify()
    # Something moves and something stays still on a racing screen.
    assert groups["busy"] or groups["active"]
    assert groups["static"] or min(min(r) for r in amap) < max(max(r) for r in amap)


@pytest.mark.slow
def test_diff_is_fast_at_console_resolution() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (224, 256, 3), dtype=np.uint8)
    b = a.copy()
    b[100:120, 50:90] += 1
    fa = Frame(256, 224, 3, a.tobytes())
    fb = Frame(256, 224, 3, b.tobytes())
    start = time.perf_counter()
    for _ in range(60):
        diff_frames(fa, fb)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"60 diffs took {elapsed:.2f}s"
