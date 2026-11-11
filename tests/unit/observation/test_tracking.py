"""T-972: following things between frames, with the screen's scroll taken away."""

from __future__ import annotations

import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.tracking import (
    ObjectTracker,
    estimate_scroll,
    moving_mask,
)

SKY = (92, 148, 252)
BROWN = (200, 76, 12)
DARK = (120, 40, 0)
BLACK = (0, 0, 0)
WHITE = (252, 252, 252)
GOLD = (252, 188, 60)
GREEN = (0, 168, 0)


def walker() -> np.ndarray:
    s = np.zeros((16, 16, 3), np.uint8)
    s[:] = SKY
    s[2:10, 3:13] = DARK
    s[1, 5:11] = DARK
    s[5:7, 5] = WHITE
    s[5:7, 9] = WHITE
    s[10:14, 6:10] = WHITE
    s[14:16, 2:7] = BLACK
    s[15, 10:13] = BLACK
    return s


def coin() -> np.ndarray:
    c = np.zeros((8, 6, 3), np.uint8)
    c[:] = SKY
    c[1:7, 1:5] = GOLD
    c[3:5, 2:4] = WHITE
    return c


def level(width: int = 1024) -> np.ndarray:
    """A long level: sky, hills (texture for the scroll), and brick ground from y=208."""
    a = np.zeros((240, width, 3), np.uint8)
    a[:] = SKY
    for x0 in range(0, width, 97):
        a[150:208, x0 : x0 + 30] = GREEN  # hills, irregular spacing
        a[150, x0 : x0 + 30] = BLACK
    for x0 in range(0, width, 16):
        a[208:240, x0 : x0 + 16] = BROWN
        a[208, x0 : x0 + 16] = BLACK
        a[208:240, x0] = BLACK
    return a


def view(
    world: np.ndarray, camera: int, things: list[tuple[int, int, np.ndarray]], number: int
) -> Frame:
    """The 256-wide screen at ``camera``, with things drawn at level positions."""
    a = world[:, camera : camera + 256].copy()
    for lx, y, spr in things:
        x = lx - camera
        h, w = spr.shape[:2]
        solid = (spr != np.array(SKY, np.uint8)).any(axis=2)
        for yy in range(h):
            for xx in range(w):
                if solid[yy, xx] and 0 <= x + xx < 256 and 0 <= y + yy < 240:
                    a[y + yy, x + xx] = spr[yy, xx]
    return Frame(256, 240, 3, a.tobytes(), number=number)


# ------------------------------------------------------------ scroll


@settings(max_examples=25, deadline=None)
@given(camera=st.integers(min_value=8, max_value=700), dx=st.integers(min_value=-8, max_value=8))
def test_scroll_is_found(camera: int, dx: int) -> None:
    w = level()
    assert estimate_scroll(view(w, camera, [], 0), view(w, camera + dx, [], 1)) == (dx, 0)


def test_a_plain_screen_reads_as_no_scroll() -> None:
    a = np.zeros((240, 256, 3), np.uint8)
    a[:] = SKY
    f = Frame(256, 240, 3, a.tobytes())
    assert estimate_scroll(f, f) == (0, 0)


def test_vertical_scroll_is_found() -> None:
    w = level()
    a = w[:, 100:356]
    before = Frame(256, 240, 3, np.ascontiguousarray(a).tobytes())
    shifted = np.roll(a, -3, axis=0)
    after = Frame(256, 240, 3, np.ascontiguousarray(shifted).tobytes())
    assert estimate_scroll(before, after) == (0, 3)


def test_scroll_settings_and_shapes_are_checked() -> None:
    f = Frame(4, 4, 1, bytes(16))
    with pytest.raises(ConfigurationError):
        estimate_scroll(f, Frame(4, 4, 3, bytes(48)))
    with pytest.raises(ConfigurationError):
        estimate_scroll(f, f, max_dx=-1)


def test_scroll_is_fast_enough() -> None:
    w = level()
    a, b = view(w, 100, [], 0), view(w, 103, [], 1)
    estimate_scroll(a, b)
    start = time.perf_counter()
    for _ in range(5):
        estimate_scroll(a, b)
    per = (time.perf_counter() - start) / 5
    assert per < 0.25, f"{per * 1000:.0f} ms"


# ------------------------------------------------------------ what moved


def test_only_the_walker_moves_once_the_scroll_is_taken_away() -> None:
    w = level()
    before = view(w, 100, [(300, 192, walker())], 0)
    after = view(w, 103, [(298, 192, walker())], 1)
    moved = moving_mask(before, after, (3, 0))
    ys, xs = np.nonzero(moved)
    assert len(ys) > 0
    # Everything that moved is round the walker (screen x 195..213, standing on the ground).
    assert xs.min() >= 194 and xs.max() <= 214 and ys.min() >= 192 and ys.max() < 208


def test_nothing_moves_on_a_still_level() -> None:
    w = level()
    assert not moving_mask(view(w, 50, [], 0), view(w, 53, [], 1), (3, 0)).any()


# ------------------------------------------------------------ tracks


def test_a_walker_on_the_ground_is_followed_with_its_level_velocity() -> None:
    """It touches the ground, so one still screen cannot cut it out; motion does."""
    w = level()
    tracker = ObjectTracker()
    for i in range(8):
        tracker.update(view(w, 100 + 2 * i, [(300 - i, 192, walker())], i))
    moving = [t for t in tracker.tracks() if t.moves]
    assert len(moving) == 1
    t = moving[0]
    assert t.velocity[0] == pytest.approx(-1.0, abs=0.3)
    assert abs(t.velocity[1]) < 0.3
    assert tracker.scroll == (2, 0)
    assert tracker.camera == (14, 0)


def test_a_coin_that_slides_past_with_the_camera_stands_still_in_the_level() -> None:
    w = level()
    tracker = ObjectTracker()
    for i in range(6):
        tracker.update(view(w, 100 + 3 * i, [(250, 100, coin())], i))
    coins = [t for t in tracker.tracks() if t.width == 4]
    assert len(coins) == 1
    assert not coins[0].moves
    assert coins[0].velocity == pytest.approx((0.0, 0.0), abs=0.01)
    assert coins[0].seen == 6


def test_two_walkers_keep_their_own_tracks() -> None:
    w = level()
    tracker = ObjectTracker()
    for i in range(6):
        tracker.update(view(w, 0, [(40 + 2 * i, 60, walker()), (200 - 2 * i, 60, walker())], i))
    ts = [t for t in tracker.tracks() if t.moves]
    assert len(ts) == 2
    left, right = sorted(ts, key=lambda t: t.x)
    assert left.velocity[0] == pytest.approx(2.0, abs=0.2)
    assert right.velocity[0] == pytest.approx(-2.0, abs=0.2)
    assert left.kind == right.kind is not None  # the same kind of thing


def test_a_thing_gone_for_a_while_is_dropped_and_a_blink_is_not() -> None:
    w = level()
    tracker = ObjectTracker(max_missing=2)
    tracker.update(view(w, 0, [(100, 60, coin())], 0))
    first = tracker.tracks()[0].id
    tracker.update(view(w, 0, [], 1))  # one frame out of sight (a flicker)
    tracker.update(view(w, 0, [(100, 60, coin())], 2))
    assert [t.id for t in tracker.tracks()] == [first]
    for i in range(3, 7):
        tracker.update(view(w, 0, [], i))
    assert tracker.tracks() == ()


def test_reset_forgets_tracks_but_keeps_kinds() -> None:
    w = level()
    tracker = ObjectTracker()
    tracker.update(view(w, 0, [(100, 60, coin())], 0))
    tracker.reset()
    assert tracker.tracks() == () and tracker.camera == (0, 0)
    assert len(tracker.catalogue.kinds()) == 1


def test_a_frame_of_another_size_starts_over() -> None:
    tracker = ObjectTracker()
    tracker.update(view(level(), 0, [(100, 60, coin())], 0))
    small = Frame(16, 16, 3, bytes(16 * 16 * 3))
    assert tracker.update(small) == ()


def test_tracker_settings_are_checked() -> None:
    for kw in ({"max_shift": -1}, {"max_jump": -1}, {"max_missing": -1}):
        with pytest.raises(ConfigurationError):
            ObjectTracker(**kw)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        ObjectTracker().update("frame")  # type: ignore[arg-type]


def test_following_costs_little_enough_to_run_now_and_then() -> None:
    """Under a tenth of a second a frame here; the AI does not need it every frame."""
    w = level()
    tracker = ObjectTracker()
    frames = [
        view(w, 100 + 2 * i, [(300 - i, 192, walker()), (260, 100, coin())], i) for i in range(6)
    ]
    tracker.update(frames[0])
    start = time.perf_counter()
    for f in frames[1:]:
        tracker.update(f)
    per = (time.perf_counter() - start) / 5
    assert per < 0.1, f"{per * 1000:.0f} ms"


def test_two_things_that_touch_keep_their_tracks() -> None:
    """Touching sprites are drawn as one patch; both tracks stay, marked as merged (T-976)."""
    w = level()
    tracker = ObjectTracker()
    for i in range(6):  # the coin waits, the walker walks into it
        tracker.update(view(w, 0, [(60 + 4 * i, 60, walker()), (90, 62, coin())], i))
    ids = {t.id for t in tracker.tracks()}
    assert len(ids) == 2
    assert any(t.merged for t in tracker.tracks())
    assert all(t.missing == 0 for t in tracker.tracks())
