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


def test_a_walker_on_the_ground_has_a_kind() -> None:
    """T-977: in the user's game the enemies walk on the ground, were cut out of it by
    their motion, had no kind and so were never found to hurt."""
    w = level()
    tracker = ObjectTracker()
    for i in range(6):
        tracker.update(view(w, 100 + 2 * i, [(300 - i, 192, walker())], i))
    moving = [t for t in tracker.tracks() if t.moves]
    assert len(moving) == 1
    assert moving[0].cut_out and moving[0].kind is not None
    # The same kind every time it is cut out: its colours, the sky left out.
    walker_kinds = {k.id for k in tracker.catalogue.kinds()}
    assert moving[0].kind in walker_kinds


# ------------------------------------------------------------ T-987: a box left out


def hero() -> np.ndarray:
    s = np.zeros((16, 12, 3), np.uint8)
    s[:] = (228, 0, 88)
    s[0:4, 2:10] = (252, 188, 176)
    return s


def test_what_is_hidden_is_no_thing_and_leaves_no_trail() -> None:
    """T-987: the player is known by its colours and followed by the caller; hidden from
    the tracker it makes no track, standing or walking, and neither does the change it
    leaves behind where it was."""
    w = level()
    tracker = ObjectTracker()
    for i in range(8):  # the hero walks right, the camera after it; a walker walks ahead
        x, camera = 300 + 6 * i, 100 + 4 * i
        frame = view(w, camera, [(x, 192, hero()), (330 + 5 * i, 192, walker())], i)
        tracks = tracker.update(frame, hide=(x - camera - 1, 191, 14, 18))
        seen = [t for t in tracks if t.missing == 0 and t.moves]
        if i >= 2:
            # Only the walker moves, as far as the tracker can tell.
            assert len(seen) == 1 and seen[0].kind is not None
            assert abs(seen[0].centre[0] - (330 + 5 * i - camera + 8)) <= 6
    # The same walk with nothing hidden: the hero is a moving thing as well.
    plain = ObjectTracker()
    for i in range(8):
        x, camera = 300 + 6 * i, 100 + 4 * i
        plain.update(view(w, camera, [(x, 192, hero()), (330 + 5 * i, 192, walker())], i))
    assert len([t for t in plain.tracks() if t.missing == 0 and t.moves]) >= 2


def test_a_thing_touching_what_is_hidden_keeps_its_own_kind() -> None:
    w = level()
    tracker = ObjectTracker()
    kinds: set[str | None] = set()
    for i in range(8):  # in the air: the hero comes up against a walker that waits
        x = min(60 + 6 * i, 88)
        tracks = tracker.update(
            view(w, 0, [(x, 60, hero()), (100, 60, walker())], i), hide=(x - 1, 59, 14, 18)
        )
        things = [t for t in tracks if t.missing == 0 and t.y < 100]
        assert len(things) == 1 and not things[0].merged
        kinds.add(things[0].kind)
    assert len(kinds) == 1 and None not in kinds  # one walker, one kind, touching or not
    assert len(tracker.catalogue.kinds()) == 1


def test_a_hidden_box_is_checked() -> None:
    tracker = ObjectTracker()
    frame = view(level(), 0, [], 0)
    for bad in ((1, 2, 3), (1, 2, 0, 4), (1, 2, 3, 4.5), (True, 2, 3, 4), "box"):
        with pytest.raises(ConfigurationError):
            tracker.update(frame, hide=bad)  # type: ignore[arg-type]
    assert tracker.update(frame, hide=(-5, -5, 3, 3)) == tracker.tracks()  # off the screen


# ------------------------------------------------------------ T-992: the camera's move


def tall_view(world: np.ndarray, camera: int, top: int) -> Frame:
    """The screen at ``camera`` with the level slid ``top`` rows up (more sky above the
    level, more ground below it, so there is always something to show)."""
    sky = np.zeros((40, world.shape[1], 3), np.uint8)
    sky[:] = SKY
    sky[10:18, 50::83] = WHITE  # a few specks of cloud
    big = np.vstack([sky, world, world[208:240], world[208:216]])
    a = big[40 + top : 280 + top, camera : camera + 256]
    return Frame(256, 240, 3, np.ascontiguousarray(a).tobytes())


@pytest.mark.parametrize(
    "shift",
    [
        (16, 0),
        (-23, 0),
        (32, 0),
        (-32, 0),
        (0, 17),
        (0, -9),
        (13, 5),
        (-20, -11),
        (31, 31),
        (9, 0),
        (3, 0),
        (-1, 1),
        (0, 0),
    ],
)
def test_a_wide_scroll_is_found_coarse_then_fine(shift: tuple[int, int]) -> None:
    w = level()
    dx, dy = shift
    before, after = tall_view(w, 300, 0), tall_view(w, 300 + dx, dy)
    assert estimate_scroll(before, after, max_dx=32, max_dy=32) == shift
    # The reach is a limit: a move past it is not found (and nothing is made up
    # beyond it).
    found = estimate_scroll(before, after, max_dx=12, max_dy=12)
    assert abs(found[0]) <= 12 and abs(found[1]) <= 12
    if max(abs(dx), abs(dy)) <= 12:
        assert found == shift


def test_a_wide_search_costs_no_more_than_the_narrow_one() -> None:
    w = level()
    a, b = view(w, 100, [], 0), view(w, 116, [], 1)
    estimate_scroll(a, b, max_dx=32, max_dy=32)

    def cost(reach: int) -> float:
        start = time.perf_counter()
        for _ in range(5):
            estimate_scroll(a, b, max_dx=reach, max_dy=reach)
        return (time.perf_counter() - start) / 5

    wide, narrow = cost(32), cost(8)
    assert wide < 0.25, f"{wide * 1000:.0f} ms"
    assert wide < 1.5 * narrow, f"{wide * 1000:.0f} ms against {narrow * 1000:.0f} ms"


def plain(width: int = 1024, gaps: tuple[tuple[int, int], ...] = ()) -> np.ndarray:
    """Sky over plain ground from y=200, grass on top, nothing to tell a shift by but
    the gaps (first x, width)."""
    a = np.zeros((240, width, 3), np.uint8)
    a[:] = SKY
    a[200:] = BROWN
    a[200:204] = GREEN
    for x, wide in gaps:
        a[200:, x : x + wide] = SKY
    return a


def strip(frame: Frame, squares: int) -> Frame:
    """The same screen with a strip along the top: ``squares`` black squares that stay
    where they are on the screen (lives, a score)."""
    a = np.frombuffer(frame.pixels, np.uint8).reshape(240, 256, 3).copy()
    for i in range(squares):
        a[8:16, 8 + 12 * i : 16 + 12 * i] = BLACK
    return Frame(256, 240, 3, a.tobytes(), number=frame.number)


def test_pixels_left_out_do_not_vote() -> None:
    # One gap slides by 2 pixels under a strip of twelve squares that stays put: the
    # strip's edges outweigh the gap's, and the screen reads as still.
    w = plain(gaps=((400, 48),))
    before, after = strip(view(w, 300, [], 0), 12), strip(view(w, 302, [], 1), 12)
    assert estimate_scroll(before, after) == (0, 0)
    # With the strip's rows left out of both screens, the gap tells.
    out = np.zeros((240, 256), np.bool_)
    out[8:16] = True
    assert estimate_scroll(before, after, leave_out_before=out, leave_out_after=out) == (2, 0)
    assert estimate_scroll(
        before, after, max_dx=32, max_dy=32, leave_out_before=out, leave_out_after=out
    ) == (2, 0)
    # Everything left out: nothing can tell, and that reads as no move.
    everything = np.ones((240, 256), np.bool_)
    assert estimate_scroll(before, after, leave_out_after=everything) == (0, 0)
    assert estimate_scroll(before, after, max_dx=32, leave_out_before=everything) == (0, 0)


def test_bad_masks_are_refused() -> None:
    f = Frame(8, 6, 1, bytes(48))
    for bad in (
        np.zeros((8, 6), np.bool_),  # width by height
        np.zeros((6, 8), np.uint8),
        np.zeros((6, 8, 1), np.bool_),
        [[False] * 8] * 6,
        "mask",
    ):
        with pytest.raises(ConfigurationError):
            estimate_scroll(f, f, leave_out_before=bad)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            estimate_scroll(f, f, leave_out_after=bad)  # type: ignore[arg-type]
    assert estimate_scroll(f, f, leave_out_before=np.zeros((6, 8), np.bool_)) == (0, 0)


def big_hero() -> np.ndarray:
    s = np.zeros((24, 16, 3), np.uint8)
    s[:] = (228, 0, 88)
    s[0:6, 3:13] = (252, 188, 176)
    return s


def test_the_hidden_player_does_not_hold_the_camera_still() -> None:
    """The camera holds the player at one spot on the screen while the level slides
    by, so the player's own pixels say "nothing moved"."""
    w = plain(gaps=((470, 400),))  # one edge of a wide gap in sight
    seen = ObjectTracker(max_shift=32)
    hidden = ObjectTracker(max_shift=32)
    for i in range(6):
        camera = 300 + 2 * i
        frame = view(w, camera, [(camera + 80, 176, big_hero())], i)
        seen.update(frame)
        hidden.update(frame, hide=(79, 175, 18, 26))
        if i:
            assert seen.scroll == (0, 0)  # the player outvotes the one edge
            assert hidden.scroll == (2, 0)
    assert hidden.camera == (10, 0) and seen.camera == (0, 0)


def test_the_tracker_follows_a_camera_faster_than_eight_pixels_a_look() -> None:
    w = level()
    tracker = ObjectTracker(max_shift=32)
    camera = 100
    for i, step in enumerate((0, 16, 16, 20, 31, -27, -16, 9, 0, 3)):
        camera += step
        tracker.update(view(w, camera, [], i))
        assert tracker.scroll == (step, 0), i
    assert tracker.camera == (52, 0)
    # With the reach meant for a single frame, the same camera is lost (what was wrong).
    slow = ObjectTracker()
    slow.update(view(w, 100, [], 0))
    slow.update(view(w, 120, [], 1))
    assert slow.scroll != (20, 0)


def test_a_lone_mover_is_not_a_camera_move() -> None:
    """A walker pacing on plain ground, nothing else in sight: every pixel that changed
    fits "the camera moved the other way". One small thing is not the camera."""
    w = plain()
    tracker = ObjectTracker(max_shift=32)
    for i in range(8):
        tracker.update(view(w, 300, [(420 + 4 * i, 184, walker())], i))
        assert tracker.scroll == (0, 0) and tracker.camera == (0, 0)
    assert tracker.searches == 0
    moving = [t for t in tracker.tracks() if t.missing == 0 and t.moves]
    assert len(moving) == 1 and abs(moving[0].velocity[0] - 4) < 1
    # A thing that pops up or goes (a coin taken) is no camera move either.
    tracker.update(view(w, 300, [(420 + 28, 184, walker()), (350, 120, coin())], 8))
    assert tracker.scroll == (0, 0) and tracker.searches == 0


def test_a_still_screen_costs_no_search() -> None:
    w = level()
    tracker = ObjectTracker(max_shift=32)
    frame = view(w, 200, [], 0)
    for _ in range(5):
        tracker.update(frame)
    assert tracker.searches == 0 and tracker.scroll == (0, 0)
    tracker.update(view(w, 205, [], 1))
    assert tracker.searches == 1 and tracker.scroll == (5, 0)


def taught(squares: int = 12) -> tuple[ObjectTracker, np.ndarray]:
    """A tracker that has watched hilly ground slide by under a strip of squares, and
    the level: hills up to x=600, then plain ground with one gap at 800."""
    w = plain(gaps=((800, 48),))
    for x0 in range(0, 600, 97):
        w[150:200, x0 : x0 + 30] = GREEN
        w[150, x0 : x0 + 30] = BLACK
    tracker = ObjectTracker(max_shift=32)
    for i in range(4):
        tracker.update(strip(view(w, 100 + 16 * i, [], i), squares))
        assert tracker.scroll == ((16, 0) if i else (0, 0))
    return tracker, w


def test_a_strip_that_stands_still_is_learnt_as_glass() -> None:
    tracker, w = taught()
    # The strip's rows stood still through two camera moves that would have changed
    # them: they are glass. Nothing else is.
    assert tracker.glass == tuple(range(8, 16))
    # Over plain ground with one gap, a pixel a look: the strip would outvote the gap
    # (see test_pixels_left_out_do_not_vote); it no longer has a say.
    tracker.reset()
    assert tracker.glass == tuple(range(8, 16))  # kept over a go, like the kinds
    fresh = ObjectTracker(max_shift=32)
    for i in range(5):
        frame = strip(view(w, 620 + i, [], i), 12)
        tracker.update(frame)
        fresh.update(frame)
        if i:
            assert tracker.scroll == (1, 0)
            assert fresh.scroll == (0, 0)
    assert tracker.camera == (4, 0)


def test_what_is_on_the_glass_does_not_move() -> None:
    """With the scroll right, a square that stays put on the screen would read as moving
    through the level at the camera's speed. It is on the glass: it moves with nothing,
    and is not cut out of the scenery as a mover."""
    tracker, w = taught(squares=3)
    for i in range(4, 10):
        tracker.update(strip(view(w, 100 + 16 * i, [], i), 3))
        assert tracker.scroll == (16, 0)
    on_glass = [t for t in tracker.tracks() if t.glass]
    assert len(on_glass) == 3 and all(t.missing == 0 for t in on_glass)
    assert [t.x for t in sorted(on_glass, key=lambda t: t.x)] == [8, 20, 32]
    assert not any(t.moves for t in tracker.tracks())
    assert all(abs(t.velocity[0]) < 0.01 for t in on_glass)
    # A thing in the level is not on the glass, wherever the camera goes.
    tracker.update(strip(view(w, 260, [(400, 100, coin())], 10), 3))
    tracker.update(strip(view(w, 276, [(400, 100, coin())], 11), 3))
    coins = [t for t in tracker.tracks() if 100 <= t.y < 108 and t.missing == 0]
    assert len(coins) == 1 and not coins[0].glass and not coins[0].moves


def test_glass_is_unlearnt_when_the_row_moves_with_the_scenery() -> None:
    tracker, w = taught()
    assert tracker.glass == tuple(range(8, 16))
    # The strip is gone and a cloud in the level slides through its rows: rows that
    # change just as the camera's move says are scenery, not glass.
    w[8:16, 300:340] = WHITE
    for i in range(3):
        tracker.update(view(w, 180 + 16 * i, [], i))
    assert tracker.scroll == (16, 0)
    assert tracker.glass == ()
    # A frame of another size starts everything over, glass too.
    tracker2, _ = taught()
    tracker2.update(Frame(64, 48, 3, bytes(64 * 48 * 3)))
    assert tracker2.glass == ()


def test_the_best_of_a_bad_lot_is_no_move() -> None:
    """After a cut to another place some shift still fits best. It explains next to
    nothing of what changed, so it is no measure: the camera's count stays where it
    was and nothing is learnt from it."""
    tracker, w = taught()
    camera = tracker.camera
    other = np.zeros((240, 1024, 3), np.uint8)  # somewhere else altogether: indoors
    other[60:180:40, 30::70] = WHITE
    other[200:] = (60, 60, 60)
    tracker.update(strip(view(other, 500, [], 9), 12))
    assert tracker.scroll == (0, 0) and tracker.camera == camera
    assert tracker.glass == tuple(range(8, 16))
    # A move past the reach is the same: not guessed at.
    far = ObjectTracker(max_shift=12)
    far.update(view(w, 100, [], 0))
    far.update(view(w, 150, [], 1))
    assert far.scroll == (0, 0)


def small_track(position: int, things: tuple[tuple[int, int], ...] = ((70, 0), (90, 2))) -> Frame:
    """A small grey screen (64 by 48) looking down a track with three lanes: lines down
    the screen, blocks across a lane (how far along, which lane) coming down it as
    ``position`` grows, and a strip of two squares on top."""
    a = np.full((48, 64, 1), 16, np.uint8)
    a[1:3, 1:3] = 200
    a[1:3, 5:7] = 200
    a[8:, 21] = 64
    a[8:, 42] = 64
    for along, lane in things:
        row = 44 - (along - position)
        if 8 <= row < 47:
            a[row : row + 2, lane * 21 + 2 : lane * 21 + 19] = 160
    return Frame(64, 48, 1, a.tobytes(), number=position)


def test_small_and_far_are_measured_against_the_screen() -> None:
    # Two blocks come down the track 5 rows a look: a move, though each block is far
    # smaller than half a thing's largest size on a big screen.
    tracker = ObjectTracker(max_shift=32)
    for i in range(5):
        tracker.update(small_track(40 + 5 * i))
        assert tracker.scroll == ((0, -5) if i else (0, 0))
    # A quarter of this screen is 12 rows: 20 rows a look is past the reach, and is not
    # guessed at.
    fast = ObjectTracker(max_shift=32)
    fast.update(small_track(40))
    fast.update(small_track(60))
    assert fast.scroll == (0, 0)


def test_only_a_row_that_stood_still_both_ways_is_glass() -> None:
    """A move up or down the screen "would have changed" more rows than the strip that
    stood still: the empty rows the strip would have slid into, and the top of the lane
    lines, where the strip's rows would have come from. Neither stood still."""
    tracker = ObjectTracker(max_shift=32)
    for i in range(8):
        tracker.update(small_track(40 + 5 * i))
    assert tracker.glass == (1, 2)  # the strip, and nothing else
    # A block coming down through the top of the track is a thing in the level.
    tops = [t for t in tracker.tracks() if t.missing == 0 and t.y >= 8]
    assert tops and not any(t.glass for t in tops)


# ---------------------------------------------------------------- T-993


def test_what_the_move_leaves_unexplained() -> None:
    """After a real move next to nothing still differs once the move is allowed for;
    after a cut to another place most of what differed still does."""
    w = level()
    tracker = ObjectTracker(max_shift=32)
    tracker.update(view(w, 100, [], 0))
    assert tracker.unexplained == 0.0  # a first frame
    tracker.update(view(w, 100, [], 1))
    assert tracker.scroll == (0, 0) and tracker.unexplained == 0.0  # a still screen
    tracker.update(view(w, 116, [], 2))
    assert tracker.scroll == (16, 0) and tracker.unexplained == 0.0  # all of it explained
    # One small thing changing needs no search, and leaves nothing to explain.
    tracker.update(view(w, 116, [(200, 100, hero())], 3))
    assert tracker.scroll == (0, 0) and tracker.unexplained == 0.0
    # A cut: indoors, nothing like the level.
    other = np.zeros((240, 1024, 3), np.uint8)
    other[60:180:40, 30::70] = WHITE
    other[200:] = (60, 60, 60)
    tracker.update(view(other, 500, [], 4))
    assert tracker.scroll == (0, 0) and tracker.unexplained > 0.5
    # Looking on at the new place it is a still screen again.
    tracker.update(view(other, 500, [], 5))
    assert tracker.unexplained == 0.0
    # A move past the reach is not explained either, though less of the screen differs.
    tracker.update(view(other, 560, [], 6))
    assert tracker.scroll == (0, 0) and 0.0 < tracker.unexplained < 0.5
    tracker.reset()
    assert tracker.unexplained == 0.0


# ------------------------------------------------------------ T-1016: seen whole

GREY = (188, 188, 188)
PURPLE = (148, 0, 132)
FEET = (40, 0, 0)


def block(colour: tuple[int, int, int] = PURPLE, feet: tuple[int, int, int] = FEET) -> np.ndarray:
    """A walker drawn in flat colour, 14 by 14, with a strip of feet: between two looks
    only the edge it leads with looks any different."""
    s = np.zeros((14, 14, 3), np.uint8)
    s[:] = colour
    s[11:] = feet
    return s


def open_ground(width: int = 1024) -> np.ndarray:
    """Plain ground from y=208 with a strip of grass, and one post near the start:
    past the post nothing tells one place from another."""
    a = np.zeros((240, width, 3), np.uint8)
    a[:] = SKY
    a[208:] = BROWN
    a[208:212] = GREEN
    a[128:208, 150:166] = GREY
    return a


def boxes(tracker: ObjectTracker) -> set[tuple[int, int, int, int]]:
    """The boxes of the things seen at the last look that have a kind."""
    return {
        (t.x, t.y, t.width, t.height)
        for t in tracker.tracks()
        if t.missing == 0 and t.kind is not None
    }


def test_a_flat_coloured_walker_on_the_ground_is_seen_whole() -> None:
    """On the practice game the walker was a sliver beside the player, or nothing, at
    the look in which it took a life: none of those lives was put down to it."""
    w = open_ground()
    tracker, old = ObjectTracker(), ObjectTracker(whole=False)
    ids = set()
    for i in range(7):
        frame = view(w, 0, [(200 - 4 * i, 194, block())], i)
        tracker.update(frame)
        old.update(frame)
        if i == 0:
            continue
        whole = (200 - 4 * i, 194, 14, 14)
        mine = [t for t in tracker.tracks() if (t.x, t.y, t.width, t.height) == whole]
        assert len(mine) == 1 and mine[0].missing == 0
        assert mine[0].kind is not None and mine[0].cut_out and mine[0].colours
        ids.add(mine[0].id)
        # Nothing else with a kind is made of it (its other edge, a second copy).
        assert boxes(tracker) == {whole}
        # The old way: only the edge it leads with, four pixels wide.
        assert whole not in boxes(old) and all(b[2] <= 4 for b in boxes(old))
    assert len(ids) == 1  # one track from the first look it moved in
    (t,) = [t for t in tracker.tracks() if t.id in ids]
    assert t.moves and t.velocity[0] == pytest.approx(-4.0, abs=0.5)
    assert len(tracker.catalogue.kinds()) == 1  # one kind, whole or edge


def test_it_is_still_seen_when_it_stops() -> None:
    w = open_ground()
    tracker, old = ObjectTracker(), ObjectTracker(whole=False)
    for i in range(4):
        frame = view(w, 0, [(200 - 4 * i, 194, block())], i)
        tracker.update(frame)
        old.update(frame)
    (walker_id,) = [t.id for t in tracker.tracks() if t.colours]
    for i in range(4, 14):  # ten looks standing still: longer than a track survives unseen
        frame = view(w, 0, [(188, 194, block())], i)
        tracker.update(frame)
        old.update(frame)
        (t,) = [t for t in tracker.tracks() if t.id == walker_id]
        assert t.missing == 0 and (t.x, t.y, t.width, t.height) == (188, 194, 14, 14)
    assert t.velocity[0] == pytest.approx(0.0, abs=0.1)
    assert not [t for t in old.tracks() if t.kind is not None]  # the old way: lost
    # It walks on, and is the same track still.
    tracker.update(view(w, 0, [(184, 194, block())], 14))
    (t,) = [t for t in tracker.tracks() if t.id == walker_id]
    assert t.missing == 0 and (t.x, t.y) == (184, 194)


def test_another_of_its_kind_is_known_the_moment_it_is_in_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What a walker looks like is learnt from the first one seen to move. Any other,
    walking or not, is then seen for what it is at once: also in the next go."""
    from retro_ai_lab.observation import tracking

    w = open_ground()
    tracker = ObjectTracker()
    assert tracker.movers == ()
    for i in range(3):
        tracker.update(view(w, 0, [(200 - 4 * i, 194, block())], i))
    (kind,) = tracker.movers
    # A second walker that has never moved, and the first one gone.
    tracker.update(view(w, 0, [(60, 194, block())], 3))
    (other,) = [t for t in tracker.tracks() if t.missing == 0 and t.kind is not None]
    assert (other.x, other.y, other.width, other.height) == (60, 194, 14, 14)
    assert other.kind == kind and not other.moves
    # A new go: nothing followed is kept, what a walker looks like is. At the very
    # first look two of them are seen, standing still.
    tracker.reset()
    assert tracker.tracks() == () and tracker.movers == (kind,)
    tracker.update(view(w, 0, [(60, 194, block()), (120, 194, block())], 4))
    assert boxes(tracker) == {(60, 194, 14, 14), (120, 194, 14, 14)}
    assert {t.kind for t in tracker.tracks()} == {kind}
    # Something else in other colours is not one of them until it is seen to move.
    # (Turning up is a change, and is cut out as one, as it always was; standing
    # there afterwards it is part of the ground again.)
    green = block((0, 88, 0), (0, 40, 0))
    tracker.update(view(w, 0, [(60, 194, block()), (180, 194, green)], 5))
    tracker.update(view(w, 0, [(60, 194, block()), (180, 194, green)], 5))
    assert boxes(tracker) == {(60, 194, 14, 14)}
    # No more kinds are known this way than the tracker may keep; one over is still
    # filled out in each look it moves in, and lost when it stops.
    monkeypatch.setattr(tracking, "MOST_MOVERS", 1)
    for i in range(1, 4):
        tracker.update(view(w, 0, [(60, 194, block()), (180 - 4 * i, 194, green)], 5 + i))
        assert boxes(tracker) == {(60, 194, 14, 14), (180 - 4 * i, 194, 14, 14)}
    assert tracker.movers == (kind,)
    for i in range(4, 12):
        tracker.update(view(w, 0, [(60, 194, block()), (168, 194, green)], 5 + i))
    assert boxes(tracker) == {(60, 194, 14, 14)}


def test_it_is_still_seen_where_only_it_tells_the_cameras_move() -> None:
    """Past the post the walker is all that changes on the screen as the camera moves.
    The camera's measure takes the walker's move for the camera's own (that is the
    next task's); the walker is seen where it is all the same."""
    w = open_ground()
    tracker, old = ObjectTracker(max_shift=32), ObjectTracker(max_shift=32, whole=False)
    walker_id = None
    wrong, unseen = 0, 0
    for i in range(18):
        camera, at = 16 * i, 420 - 4 * i  # they meet: 20 pixels a look on the screen
        frame = view(w, camera, [(at, 194, block())], i)
        tracker.update(frame)
        old.update(frame)
        x = at - camera
        if i < 12:  # coming into view, and found, while the post is in view too
            continue
        if walker_id is None:
            (walker_id,) = [t.id for t in tracker.tracks() if t.colours]
        assert x >= 0 and x + 14 <= 256 and camera > 166  # whole on the screen, post gone
        (t,) = [t for t in tracker.tracks() if t.id == walker_id]
        assert t.missing == 0 and (t.x, t.y, t.width, t.height) == (x, 194, 14, 14)
        wrong += tracker.scroll != (16, 0)
        unseen += (x, 194, 14, 14) not in boxes(old) and not any(
            b[0] >= x - 4 and b[0] + b[2] <= x + 18 for b in boxes(old)
        )
    assert wrong >= 4  # the scenario is the one meant: the camera's move is misjudged
    assert unseen >= 4  # ... and the old way saw nothing of the walker in those looks


def test_scenery_walked_in_front_of_is_not_the_mover() -> None:
    """The strip of hill a walker uncovers is drawn in the hill's colours and joined to
    the whole hill; the hill did not move, so it is not taken for the thing."""
    w = level()
    tracker = ObjectTracker()
    for i in range(8):  # in front of the hill at 291 to 321, the camera moving
        tracker.update(view(w, 100 + 2 * i, [(330 - 2 * i, 194, block())], i))
        if i == 0:
            continue
        x = 330 - 2 * i - (100 + 2 * i)
        assert (x, 194, 14, 14) in boxes(tracker)
        assert all(t.width <= 14 and t.height <= 14 for t in tracker.tracks())
    assert tracker.scroll == (2, 0)


def test_colours_that_run_on_into_the_scenery_are_not_filled_out() -> None:
    """A walker in its ground's own colours cannot be told from the ground by them:
    it is cut out by its motion exactly as before."""
    w = open_ground()
    tracker, old = ObjectTracker(), ObjectTracker(whole=False)
    for i in range(7):
        frame = view(w, 0, [(200 - 4 * i, 194, block(BROWN, GREEN))], i)
        tracker.update(frame)
        old.update(frame)

        def seen(which: ObjectTracker) -> list[tuple[int | str | None, ...]]:
            return [(t.id, t.kind, t.x, t.y, t.width, t.height, t.missing) for t in which.tracks()]

        assert seen(tracker) == seen(old)
        assert not any(t.colours for t in tracker.tracks())
    assert seen(tracker)  # something was cut out: the edge that moved


def test_what_is_hidden_is_no_part_of_it() -> None:
    w = open_ground()
    tracker = ObjectTracker()
    hide = (80, 190, 14, 18)  # the player's box, as the watch hides it: to x = 94
    for i in range(6):
        x = 120 - 6 * i  # the walker walks up to it and half behind it
        tracker.update(view(w, 0, [(x, 194, block())], i), hide=hide)
        if i == 0:
            continue
        left = max(x, 94)
        assert boxes(tracker) == {(left, 194, x + 14 - left, 14)}
    (t,) = [t for t in tracker.tracks() if t.kind is not None]
    assert t.missing == 0 and t.x == 94 and t.width == 10  # the part that shows


def test_whole_must_be_true_or_false() -> None:
    for nonsense in (None, 1, "yes"):
        with pytest.raises(ConfigurationError, match="whole"):
            ObjectTracker(whole=nonsense)  # type: ignore[arg-type]


# ------------------------------------------------------------ T-1017: the camera near a mover
#
# Leaving the things known to move out of the camera's vote was tried and measured
# worse (see TASK_TRACKER.md, T-1017). These hold what the measure does today, so a
# later change to it is made knowingly.


def gapped(gap: int = 520) -> np.ndarray:
    """Open ground with one gap two tiles wide, far from the post."""
    a = open_ground()
    a[208:, gap : gap + 32] = SKY
    return a


def walk_past(world: np.ndarray, tracker: ObjectTracker, looks: range) -> list[tuple[int, int]]:
    """The camera moves 16 pixels a look and a walker comes the other way at 4; the
    camera's move as the tracker measured it at each of ``looks``."""
    measured = []
    for i in range(looks.stop):
        tracker.update(view(world, 16 * i, [(640 - 4 * i, 194, block())], i))
        if i in looks:
            measured.append(tracker.scroll)
    return measured


def test_a_gap_in_view_outvotes_a_walker() -> None:
    """The two edges of a gap in the ground are worth more pixels than a walker's two
    edges: with both in view the camera's move is measured right."""
    looks = range(21, 26)  # the post long gone; the gap (520) and the walker in view
    assert all(16 * i > 166 and 16 * i + 32 < 520 < 16 * i + 224 for i in looks)
    assert walk_past(gapped(), ObjectTracker(max_shift=32), looks) == [(16, 0)] * len(looks)


def test_where_only_a_walker_tells_its_move_is_taken_for_the_cameras() -> None:
    """One thing seen moving against nothing is one number for two unknowns. The
    measure takes the walker for scenery: 20 where the camera moved 16. Seeing things
    whole (T-1016) changes nothing in the measure."""
    looks = range(21, 26)  # past the post, nothing but the walker in view
    measured = walk_past(open_ground(), ObjectTracker(max_shift=32), looks)
    old = walk_past(open_ground(), ObjectTracker(max_shift=32, whole=False), looks)
    assert measured == old == [(20, 0)] * len(looks)


def test_a_walker_on_a_still_screen_is_no_camera_move() -> None:
    w = gapped(100)
    tracker = ObjectTracker(max_shift=32)
    for i in range(8):  # 12 pixels a look: more than "one small thing changing" covers
        tracker.update(view(w, 0, [(220 - 12 * i, 194, block())], i))
        assert tracker.scroll == (0, 0)
        if i:
            assert (220 - 12 * i, 194, 14, 14) in boxes(tracker)
    assert tracker.camera == (0, 0)


def test_a_known_kind_standing_still_tells_when_nothing_else_does() -> None:
    """A thing of a kind known to move that is standing still is all there is to go
    by on plain ground: the camera's move is measured by it."""
    w = open_ground()
    tracker = ObjectTracker(max_shift=32)
    for i in range(3):  # its kind is learnt
        tracker.update(view(w, 0, [(200 - 4 * i, 194, block())], i))
    assert tracker.movers
    tracker.reset()
    for i in range(8):  # past the post, one of the kind stands at 600 and never moves
        tracker.update(view(w, 400 + 20 * i, [(600, 194, block())], 3 + i))
        if i:
            assert tracker.scroll == (20, 0)
    assert tracker.camera == (140, 0)
