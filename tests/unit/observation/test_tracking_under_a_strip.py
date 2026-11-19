"""T-1061: the camera's move read under a strip of the screen that stands still.

The screens here are a stand-in made in this file: a wide, tall place with little in
it (a backdrop of one colour and small marks scattered over it, none in step with
another, as the lamps and posts along a road or the stars of a starfield are), seen
through a screen of 256 by 240 with a solid status bar along one edge.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation import tracking
from retro_ai_lab.observation.tracking import ObjectTracker, estimate_scroll

W, H = 256, 240
NIGHT = (16, 16, 40)
MARK = (252, 252, 252)
POST = (0, 168, 0)
LAMP = (252, 188, 60)
BAR = (0, 0, 0)
LIFE = (216, 40, 0)
BRICK = (200, 76, 12)
DARK = (120, 40, 0)
CLOUD = (188, 188, 252)
REACH = 32  # as the watch sets it (training/vision.py LOOK_SHIFT)
START = (500, 500)  # where the camera stands in the place at first
TALL = 16  # rows of a bar along the top or bottom
WIDE = 24  # columns of a panel at the left or right

Thing = tuple[int, int, np.ndarray]  # where it is on the screen, and its picture


def place(marks: int = 700, seed: int = 7, size: int = 1400) -> np.ndarray:
    """A place with little in it: dashes, posts and lamps scattered over one colour,
    about twenty to a screen."""
    rng = np.random.default_rng(seed)
    a = np.zeros((size, size, 3), np.uint8)
    a[:] = NIGHT
    for _ in range(marks):
        y, x = (int(v) for v in rng.integers(0, size - 16, 2))
        kind = int(rng.integers(0, 3))
        if kind == 0:
            a[y : y + 16, x : x + 2] = MARK
        elif kind == 1:
            a[y : y + 8, x : x + 8] = POST
        else:
            a[y : y + 4, x : x + 4] = LAMP
    return a


THIN = place()


def rich(size: int = 1400) -> np.ndarray:
    """A place full of things to tell a move by: bricks all over, none like the next."""
    rng = np.random.default_rng(3)
    a = np.zeros((size, size, 3), np.uint8)
    a[:] = NIGHT
    for y in range(0, size, 16):
        for x in range(0, size, 16):
            if rng.integers(0, 3) == 0:
                a[y : y + 12, x : x + 12] = BRICK
                a[y + 3 : y + 6, x + int(rng.integers(2, 8))] = DARK
    return a


def bar(a: np.ndarray, where: str, count: int = 0, tall: int = TALL) -> None:
    """Paint a solid status bar along one edge of the screen ``a``: three lives and a
    counter that shows ``count``."""
    if where in ("top", "bottom"):
        y = 0 if where == "top" else H - tall
        a[y : y + tall] = BAR
        for i in range(3):
            a[y + 4 : y + 12, 8 + 12 * i : 16 + 12 * i] = LIFE
        for i in range(4):  # four figures, each a block as tall as the figure it shows
            figure = (count // 10**i) % 10
            a[y + 12 - 1 - figure % 8 : y + 12, W - 16 - 10 * i : W - 10 - 10 * i] = MARK
    else:
        x = 0 if where == "left" else W - WIDE
        a[:, x : x + WIDE] = BAR
        for i in range(3):
            a[8 + 12 * i : 16 + 12 * i, x + 8 : x + 16] = LIFE
        for i in range(4):
            figure = (count // 10**i) % 10
            a[H - 16 - 10 * i : H - 10 - 10 * i, x + 20 - 1 - figure % 8 : x + 20] = MARK


def seen(
    world: np.ndarray,
    camera: tuple[int, int],
    *,
    where: str | None = "top",
    count: int = 0,
    things: Sequence[Thing] = (),
    tall: int = TALL,
) -> Frame:
    """The screen with the camera at ``camera`` (across, down) in ``world``, the things
    drawn where they are on the screen, and the bar over it all."""
    x, y = camera
    a = world[y : y + H, x : x + W].copy()
    for tx, ty, picture in things:
        h, w = picture.shape[:2]
        a[ty : ty + h, tx : tx + w] = picture
    if where is not None:
        bar(a, where, count, tall)
    return Frame(W, H, 3, a.tobytes())


def level() -> np.ndarray:
    """A place that stands still in these tests: one colour, and a floor of bricks, none
    like the next, across the bottom of the screen seen from ``START``."""
    rng = np.random.default_rng(5)
    a = np.zeros((1400, 1400, 3), np.uint8)
    a[:] = NIGHT
    for x in range(0, 1400, 16):
        a[700:740, x : x + 16] = BRICK
        a[700:740, x] = DARK
        a[700 + int(rng.integers(4, 30)), x : x + 10] = DARK
    return a


LEVEL = level()
RICH = rich()


def still(*things: Sequence[Thing], where: str | None = "top") -> ObjectTracker:
    """A tracker shown the level from one spot, look after look, with other things on
    it each time."""
    tracker = ObjectTracker(max_shift=REACH)
    for now in things:
        tracker.update(seen(LEVEL, START, where=where, things=now))
        if now is not things[0]:
            assert tracker.scroll == (0, 0) and tracker.sighted == (0, 0)
            assert not tracker.windowed
    assert tracker.camera == (0, 0) and tracker.read_as == (0, 0)
    return tracker


def block(width: int, height: int, colour: tuple[int, int, int] = BRICK) -> np.ndarray:
    a = np.zeros((height, width, 3), np.uint8)
    a[:] = colour
    a[height // 3 : height // 3 + 2, width // 4 : width // 2] = DARK
    return a


def moved_by(
    move: tuple[int, int],
    *,
    where: str | None = "top",
    world: np.ndarray = THIN,
    tracker: ObjectTracker | None = None,
    counts: tuple[int, int] = (0, 0),
) -> ObjectTracker:
    """A tracker shown the place before and after the camera moved by ``move``."""
    tracker = tracker if tracker is not None else ObjectTracker(max_shift=REACH)
    there = (START[0] + move[0], START[1] + move[1])
    tracker.update(seen(world, START, where=where, count=counts[0]))
    tracker.update(seen(world, there, where=where, count=counts[1]))
    return tracker


@pytest.fixture
def searched(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """The searches through the shifts, in turn: "whole" for one in which every pixel
    with a say votes, "changed" for one in which only those that changed do."""
    calls: list[str] = []
    whole, changed = tracking._search, tracking._search_each

    def counted_whole(*args: object, **kwargs: object) -> tuple[int, int]:
        calls.append("whole")
        return whole(*args, **kwargs)  # type: ignore[arg-type]

    def counted_changed(*args: object, **kwargs: object) -> tuple[int, int]:
        calls.append("changed")
        return changed(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tracking, "_search", counted_whole)
    monkeypatch.setattr(tracking, "_search_each", counted_changed)
    yield calls


# ------------------------------------------------------------ the cause


def test_a_bar_along_the_top_outvotes_thin_scenery() -> None:
    """The fault this task cures, on one pair of screens. The camera moved 8 rows. Laid
    over each other as they are, the screens differ in the few marks that moved; moved
    by the 8 rows, they differ in the 8 rows of scenery that came out from under the
    bar, which is more. So the whole screen's vote says it stood still."""
    before, after = seen(THIN, START), seen(THIN, (500, 508))
    assert estimate_scroll(before, after, max_dx=REACH, max_dy=REACH) == (0, 0)
    same = np.frombuffer(before.pixels, np.uint8) != np.frombuffer(after.pixels, np.uint8)
    assert 0 < int(same.reshape(H, W, 3).any(axis=2).sum()) < 8 * W
    # Without the bar the same two screens tell.
    bare = seen(THIN, START, where=None), seen(THIN, (500, 508), where=None)
    assert estimate_scroll(*bare, max_dx=REACH, max_dy=REACH) == (0, 8)


# ------------------------------------------------------------ read under a bar

ACROSS_THE_BAR = {  # moves that take scenery under the bar or out from under it
    "top": [(0, -4), (0, -8), (0, 12), (0, -16), (0, 24), (0, -1), (0, 32), (-7, 5), (12, -9)],
    "bottom": [(0, -4), (0, 8), (0, -12), (0, 16), (0, -24), (0, 2), (0, -32), (-7, 5), (12, -9)],
    "left": [(4, 0), (-8, 0), (9, 0), (-16, 0), (20, 0), (3, 0), (32, 0), (-7, 5), (12, -9)],
    "right": [(-4, 0), (8, 0), (-7, 0), (16, 0), (-24, 0), (-1, 0), (-32, 0), (-7, 5), (12, -9)],
}
ALONG_THE_BAR = {
    "top": [(9, 0), (-7, 0), (20, 0), (3, 0), (32, 0)],
    "bottom": [(-9, 0), (7, 0), (-20, 0), (-3, 0), (-32, 0)],
    "left": [(0, -4), (0, -8), (0, 12), (0, 24), (0, -32)],
    "right": [(0, 4), (0, 8), (0, -12), (0, -24), (0, 32)],
}


@pytest.mark.parametrize(
    ("where", "move"), [(w, m) for w, moves in ACROSS_THE_BAR.items() for m in moves]
)
def test_the_move_is_read_under_a_bar(
    where: str, move: tuple[int, int], searched: list[str]
) -> None:
    tracker = moved_by(move, where=where)
    assert tracker.sighted == move and tracker.scroll == move and tracker.camera == move
    assert tracker.windowed and tracker.read_as == (0, 1)
    assert tracker.unexplained < 0.01  # next to nothing is left unexplained
    # The whole screen was asked first; then what changed, twice.
    assert searched == ["whole", "changed", "changed"] and tracker.searches == 1


@pytest.mark.parametrize(
    ("where", "move"), [(w, m) for w, moves in ALONG_THE_BAR.items() for m in moves]
)
def test_a_move_along_the_bar_is_told_by_the_whole_screen(
    where: str, move: tuple[int, int], searched: list[str]
) -> None:
    """Scenery that slides along a bar does not come out from under it: the whole
    screen tells, as it did before this task, and nothing more is looked for."""
    tracker = moved_by(move, where=where)
    assert tracker.sighted == move and tracker.scroll == move
    assert not tracker.windowed and tracker.read_as == (1, 0)
    assert searched == ["whole"]


@pytest.mark.parametrize("move", [(0, -8), (0, 24), (9, 0), (-7, 5), (12, -9), (0, 1), (32, 0)])
def test_without_a_bar_nothing_has_changed(move: tuple[int, int], searched: list[str]) -> None:
    tracker = moved_by(move, where=None)
    assert tracker.sighted == move and not tracker.windowed and tracker.read_as == (1, 0)
    assert searched == ["whole"]


@pytest.mark.parametrize("where", ["top", "bottom", "left", "right"])
def test_a_bar_over_rich_scenery_is_told_by_the_whole_screen(
    where: str, searched: list[str]
) -> None:
    tracker = ObjectTracker(max_shift=REACH)
    for i in range(4):
        tracker.update(seen(RICH, (500 + 5 * i, 500 + 8 * i), where=where))
        assert not tracker.windowed
        assert tracker.scroll == ((5, 8) if i else (0, 0))
    assert tracker.read_as == (3, 0) and searched == ["whole"] * 3


@pytest.mark.parametrize(
    ("where", "move"), [("top", (0, 8)), ("left", (6, 0)), ("bottom", (3, -5))]
)
def test_a_counter_running_in_the_bar_does_not_stop_it(where: str, move: tuple[int, int]) -> None:
    """Two of the counter's figures change at every look. Some of what changed in the
    bar may happen to be as the move says; it is not taken for part of what moved."""
    tracker = ObjectTracker(max_shift=REACH)
    for i in range(7):
        camera = (START[0] + move[0] * i, START[1] + move[1] * i)
        tracker.update(seen(THIN, camera, where=where, count=17 * i))
        assert tracker.scroll == (move if i else (0, 0))
        assert tracker.windowed == bool(i)
    assert tracker.camera == (6 * move[0], 6 * move[1]) and tracker.read_as == (0, 6)


LOOK_AFTER_LOOK = {  # moves of every size and slant, twelve looks of each
    "top": [(0, -1), (0, -5), (0, -13), (0, -31), (6, -4), (-3, -7), (9, -13)],
    "bottom": [(0, -3), (0, -7), (0, -22), (6, -4), (9, -13)],
    "left": [(5, 0), (-10, 0), (-14, 0), (21, 0), (6, -4), (5, 3), (2, -1), (-1, -1)],
    "right": [(3, 0), (-10, 0), (-14, 0), (21, 0), (6, -4), (-3, -7), (2, -1), (-1, -1)],
}


@pytest.mark.parametrize(
    ("where", "move"), [(w, m) for w, moves in LOOK_AFTER_LOOK.items() for m in moves]
)
def test_look_after_look_a_move_of_any_size_is_read(where: str, move: tuple[int, int]) -> None:
    """Thin things tell a move only at the very shift it was: a dash two pixels wide is
    not seen at all by a look at every fourth pixel and every fourth shift, which is how
    the whole screen is searched. So what changed is tried at every shift."""
    tracker = ObjectTracker(max_shift=REACH)
    for i in range(13):
        camera = (START[0] + move[0] * i, START[1] + move[1] * i)
        tracker.update(seen(THIN, camera, where=where, count=i))
        assert tracker.scroll == (move if i else (0, 0)), i
    assert tracker.camera == (12 * move[0], 12 * move[1])


def changed_between(
    was: tuple[int, int], now: tuple[int, int], where: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two screens packed as the tracker has them, the pixels that changed between them,
    and a mask that lets every pixel of the one before vote."""
    a = tracking._packed(seen(THIN, now, where=where))
    b = tracking._packed(seen(THIN, was, where=where))
    return a, b, a != b, np.ones(a.shape, np.bool_)


def test_the_whole_screens_kind_of_search_misses_some(monkeypatch: pytest.MonkeyPatch) -> None:
    """Why what changed has a search of its own: with the whole screen's kind in its
    place, three looks in twelve of a slide of (6, -4) under a panel are not read."""
    monkeypatch.setattr(tracking, "_search_each", tracking._search)
    tracker = ObjectTracker(max_shift=REACH)
    read = []
    for i in range(13):
        tracker.update(seen(THIN, (START[0] + 6 * i, START[1] - 4 * i), where="left", count=i))
        read.append(tracker.scroll)
    assert read.count((6, -4)) == 9 and read.count((0, 0)) == 4


@pytest.mark.parametrize("few", [128, 256, 512])
def test_where_much_changed_those_that_vote_are_picked_without_a_pattern(
    few: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other pixel in turn would be one side of each dash, and a move a pixel
    short fits one side of a dash as well as the right one does."""
    monkeypatch.setattr(tracking, "FEW", few)
    for i in range(4):
        was, now = (500 - 14 * i, 500), (500 - 14 * (i + 1), 500)
        a, b, changed, all_of_it = changed_between(was, now, "right")
        assert int(np.count_nonzero(changed)) > 2 * few
        assert tracking._search_each(a, b, changed, all_of_it, REACH, REACH) == (-14, 0)
    # The same pixels are picked each time: the answer does not change from run to run.
    again = tracking._search_each(a, b, changed, all_of_it, REACH, REACH)
    assert again == (-14, 0)


def test_what_changed_is_searched_as_the_whole_screen_is() -> None:
    night = np.zeros((12, 16), np.int32)
    a, b = night.copy(), night.copy()
    b[2, 2], a[2, 5] = 9, 9  # one mark, three pixels to the right at the look after
    changed, all_of_it = a != b, np.ones(a.shape, np.bool_)
    # The pixel it left has no partner on the screen under that move, and is not counted.
    assert tracking._search_each(a, b, changed, all_of_it, 4, 4) == (-3, 0)
    assert tracking._search(a, b, changed, all_of_it, 4, 4) == (-3, 0)
    # Nothing votes: no move. Nothing to vote with on the screen before: no move.
    nothing = np.zeros(a.shape, np.bool_)
    assert tracking._search_each(a, b, nothing, all_of_it, 4, 4) == (0, 0)
    assert tracking._search_each(a, b, changed, nothing, 4, 4) == (0, 0)
    # Moves that fit alike: the smallest, as everywhere.
    assert tracking._search_each(night, night, changed, all_of_it, 4, 4) == (0, 0)
    stripes = np.zeros((12, 16), np.int32)
    stripes[:, ::4] = 9
    slid = np.roll(stripes, 1, axis=1)  # one pixel on, or three back: one fits as the other
    assert tracking._search_each(slid, stripes, slid != stripes, all_of_it, 4, 4) == (-1, 0)


def test_the_players_box_left_out_changes_nothing() -> None:
    """The watch hides the player (the camera holds it still on the screen, and here it
    steers across as the road slides): hidden or not, the move is read."""
    car = block(14, 20, MARK)
    tracker = ObjectTracker(max_shift=REACH)
    for i in range(6):
        x = 120 + 6 * i
        frame = seen(THIN, (500, 500 - 10 * i), things=[(x, 196, car)])
        tracker.update(frame, hide=(x - 2, 194, 18, 24))
        assert tracker.scroll == ((0, -10) if i else (0, 0))
    assert tracker.camera == (0, -50) and tracker.read_as == (0, 5)
    # Not hidden, the car is one thing more that moved by itself.
    tracker = ObjectTracker(max_shift=REACH)
    for i in range(6):
        tracker.update(seen(THIN, (500, 500 - 10 * i), things=[(120 + 6 * i, 196, car)]))
    assert tracker.camera == (0, -50)


# ------------------------------------------------------------ not fooled


def test_nothing_changing_is_no_move_and_no_search(searched: list[str]) -> None:
    tracker = still([], [])
    assert tracker.searches == 0 and searched == []


def test_one_thing_moving_is_no_move(searched: list[str]) -> None:
    thing = block(16, 16)
    still([(100, 150, thing)], [(96, 150, thing)], [(92, 150, thing)])
    assert searched == []  # one small thing changing needs no search at all


def test_two_things_moving_alike_are_no_move() -> None:
    """Two walkers in step on a bare row, over a floor that stands: what they explain
    covers neither a quarter of the screen's rows nor a quarter of its columns."""
    thing = block(16, 16)
    still(
        [(40, 184, thing), (160, 184, thing)],
        [(36, 184, thing), (156, 184, thing)],
        [(32, 184, thing), (152, 184, thing)],
    )


def test_a_row_of_things_moving_alike_is_no_move() -> None:
    """Six in step along one row cover a quarter of the columns; but the rows they move
    in are a strip between the bar and the floor, not the larger part of the screen."""
    thing = block(16, 16)
    still(*([(20 + 36 * i - 4 * n, 184, thing) for i in range(6)] for n in range(3)))


def test_clouds_drifting_over_a_level_that_stands_still_are_no_move() -> None:
    cloud = block(40, 14, CLOUD)
    still(*([(10 + 60 * i - 3 * n, 30 + 9 * (i % 3), cloud) for i in range(4)] for n in range(4)))


@pytest.mark.parametrize("there_first", [True, False])
def test_a_wide_thing_vanishing_or_appearing_is_no_move(there_first: bool) -> None:
    looks: list[list[Thing]] = [[(60, 100, block(120, 60))], []]
    still(*(looks if there_first else looks[::-1]))


def test_a_floor_that_opens_is_no_move() -> None:
    """What an existing check caught in the first try-out: "the screen slid down" explains
    a piece of floor turning into what is above it. All that such a move explains turned
    one colour: nothing came as something went."""
    hole = np.full((40, 100, 3), NIGHT, np.uint8)
    still([], [(60, 200, hole)])
    still([(60, 200, hole)], [])


def test_a_cut_to_another_place_is_no_move() -> None:
    tracker = ObjectTracker(max_shift=REACH)
    tracker.update(seen(THIN, START))
    tracker.update(seen(place(seed=11), START))
    assert tracker.scroll == (0, 0) and not tracker.windowed and tracker.read_as == (0, 0)
    assert tracker.unexplained > 0.01  # and it knows it could not tell


@pytest.mark.parametrize(
    ("where", "move"), [("top", (0, 60)), ("top", (0, -45)), ("left", (60, 0)), ("right", (-50, 9))]
)
def test_a_move_past_the_reach_is_not_read(where: str, move: tuple[int, int]) -> None:
    tracker = moved_by(move, where=where)
    assert tracker.scroll == (0, 0) and not tracker.windowed and tracker.read_as == (0, 0)


def test_every_pixel_changing_is_no_move() -> None:
    tracker = ObjectTracker(max_shift=REACH)
    for shade in (10, 200, 90):
        a = np.full((H, W, 3), shade, np.uint8)
        bar(a, "top")
        tracker.update(Frame(W, H, 3, a.tobytes()))
        assert tracker.scroll == (0, 0) and not tracker.windowed


def test_a_bar_that_is_most_of_the_screen_is_no_strip() -> None:
    """A strip is a strip: the part that moved must be the larger part of the screen. A
    window of scenery under a panel that fills more than half of it is not read."""
    tracker = ObjectTracker(max_shift=REACH)
    for i in range(3):
        tracker.update(seen(THIN, (500, 500 + 8 * i), tall=130))
        assert tracker.scroll == (0, 0) and not tracker.windowed
    # A panel of a third of the screen is a strip still.
    tracker = ObjectTracker(max_shift=REACH)
    for i in range(3):
        tracker.update(seen(THIN, (500, 500 + 8 * i), tall=80))
    assert tracker.camera == (0, 16) and tracker.read_as == (0, 2)


# ------------------------------------------------------------ only on a game read that way


def test_under_a_bar_is_not_looked_where_the_whole_screen_has_told(searched: list[str]) -> None:
    """A game whose scenery tells by itself is not one to look under a strip in: there a
    band drifting over a still level would be the only thing such a look could find."""
    tracker = ObjectTracker(max_shift=REACH)
    for i in range(3):
        tracker.update(seen(RICH, (500, 500 + 8 * i)))
    assert tracker.read_as == (2, 0) and searched == ["whole"] * 2
    tracker.update(seen(THIN, START))  # a cut to a place with little in it
    for i in (1, 2):
        tracker.update(seen(THIN, (500, 500 + 8 * i)))
        assert tracker.scroll == (0, 0) and not tracker.windowed
    assert tracker.read_as == (2, 0)
    assert searched == ["whole"] * 5  # one search a look: under the bar was never looked


def test_a_game_read_under_a_bar_stays_so() -> None:
    """Where under the bar has told more often, a look the whole screen happens to tell
    does not end it."""
    tracker = ObjectTracker(max_shift=REACH)
    for i in range(4):
        tracker.update(seen(THIN, (500, 500 + 8 * i)))
    assert tracker.read_as == (0, 3)
    tracker.update(seen(RICH, START))
    tracker.update(seen(RICH, (500, 508)))
    assert tracker.read_as == (1, 3) and not tracker.windowed
    tracker.update(seen(THIN, START))
    tracker.update(seen(THIN, (500, 508)))
    assert tracker.scroll == (0, 8) and tracker.windowed and tracker.read_as == (1, 4)


def test_how_a_game_is_read_is_kept_over_a_go_and_handed_to_a_try() -> None:
    told = ObjectTracker(max_shift=REACH)
    for i in range(3):
        told.update(seen(RICH, (500, 500 + 8 * i)))
    told.reset()
    assert told.read_as == (2, 0) and not told.windowed
    # A new tracker reads under the bar; one that reads like ``told`` does not.
    assert moved_by((0, 8)).windowed
    like = ObjectTracker(max_shift=REACH)
    like.read_like(told)
    assert like.read_as == (2, 0)
    assert not moved_by((0, 8), tracker=like).windowed and like.scroll == (0, 0)
    under = moved_by((0, 8))
    under.reset()
    assert under.read_as == (0, 1) and not under.windowed  # the last move is forgotten
    with pytest.raises(ConfigurationError, match="another tracker"):
        like.read_like("told")  # type: ignore[arg-type]


# ------------------------------------------------------------ after it


def test_the_bar_becomes_glass_and_the_scenery_stands_still_in_the_level() -> None:
    tracker = ObjectTracker(max_shift=REACH)
    for i in range(6):
        tracker.update(seen(THIN, (500 + 3 * i, 500 + 8 * i), count=i))
        if i == 1:
            assert tracker.glass == ()  # once is not enough
    # The rows of the bar that hold the lives stood still through two moves and more.
    assert tracker.glass == tuple(range(4, 12))
    assert tracker.camera == (15, 40)
    tracks = tracker.tracks()
    # The bar is one piece of scenery, lives and all: nothing of it is a thing, and no
    # edge of it is taken for a thing moving through the level at the camera's speed.
    # (A thing that slid under the bar is followed there for a few looks, unseen.)
    assert all(t.y >= TALL for t in tracks if t.missing == 0)
    assert tracker.movers == ()
    # What was in sight at all six looks stands still in the level: where the camera
    # is read as standing, each of these would move 8 rows a look.
    scenery = [t for t in tracks if t.missing == 0 and t.seen == 6]
    assert len(scenery) >= 8
    assert all(not t.moves and max(abs(v) for v in t.velocity) < 0.5 for t in scenery)


# ------------------------------------------------------------ cost and edges


def test_a_look_under_a_bar_is_quick() -> None:
    frames = [seen(THIN, (500, 500 + 8 * i), count=i) for i in range(11)]
    tracker = ObjectTracker(max_shift=REACH)
    tracker.update(frames[0])
    began = time.perf_counter()
    for frame in frames[1:]:
        tracker.update(frame)
    took = (time.perf_counter() - began) / 10
    assert tracker.read_as == (0, 10)
    assert took < 0.25, f"a look under a bar took {took:.3f} s"


def test_a_small_screen_is_read_too() -> None:
    """A screen of 64 by 48 (the practice racer's): a quarter of it is 12 rows."""
    rng = np.random.default_rng(2)
    world = np.zeros((200, 64, 3), np.uint8)
    world[:] = NIGHT
    for _ in range(60):
        y, x = int(rng.integers(0, 190)), int(rng.integers(0, 60))
        world[y : y + 3, x : x + 2] = MARK if rng.integers(0, 2) else POST

    def small(camera: int) -> Frame:
        a = world[camera : camera + 48].copy()
        a[:6] = BAR
        a[2:4, 2:5] = LIFE
        return Frame(64, 48, 3, a.tobytes())

    tracker = ObjectTracker(max_shift=REACH)
    for i in range(5):
        tracker.update(small(100 - 3 * i))
    assert tracker.camera == (0, -12) and tracker.read_as == (0, 4)
