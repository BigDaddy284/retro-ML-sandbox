"""T-1020: the camera's move where the screen cannot tell it. The tracker is told
what the move is expected to be (from the game's memory) and takes that where the
picture has nothing better; where the picture can tell, the picture decides."""

from __future__ import annotations

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.tracking import EXPLAINS, SLACK, ObjectTracker
from tests.unit.observation.test_tracking import (
    BLACK,
    BROWN,
    SKY,
    WHITE,
    block,
    gapped,
    level,
    open_ground,
    plain,
    strip,
    view,
)

Move = tuple[int, int]
Expect = tuple[int | None, int | None] | None


def walked_past(
    world: np.ndarray, tracker: ObjectTracker, looks: int, expect: Expect
) -> list[tuple[Move, Move]]:
    """The camera moves 16 pixels a look and a walker comes the other way at 4 (as in
    T-1017's tests): the camera's move at each look as the tracker has it, and as the
    picture alone had it."""
    out = []
    for i in range(looks):
        tracker.update(view(world, 16 * i, [(640 - 4 * i, 194, block())], i), expect=expect)
        out.append((tracker.scroll, tracker.sighted))
    return out


def test_a_lone_walker_on_bare_ground() -> None:
    """One thing walking against nothing: the picture takes its 4 pixels for part of
    the camera's move (T-1017). Told the move, the tracker has it right, and the
    walker's own speed with it."""
    assert SLACK == 0.01
    looks = slice(21, 26)  # past the post, nothing but the walker in view
    alone = walked_past(open_ground(), ObjectTracker(max_shift=32), 26, None)
    assert alone[looks] == [((20, 0), (20, 0))] * 5  # as it is without being told
    for expect in ((16, None), (16, 0)):
        tracker = ObjectTracker(max_shift=32)
        told = walked_past(open_ground(), tracker, 26, expect)
        assert told[0] == ((0, 0), (0, 0))  # a first screen has nothing to compare
        assert all(scroll == (16, 0) for scroll, _ in told[1:])
        assert told[looks] == [((16, 0), (20, 0))] * 5  # what the picture said is kept
        assert tracker.camera == (16 * 25, 0)
        # Nothing of what changed is left unexplained but the walker's own few pixels.
        assert 0.0 < tracker.unexplained < SLACK
        (walker,) = [t for t in tracker.tracks() if t.missing == 0 and t.moves]
        assert abs(walker.velocity[0] + 4) < 1 and abs(walker.velocity[1]) < 1
    # Left to the picture, the walker stands still in the level and the ground runs.
    untold = ObjectTracker(max_shift=32)
    walked_past(open_ground(), untold, 26, None)
    assert not [t for t in untold.tracks() if t.missing == 0 and abs(t.velocity[0] + 4) < 1]
    # With a gap in view the picture has it right already, and being told changes nothing.
    assert (
        walked_past(gapped(), ObjectTracker(max_shift=32), 26, (16, None))[looks]
        == [((16, 0), (16, 0))] * 5
    )


def bricks(width: int = 1024) -> np.ndarray:
    """Ground of bricks 16 pixels wide under an empty sky: every 16 pixels along it
    looks the same."""
    a = np.zeros((240, width, 3), np.uint8)
    a[:] = SKY
    a[208:] = BROWN
    a[208] = BLACK
    a[208:, ::16] = BLACK
    return a


def test_where_nothing_in_view_tells() -> None:
    # Bare ground and nothing on it: the screen does not change at all.
    w = plain()
    tracker = ObjectTracker(max_shift=32)
    for i in range(6):
        tracker.update(view(w, 300 + 16 * i, [], i), expect=(16, None))
        assert tracker.sighted == (0, 0) and tracker.scroll == ((16, 0) if i else (0, 0))
        assert tracker.unexplained == 0.0
    assert tracker.camera == (80, 0) and tracker.searches == 0
    tracker.update(view(w, 380, [], 6), expect=(0, None))
    assert tracker.scroll == (0, 0) and tracker.camera == (80, 0)
    # The same with a few lives drawn along the top: they stand still on the screen
    # and are few enough to be let pass, and from the moves taken they are learnt to
    # be on the glass.
    tracker = ObjectTracker(max_shift=32)
    for i in range(4):
        tracker.update(strip(view(w, 300 + 16 * i, [], i), 3), expect=(16, None))
        assert tracker.scroll == ((16, 0) if i else (0, 0)), i
    assert tracker.glass == tuple(range(8, 16))
    # Scenery that repeats every 16 pixels: a move of 16 shows nothing, and a move of
    # 20 looks like one of 4.
    w = bricks()
    seen, told = ObjectTracker(max_shift=32), ObjectTracker(max_shift=32)
    camera = 100
    for i, step in enumerate((0, 16, 20, 8, 16, -12, 32)):
        camera += step
        frame = view(w, camera, [], i)
        seen.update(frame)
        told.update(frame, expect=(step, None))
        if i:
            assert told.scroll == (step, 0), i
            assert told.sighted == seen.scroll, i
    assert seen.camera != (80, 0) and told.camera == (80, 0)
    assert [s for s in (16, 20, -12, 32) if s % 16 == 0] == [16, 32]  # the ones not seen at all


def test_the_picture_decides_where_it_can() -> None:
    assert EXPLAINS == 0.5
    w = level()  # hills at uneven spacing: every shift but the right one fits badly
    for expect in ((8, None), (20, None), (0, None), (16, 4), (None, 4)):
        tracker = ObjectTracker(max_shift=32)
        for i in range(5):
            tracker.update(view(w, 100 + 16 * i, [], i), expect=expect)
            if i:
                assert tracker.scroll == tracker.sighted == (16, 0), expect
                assert tracker.unexplained == 0.0
    # A screen that plainly stands still, and memory says it moved.
    tracker = ObjectTracker(max_shift=32)
    for i in range(4):
        tracker.update(view(w, 100, [], i), expect=(16, None))
        assert tracker.scroll == (0, 0) and tracker.searches == 0
    # A cut to another place: no move explains it, whatever is expected, and as much
    # is left unexplained as before (what the watch knows a jolt by).
    for expect in (None, (16, None), (0, 0), (-8, 3)):
        tracker = ObjectTracker(max_shift=32)
        tracker.update(view(w, 100, [], 0), expect=expect)
        tracker.update(view(w, 116, [], 1), expect=(16, None))
        other = np.zeros((240, 1024, 3), np.uint8)  # indoors, nothing like the level
        other[60:180:40, 30::70] = WHITE
        other[200:] = (60, 60, 60)
        tracker.update(view(other, 500, [], 2), expect=expect)
        assert tracker.scroll == tracker.sighted == (0, 0), expect
        assert tracker.unexplained > 0.5 and tracker.camera == (16, 0)
    # A move further than the camera may go between two looks is not taken on trust.
    near = ObjectTracker(max_shift=12)
    bare = plain()
    for i, (step, taken) in enumerate(((0, 0), (12, 12), (13, 0), (-12, -12), (-16, 0))):
        near.update(view(bare, 300 + 20 * i, [], i), expect=(step, None))
        assert near.scroll == (taken, 0), step
    # Up and down on bare ground the picture can tell: the grass would have moved.
    tracker = ObjectTracker(max_shift=32)
    for i in range(3):
        tracker.update(view(bare, 300, [], i), expect=(None, 5))
        assert tracker.scroll == (0, 0)


def test_what_is_expected_is_checked() -> None:
    w = open_ground()
    plainly, with_none, each_none = (ObjectTracker(max_shift=32) for _ in range(3))
    for i in range(26):
        frame = view(w, 16 * i, [(640 - 4 * i, 194, block())], i)
        plainly.update(frame)
        with_none.update(frame, expect=None)
        each_none.update(frame, expect=(None, None))
        assert plainly.scroll == with_none.scroll == each_none.scroll
        assert plainly.sighted == plainly.scroll
    assert plainly.camera == with_none.camera == each_none.camera
    tracker = ObjectTracker(max_shift=32)
    frame = view(w, 0, [], 0)
    for nonsense in (16, (16,), (16, 0, 0), ("16", 0), (1.5, 0), (True, 0), [16, 0], (0, "0")):
        with pytest.raises(ConfigurationError, match="expect"):
            tracker.update(frame, expect=nonsense)  # type: ignore[arg-type]
    assert tracker.tracks() == () and tracker.sighted == (0, 0)  # nothing was taken in
    tracker.update(frame, expect=(16, 0))
    assert tracker.scroll == tracker.sighted == (0, 0)  # a first screen
    tracker.update(view(plain(), 300, [], 1), expect=(16, None))
    tracker.reset()
    assert tracker.sighted == (0, 0)
    assert isinstance(frame, Frame)
