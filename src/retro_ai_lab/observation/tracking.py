"""Following the things on the screen from one frame to the next (T-972).

Two questions per frame pair, both answered without help:

* **How far did the screen scroll?** A platformer's camera follows the
  player, so the whole level slides past. :func:`estimate_scroll` tries each
  small shift and keeps the one under which the most of the new frame equals
  the old one (the smallest shift wins a tie, so a plain sky reads as no
  scroll).
* **What moved in the level?** With the scroll taken away, the pixels that
  still differ belong to things moving by themselves or being moved: the
  player, enemies, a coin popping up. This also cuts a walker out of the
  ground it stands on, which a single still screen cannot (T-971's limit).

An :class:`ObjectTracker` gives each thing a :class:`Track` that lasts while
it stays on screen: where it is, its kind when the catalogue knows it, and
its velocity *in the level* (screen movement plus scroll), so a coin that
slides past as the camera moves reads as standing still and a walker reads
as walking.

Since T-992 the tracker's camera is measured at the pace the AI looks, and
only by what can tell: see :meth:`ObjectTracker.update`.

Since T-1016 a thing cut out of the scenery by its motion is filled out by its
colours to the whole thing (what differs between two screens is only the edge
a flat-coloured thing leads with), taken only if it stood elsewhere a look
ago, and its kind is known by those colours from then on: every thing of the
kind is seen at every look, moving or not. ``ObjectTracker(whole=False)`` is
the tracker without that.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.objects import (
    DEFAULT_MAX_SIZE,
    DEFAULT_MIN_PIXELS,
    PALETTE_SHARE,
    ObjectCatalogue,
    ScreenObject,
    _dilate,
    _label,
)
from retro_ai_lab.observation.screen import to_array

DEFAULT_MAX_SHIFT = 8  # pixels a frame the camera may move either way
# Every shift is tried while that costs no more than 8 pixels of reach either way on a
# 256 by 240 screen did; past that, coarse then fine (T-992).
EXHAUSTIVE = 17 * 17 * 256 * 240
COARSE = 4  # the coarse search looks at every fourth pixel and tries every fourth shift
KEEP = 3  # coarse answers looked at closely
GLASS_AFTER = 2  # camera moves a row stood still through before it counts as glass
GLASS_MOST = 4  # ... and the most that are counted, so a row can be unlearnt again
SCREEN_SHARE = 4  # the camera's reach is at most a quarter of the screen either way
EXPLAINS = 0.5  # a move must leave at most this share of what differed still differing
# A move the caller expects is taken where it leaves at most this share of the pixels
# with a say differing more than the picture's own answer does (T-1020): a thing or
# two that moved by themselves, not scenery that tells.
SLACK = 0.01
MOST_MOVERS = 16  # kinds known by their colours at most: each costs a pass over the screen
FITS_MOST = 4_000_000  # pixels compared at most to see whether a patch stood elsewhere (T-1016)
DEFAULT_MAX_JUMP = 24  # farthest a thing may be from where it was expected to be
DEFAULT_MAX_MISSING = 5  # frames a track survives without being seen
HISTORY = 30  # positions kept per track


def _packed(frame: Frame) -> NDArray[np.int32]:
    """One int per pixel (the channels packed), so comparing pixels is one comparison."""
    arr = to_array(frame).astype(np.int32)
    out = arr[..., 0].copy()
    for ch in range(1, frame.channels):
        out = (out << 8) | arr[..., ch]
    return out


def _check_pair(before: Frame, after: Frame) -> None:
    if (before.width, before.height, before.channels) != (
        after.width,
        after.height,
        after.channels,
    ):
        raise ConfigurationError("Both frames must have the same size and channels.")


_Pixels = TypeVar("_Pixels", NDArray[np.int32], NDArray[np.bool_])


def _overlap(a: _Pixels, b: _Pixels, dx: int, dy: int) -> tuple[_Pixels, _Pixels]:
    """The parts of ``a`` (after) and ``b`` (before) showing one spot: a[y,x] ~ b[y+dy,x+dx]."""
    h, w = a.shape
    ay0, ay1 = max(0, -dy), min(h, h - dy)
    ax0, ax1 = max(0, -dx), min(w, w - dx)
    return a[ay0:ay1, ax0:ax1], b[ay0 + dy : ay1 + dy, ax0 + dx : ax1 + dx]


def _mismatch(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
    va: NDArray[np.bool_] | None,
    vb: NDArray[np.bool_] | None,
    dx: int,
    dy: int,
) -> float | None:
    """The share of pixels that differ with ``a`` laid over ``b`` moved by the shift,
    among those both masks let vote; None when none do."""
    aa, bb = _overlap(a, b, dx, dy)
    if aa.size == 0:
        return None
    differ = aa != bb
    if va is None and vb is None:
        return float(np.count_nonzero(differ)) / aa.size
    ok = np.ones(aa.shape, np.bool_)
    if va is not None:
        ok &= _overlap(va, va, dx, dy)[0]
    if vb is not None:
        ok &= _overlap(vb, vb, dx, dy)[1]
    votes = int(np.count_nonzero(ok))
    if votes == 0:
        return None
    return float(np.count_nonzero(differ & ok)) / votes


_Key = tuple[float, int, int, int]


def _rank(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
    va: NDArray[np.bool_] | None,
    vb: NDArray[np.bool_] | None,
    shifts: Iterable[tuple[int, int]],
) -> list[_Key]:
    """The shifts, best first: least mismatch, then the smallest move."""
    keys: list[_Key] = []
    for dx, dy in shifts:
        mismatch = _mismatch(a, b, va, vb, dx, dy)
        if mismatch is not None:
            keys.append((round(mismatch, 6), abs(dx) + abs(dy), dy, dx))
    keys.sort()
    return keys


def _search(
    a: NDArray[np.int32],
    b: NDArray[np.int32],
    va: NDArray[np.bool_] | None,
    vb: NDArray[np.bool_] | None,
    max_dx: int,
    max_dy: int,
) -> tuple[int, int]:
    """:func:`estimate_scroll` on packed screens and masks of the pixels that vote."""
    if (2 * max_dx + 1) * (2 * max_dy + 1) * a.size <= EXHAUSTIVE:
        shifts = {
            (dx, dy) for dy in range(-max_dy, max_dy + 1) for dx in range(-max_dx, max_dx + 1)
        }
    else:
        # Every fourth pixel, every fourth shift, over the whole reach; then every
        # shift round the best few. A reach of 32 costs less than 8 does in full.
        c = COARSE
        kx, ky = -(-max_dx // c), -(-max_dy // c)
        coarse = _rank(
            a[::c, ::c],
            b[::c, ::c],
            None if va is None else va[::c, ::c],
            None if vb is None else vb[::c, ::c],
            ((x, y) for y in range(-ky, ky + 1) for x in range(-kx, kx + 1)),
        )
        shifts = {(0, 0)}
        for key in coarse[:KEEP]:
            y, x = key[2], key[3]
            for dy in range(c * y - c + 1, c * y + c):
                for dx in range(c * x - c + 1, c * x + c):
                    if abs(dx) <= max_dx and abs(dy) <= max_dy:
                        shifts.add((dx, dy))
    ranked = _rank(a, b, va, vb, shifts)
    if not ranked:
        return (0, 0)
    return ranked[0][3], ranked[0][2]


def _votes(mask: object, frame: Frame, what: str) -> NDArray[np.bool_] | None:
    """A leave-out mask turned into the pixels that vote; None when nothing is left out."""
    if mask is None:
        return None
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.shape != (frame.height, frame.width)
    ):
        raise ConfigurationError(f"{what} must be a mask of true and false, one for each pixel.")
    return ~mask


def estimate_scroll(
    before: Frame,
    after: Frame,
    *,
    max_dx: int = DEFAULT_MAX_SHIFT,
    max_dy: int = DEFAULT_MAX_SHIFT,
    leave_out_before: NDArray[np.bool_] | None = None,
    leave_out_after: NDArray[np.bool_] | None = None,
) -> tuple[int, int]:
    """How far the camera moved: ``(dx, dy)`` with ``after[y, x] == before[y+dy, x+dx]``.

    Positive ``dx`` is the camera moving right (the level sliding left). Pixels true
    in a ``leave_out`` mask (one for each screen, height by width) have no say: what
    is known not to move with the scenery, such as the player the camera holds still
    (T-992). Every shift is tried while that is cheap (``EXHAUSTIVE``: a small reach, or
    a small screen); past that the search is coarse, then fine round the best few, and
    can miss what only features thinner than ``COARSE`` pixels tell.
    """
    _check_pair(before, after)
    if not isinstance(max_dx, int) or not isinstance(max_dy, int) or max_dx < 0 or max_dy < 0:
        raise ConfigurationError("The largest shift must be a whole number, 0 or more.")
    max_dx = min(max_dx, before.width - 1)
    max_dy = min(max_dy, before.height - 1)
    vb = _votes(leave_out_before, before, "leave_out_before")
    va = _votes(leave_out_after, after, "leave_out_after")
    return _search(_packed(after), _packed(before), va, vb, max_dx, max_dy)


def moving_mask(before: Frame, after: Frame, scroll: tuple[int, int]) -> NDArray[np.bool_]:
    """Pixels of ``after`` that differ from ``before`` once the scroll is taken away.

    The strip the scroll brought into view has nothing to compare with and is
    left out.
    """
    _check_pair(before, after)
    dx, dy = scroll
    a, b = _packed(after), _packed(before)
    out = np.zeros(a.shape, bool)
    h, w = a.shape
    if abs(dx) >= w or abs(dy) >= h:
        return out
    aa, bb = _overlap(a, b, dx, dy)
    ay0, ax0 = max(0, -dy), max(0, -dx)
    out[ay0 : ay0 + aa.shape[0], ax0 : ax0 + aa.shape[1]] = aa != bb
    return out


@dataclass(slots=True)
class Track:
    """One thing followed over frames."""

    id: int
    kind: str | None
    x: int
    y: int
    width: int
    height: int
    first_frame: int
    last_frame: int
    velocity: tuple[float, float] = (0.0, 0.0)  # pixels a frame in the level
    missing: int = 0  # frames since last seen
    seen: int = 1
    history: list[tuple[int, int, int]] = field(default_factory=list)  # (frame, level x, level y)
    merged: bool = False  # this frame it was part of a patch with another thing (touching)
    glass: bool = False  # drawn on the glass (a life counter): it stays put on the screen
    cut_out: bool = False  # last seen cut out of the scenery by its motion (T-977)
    # The colours (packed) it was filled out by when it was cut out of the scenery
    # (T-1016): by these it is seen whole, and seen while it stands still.
    colours: tuple[int, ...] = ()

    @property
    def centre(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def moves(self) -> bool:
        """It has moved in the level (not just slid past with the camera)."""
        if len(self.history) < 2:
            return False
        xs = [h[1] for h in self.history]
        ys = [h[2] for h in self.history]
        return max(xs) - min(xs) > 1 or max(ys) - min(ys) > 1


def _inside(t: Track, o: ScreenObject, sx: int, sy: int, slack: int = 3) -> bool:
    """Where ``t`` should be now (moved, slid by the scroll) lies within the patch ``o``."""
    x = t.x + t.velocity[0] - sx
    y = t.y + t.velocity[1] - sy
    return (
        x >= o.x - slack
        and y >= o.y - slack
        and x + t.width <= o.x + o.width + slack
        and y + t.height <= o.y + o.height + slack
    )


class ObjectTracker:
    """Gives each thing on the screen a track that lasts while it is on screen."""

    def __init__(
        self,
        catalogue: ObjectCatalogue | None = None,
        *,
        max_shift: int = DEFAULT_MAX_SHIFT,
        max_jump: int = DEFAULT_MAX_JUMP,
        max_missing: int = DEFAULT_MAX_MISSING,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_size: int = DEFAULT_MAX_SIZE,
        whole: bool = True,
    ) -> None:
        if not isinstance(whole, bool):
            raise ConfigurationError("whole must be true or false.")
        for value, what in (
            (max_shift, "max_shift"),
            (max_jump, "max_jump"),
            (max_missing, "max_missing"),
        ):
            if not isinstance(value, int) or value < 0:
                raise ConfigurationError(f"{what} must be a whole number, 0 or more.")
        self._catalogue = (
            catalogue
            if catalogue is not None
            else ObjectCatalogue(min_pixels=min_pixels, max_size=max_size)
        )
        self._max_shift = max_shift
        self._max_jump = max_jump
        self._max_missing = max_missing
        self._min_pixels = min_pixels
        self._max_size = max_size
        # Whether a thing cut out of the scenery by its motion is filled out by its
        # colours and found again by them (T-1016).
        self._whole = whole
        # The colours of each kind seen to move, and the kind. Kept over a reset, like
        # the kinds: what an enemy looks like is the same every go.
        self._movers: dict[tuple[int, ...], str] = {}
        self._previous: Frame | None = None
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._camera = (0, 0)  # where the camera is in the level, summed scrolls
        self._scroll = (0, 0)
        self._sighted = (0, 0)  # the camera's last move as the picture alone had it (T-1020)
        self._hidden: tuple[int, int, int, int] | None = None  # the box left out last frame
        # Camera moves each row stood still through although the move would have changed
        # it (T-992). Kept over a reset, like the kinds: the strip of lives and score is
        # in the same place every go.
        self._stuck: NDArray[np.int16] | None = None
        self._searches = 0
        self._unexplained = 0.0

    @property
    def catalogue(self) -> ObjectCatalogue:
        return self._catalogue

    @property
    def scroll(self) -> tuple[int, int]:
        """The camera's move between the last two frames."""
        return self._scroll

    @property
    def camera(self) -> tuple[int, int]:
        """The camera's position in the level, counted from the first frame."""
        return self._camera

    @property
    def glass(self) -> tuple[int, ...]:
        """The rows found to be painted on the glass: they stand still while the camera
        moves (the lives, the score), so they have no say in how far it moved."""
        if self._stuck is None:
            return ()
        return tuple(int(y) for y in np.flatnonzero(self._stuck >= GLASS_AFTER))

    @property
    def movers(self) -> tuple[str, ...]:
        """The kinds seen to move and known by their colours since (T-1016): things of
        these kinds are seen whether or not they move."""
        return tuple(dict.fromkeys(self._movers.values()))

    @property
    def searches(self) -> int:
        """Looks in which the camera's move had to be searched for (a still screen and a
        change no bigger than one small thing need no search)."""
        return self._searches

    @property
    def sighted(self) -> tuple[int, int]:
        """The camera's move at the last look as the picture alone had it (T-1020):
        :attr:`scroll` unless a move the caller expected was taken instead."""
        return self._sighted

    @property
    def unexplained(self) -> float:
        """The share of the pixels with a say that still differed at the last look once
        the camera's move was allowed for (T-993): next to nothing after a real move,
        most of the screen after a cut to another place. 0 when the move needed no
        search (a still screen, one small thing changing) and on a first frame."""
        return self._unexplained

    def tracks(self) -> tuple[Track, ...]:
        """The things on screen now (seen this frame or missing only briefly)."""
        return tuple(sorted(self._tracks.values(), key=lambda t: t.id))

    def reset(self) -> None:
        """Forget everything followed (a new go, a loaded state); kinds are kept."""
        self._previous = None
        self._tracks.clear()
        self._camera = (0, 0)
        self._scroll = (0, 0)
        self._sighted = (0, 0)
        self._hidden = None
        self._unexplained = 0.0

    def update(
        self,
        frame: Frame,
        *,
        hide: tuple[int, int, int, int] | None = None,
        expect: tuple[int | None, int | None] | None = None,
    ) -> tuple[Track, ...]:
        """Follow the things on ``frame``. ``hide`` is a box (x, y, width, height) to
                leave out of the picture: nothing in it is a thing, and neither is the change
                it leaves behind when it moves (T-987: the player, which the caller knows by
                its looks and follows itself).

                The camera's move since the last frame (T-992) is voted on by the pixels that
        can tell: not the hidden box, here or where it was (the camera holds the player
        still on the screen), and not the rows found to be glass. And one small thing
        changing is no camera move: when all that changed fits in half a thing's
        largest size, the camera stood still. (Things known to move are not left out:
        a wrong scroll makes the scenery's own edges look like movers, and leaving
        those out would keep the scroll wrong.)

        ``expect`` is the camera's move since the last frame as the caller has it from
        somewhere that is not the picture (T-1020: the game's own count of where its
        camera is, :class:`~retro_ai_lab.observation.camera_memory.CameraMemory`):
        across and down, each a whole number or None for "not known". One thing
        walking on bare ground, a screen on which nothing tells, scenery that repeats:
        there the picture cannot say how far the camera moved (T-1017), and what is
        expected is taken, if the new screen laid over the old one that way differs in
        at most :data:`SLACK` more of the pixels with a say than under the picture's
        own answer, and it explains what changed as a move must (:data:`EXPLAINS`)
        unless next to nothing changed at all. So scenery that tells, a cut to another
        place and a move past the camera's reach are still the picture's to judge.
        :attr:`sighted` is what the picture alone said."""
        if not isinstance(frame, Frame):
            raise ConfigurationError("update() needs a Frame.")
        if expect is not None and (
            not isinstance(expect, tuple)
            or len(expect) != 2
            or any(
                v is not None and (isinstance(v, bool) or not isinstance(v, int)) for v in expect
            )
        ):
            raise ConfigurationError(
                "expect must be the camera's move across and down: each a whole number, or None."
            )
        if hide is not None and (
            len(hide) != 4
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in hide)
            or hide[2] < 1
            or hide[3] < 1
        ):
            raise ConfigurationError("hide must be a box: x, y, width, height.")
        prev = self._previous
        if prev is not None and (prev.width, prev.height, prev.channels) != (
            frame.width,
            frame.height,
            frame.channels,
        ):
            self.reset()
            self._stuck = None
            prev = None
        objects = [o for o in self._catalogue.observe(frame, hide=hide) if not o.scenery]
        if self._whole:
            objects.extend(self._known_by_their_colours(frame, objects, hide))
        if prev is not None:
            self._scroll = self._camera_move(prev, frame, hide, expect)
            self._camera = (self._camera[0] + self._scroll[0], self._camera[1] + self._scroll[1])
            objects.extend(self._movers_in_scenery(prev, frame, objects, hide))
        else:
            self._scroll = self._sighted = (0, 0)
            self._unexplained = 0.0
        self._match(objects, frame.number)
        self._previous = frame
        self._hidden = hide
        return self.tracks()

    # ------------------------------------------------------------ internals

    def _camera_move(
        self,
        prev: Frame,
        frame: Frame,
        hide: tuple[int, int, int, int] | None,
        expect: tuple[int | None, int | None] | None = None,
    ) -> tuple[int, int]:
        a, b = _packed(frame), _packed(prev)
        self._unexplained = 0.0
        if self._stuck is None:
            self._stuck = np.zeros(frame.height, np.int16)
        out_a = np.zeros(a.shape, np.bool_)  # what has no say, on the new screen ...
        out_b = np.zeros(a.shape, np.bool_)  # ... and on the old one

        for mask, box in ((out_a, hide), (out_b, self._hidden)):
            if box is not None:
                x, y, w, h = box
                mask[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)] = True
        changed = (a != b) & ~(out_a | out_b)  # with no shift, what either screen leaves out
        glass = self._stuck >= GLASS_AFTER
        va, vb = ~(out_a | glass[:, None]), ~(out_b | glass[:, None])
        reach = (
            min(self._max_shift, frame.width // SCREEN_SHARE),
            min(self._max_shift, frame.height // SCREEN_SHARE),
        )
        scroll = self._sighted = self._sighted_move(a, b, va, vb, changed & ~glass[:, None], reach)
        told = self._told(a, b, va, vb, scroll, expect, reach)
        if told is not None:
            scroll = told
        if scroll != (0, 0):
            self._learn_glass(a, b, changed, out_a, out_b, scroll)
        return scroll

    def _sighted_move(
        self,
        a: NDArray[np.int32],
        b: NDArray[np.int32],
        va: NDArray[np.bool_],
        vb: NDArray[np.bool_],
        telling: NDArray[np.bool_],
        reach: tuple[int, int],
    ) -> tuple[int, int]:
        """The camera's move as the picture alone has it (T-992)."""
        if not telling.any():
            return (0, 0)
        ys, xs = np.nonzero(telling)
        height, width = a.shape
        # Small, and far, are measured against the screen: on a small screen a thing
        # is smaller and a move past a quarter of it leaves too little to compare.
        thin = min(self._max_size // 2, width // (2 * SCREEN_SHARE))
        tall = min(self._max_size // 2, height // (2 * SCREEN_SHARE))
        if int(np.max(xs) - np.min(xs)) < thin and int(np.max(ys) - np.min(ys)) < tall:
            return (0, 0)  # one small thing changed: that is no camera move
        self._searches += 1
        scroll = _search(a, b, va, vb, *reach)
        still = _mismatch(a, b, va, vb, 0, 0)
        if scroll == (0, 0):
            self._unexplained = still or 0.0
            return scroll
        # The best of a bad lot is no measure: after a cut to another place, or a move
        # past the reach, some shift still fits best. A real move explains most of
        # what differed.
        moved = _mismatch(a, b, va, vb, *scroll)
        if still is None or moved is None or moved > EXPLAINS * still:
            self._unexplained = still or 0.0
            return (0, 0)
        self._unexplained = moved
        return scroll

    def _told(
        self,
        a: NDArray[np.int32],
        b: NDArray[np.int32],
        va: NDArray[np.bool_],
        vb: NDArray[np.bool_],
        sighted: tuple[int, int],
        expect: tuple[int | None, int | None] | None,
        reach: tuple[int, int],
    ) -> tuple[int, int] | None:
        """The move the caller expects, where it is to be taken over the picture's own
        (T-1020); None where the picture stands."""
        if expect is None:
            return None
        told = (
            sighted[0] if expect[0] is None else expect[0],
            sighted[1] if expect[1] is None else expect[1],
        )
        if told == sighted or abs(told[0]) > reach[0] or abs(told[1]) > reach[1]:
            return None
        still = _mismatch(a, b, va, vb, 0, 0)
        if still is None:
            return None  # nothing on the screen has a say: nothing to check it against
        here = still if sighted == (0, 0) else _mismatch(a, b, va, vb, *sighted)
        there = still if told == (0, 0) else _mismatch(a, b, va, vb, *told)
        if here is None or there is None:
            return None
        # No worse than the picture's own answer by more than a thing or two moving by
        # themselves; and a move at all: after a cut to another place nothing fits, and
        # what fits least badly is no measure.
        if there > here + SLACK or there > max(EXPLAINS * still, SLACK):
            return None
        self._unexplained = there
        return told

    def _learn_glass(
        self,
        a: NDArray[np.int32],
        b: NDArray[np.int32],
        changed: NDArray[np.bool_],
        out_a: NDArray[np.bool_],
        out_b: NDArray[np.bool_],
        scroll: tuple[int, int],
    ) -> None:
        """The camera moved. A row stood still if it holds something (it is not one
        colour from end to end), nothing in it changed, and the move would have changed
        it both ways: what the move would bring into the row is not there, and what the
        row held has not gone where the move would take it. (One way is not enough: a
        line running down the screen "should" be where the strip above it is, yet did
        not stand still. Where one of the two lies off the screen, the other decides.)
        Once more and the row is glass. A row that changed and is wholly as the move
        says moved with the scenery: it is not glass. Any other row loses one count,
        so a row counted by mistake is let go again by the moves that follow."""
        assert self._stuck is not None
        dx, dy = scroll
        height = a.shape[0]
        aa, bb = _overlap(a, b, dx, dy)
        ok = _overlap(~out_a, ~out_a, dx, dy)[0] & _overlap(~out_b, ~out_b, dx, dy)[1]
        differ = np.asarray(((aa != bb) & ok).any(axis=1), np.bool_)
        rows = len(differ)
        top, source = max(0, -dy), max(0, dy)
        # What the move brings to a row is not there / what it held is not where the
        # move takes it; and whether the screen shows enough to say.
        not_come, not_gone = np.zeros(height, np.bool_), np.zeros(height, np.bool_)
        come_seen, gone_seen = np.zeros(height, np.bool_), np.zeros(height, np.bool_)
        not_come[top : top + rows], come_seen[top : top + rows] = differ, True
        not_gone[source : source + rows], gone_seen[source : source + rows] = differ, True
        row_changed = np.asarray(changed.any(axis=1), np.bool_)
        holds = np.asarray((a != a[:, :1]).any(axis=1), np.bool_)
        stood = (
            ~row_changed
            & holds
            & (not_come | ~come_seen)
            & (not_gone | ~gone_seen)
            & (come_seen | gone_seen)
        )
        self._stuck[stood] = np.minimum(self._stuck[stood] + 1, GLASS_MOST)
        self._stuck[~stood] = np.maximum(self._stuck[~stood] - 1, 0)
        self._stuck[row_changed & come_seen & ~not_come] = 0

    def _on_glass(self, y: int, height: int) -> bool:
        """Every row of a thing this tall at ``y`` is glass."""
        if self._stuck is None:
            return False
        rows = self._stuck[max(0, y) : max(0, y + height)]
        return bool(rows.size) and bool((rows >= GLASS_AFTER).all())

    def _same_colours(
        self,
        packed: NDArray[np.int32],
        colours: tuple[int, ...],
        window: tuple[int, int, int, int],
        hide: tuple[int, int, int, int] | None,
    ) -> list[tuple[tuple[int, int, int, int], NDArray[np.bool_]]]:
        """The patches drawn in ``colours`` within ``window`` (x0, y0, x1, y1): the box
        of each on the screen and its own pixels in that box. Parts a pixel apart are
        one patch; what the caller hides is in none; specks are left out."""
        x0, y0, x1, y1 = window
        same = np.isin(packed[y0:y1, x0:x1], np.array(colours, np.int32))
        if hide is not None:
            hx, hy, hw, hh = hide
            same[max(0, hy - y0) : max(0, hy + hh - y0), max(0, hx - x0) : max(0, hx + hw - x0)] = (
                False
            )
        labels, count = _label(_dilate(same))
        out: list[tuple[tuple[int, int, int, int], NDArray[np.bool_]]] = []
        for i in range(1, count + 1):
            patch = (labels == i) & same
            ys, xs = np.nonzero(patch)
            if len(ys) < self._min_pixels:
                continue
            bx, by = int(np.min(xs)), int(np.min(ys))
            w, h = int(np.max(xs)) - bx + 1, int(np.max(ys)) - by + 1
            out.append(((x0 + bx, y0 + by, w, h), patch[by : by + h, bx : bx + w]))
        return out

    @staticmethod
    def _window(
        box: tuple[int, int, int, int], reach: int, frame: Frame
    ) -> tuple[int, int, int, int]:
        """``box`` grown by ``reach`` each way, as far as the screen goes."""
        x, y, w, h = box
        return (
            max(0, x - reach),
            max(0, y - reach),
            min(frame.width, x + w + reach),
            min(frame.height, y + h + reach),
        )

    def _filled_out(
        self,
        packed: NDArray[np.int32],
        own: NDArray[np.bool_],
        frame: Frame,
        hide: tuple[int, int, int, int] | None,
    ) -> tuple[tuple[int, int, int, int], NDArray[np.bool_], tuple[int, ...]] | None:
        """A moving patch filled out by its colours (T-1016): its box, its own pixels
        in that box, and the colours.

        What changed between two screens is only the part of a mover that does not
        look the same where it was: of a thing drawn in flat colour, the edge it leads
        with. ``own`` are that part's own pixels (the backdrop left out). The whole
        thing is what is drawn in the same colours and joined to them. None when that
        is bigger than a thing may be: its colours run on into the scenery, and the
        part that moved is all that can be told apart.
        """
        values, counts = np.unique(packed[own], return_counts=True)
        keep = values[(counts >= 2) & (counts >= PALETTE_SHARE * int(counts.sum()))]
        if keep.size == 0:
            return None
        colours = tuple(int(v) for v in keep)
        ys, xs = np.nonzero(own)
        x, y = int(np.min(xs)), int(np.min(ys))
        seed = (x, y, int(np.max(xs)) - x + 1, int(np.max(ys)) - y + 1)
        window = self._window(seed, self._max_size, frame)
        x0, y0, x1, y1 = x, y, x + seed[2], y + seed[3]
        joined = False
        for (bx, by, w, h), patch in self._same_colours(packed, colours, window, hide):
            # The patches the part that moved is in become one thing.
            sy0, sy1, sx0, sx1 = max(by, y), min(by + h, y1), max(bx, x), min(bx + w, x1)
            if sy0 >= sy1 or sx0 >= sx1:
                continue
            if not (own[sy0:sy1, sx0:sx1] & patch[sy0 - by : sy1 - by, sx0 - bx : sx1 - bx]).any():
                continue
            x0, y0, x1, y1 = min(x0, bx), min(y0, by), max(x1, bx + w), max(y1, by + h)
            joined = True
        if not joined or x1 - x0 > self._max_size or y1 - y0 > self._max_size:
            return None
        body = np.isin(packed[y0:y1, x0:x1], np.array(colours, np.int32))
        return (x0, y0, x1 - x0, y1 - y0), body, colours

    def _was_elsewhere(
        self,
        a: NDArray[np.int32],
        b: NDArray[np.int32],
        box: tuple[int, int, int, int],
        body: NDArray[np.bool_],
    ) -> bool:
        """Whether the patch ``body`` at ``box`` on the new screen ``a`` stood somewhere
        else on the old one ``b`` (T-1016): laid over the old screen, with the camera's
        move allowed for, it fits better a little way off than where it is now, and that
        explains most of what differed (:data:`EXPLAINS`). Scenery that something walked
        in front of fits best where it is. Where the old screen cannot say (off its
        edge, or what the caller hid on it) nothing counts either way, and a place where
        it can say for less than half the patch is no answer.
        """
        sx, sy = self._scroll
        x, y, w, h = box
        # As far as a thing may jump, or less for a big patch: the cost is the patch
        # times the places tried.
        reach = self._max_jump
        while reach > 1 and (2 * reach + 1) ** 2 * w * h > FITS_MOST:
            reach -= 1
        unknown = np.iinfo(np.int64).min
        under = np.full((h + 2 * reach, w + 2 * reach), unknown, np.int64)
        ox, oy = x - reach + sx, y - reach + sy  # the old screen's pixel under its corner
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(b.shape[1], ox + under.shape[1]), min(b.shape[0], oy + under.shape[0])
        if x0 >= x1 or y0 >= y1:
            return False
        old = b[y0:y1, x0:x1].astype(np.int64)
        if self._hidden is not None:
            hx, hy, hw, hh = self._hidden
            old[max(0, hy - y0) : max(0, hy + hh - y0), max(0, hx - x0) : max(0, hx + hw - x0)] = (
                unknown
            )
        under[y0 - oy : y1 - oy, x0 - ox : x1 - ox] = old
        places = np.lib.stride_tricks.sliding_window_view(under, (h, w))
        known = (places != unknown) & body
        seen = known.sum(axis=(2, 3))
        differ = ((places != a[y : y + h, x : x + w]) & known).sum(axis=(2, 3))
        share = np.where(
            2 * seen >= int(np.count_nonzero(body)), differ / np.maximum(seen, 1), np.inf
        )
        here = float(share[reach, reach])
        share[reach, reach] = np.inf
        there = float(share.min())
        return bool(np.isfinite(here)) and here > 0 and there <= EXPLAINS * here

    def _known_by_their_colours(
        self,
        frame: Frame,
        things: list[ScreenObject],
        hide: tuple[int, int, int, int] | None,
    ) -> list[ScreenObject]:
        """Every patch on the screen drawn in the colours of a kind once seen to move
        (T-1016), whether or not it moves now: a mover that stops is still there, so
        is one whose move the camera's measure took for the camera's own, and another
        of its kind is known the moment it comes into view. Not what is a thing on the
        still screen already, nor a patch bigger than a thing may be."""
        if not self._movers:
            return []
        packed = _packed(frame)
        screen = (0, 0, frame.width, frame.height)
        found: list[ScreenObject] = []
        for colours, kind in self._movers.items():
            for (x, y, w, h), patch in self._same_colours(packed, colours, screen, hide):
                if w > self._max_size or h > self._max_size:
                    continue
                cx, cy = x + w // 2, y + h // 2
                if any(o.x <= cx < o.x + o.width and o.y <= cy < o.y + o.height for o in things):
                    continue  # clear of the scenery it is a thing on the still screen already
                pixels = int(np.count_nonzero(patch))
                found.append(ScreenObject(x, y, w, h, pixels, False, kind, True, colours))
        return found

    def _movers_in_scenery(
        self,
        prev: Frame,
        frame: Frame,
        things: list[ScreenObject],
        hide: tuple[int, int, int, int] | None = None,
    ) -> list[ScreenObject]:
        """Moving patches not already covered by a thing: a walker on the ground."""
        moved = moving_mask(prev, frame, self._scroll)
        if self._stuck is not None:
            moved[self._stuck >= GLASS_AFTER] = False  # what is on the glass moves with nothing
        sx, sy = self._scroll
        # What is already a thing needs no second look, and neither does the spot a
        # thing of a known kind has just left (its trail is a change too). A mover cut
        # out of the scenery keeps its trail: its patch is where it was and is, and
        # the middle of that moves as fast as it does.
        boxes = [(o.x, o.y, o.width, o.height) for o in things]
        boxes += [
            (t.x - sx, t.y - sy, t.width, t.height)
            for t in self._tracks.values()
            if t.kind is not None and (not t.cut_out or t.colours)
        ]
        # What the caller hides is no thing, here or where it was a frame ago.
        if hide is not None:
            boxes.append(hide)
        if self._hidden is not None:
            hx, hy, hw, hh = self._hidden
            boxes.append((hx - sx, hy - sy, hw, hh))
        for x, y, w, h in boxes:
            moved[max(0, y - 1) : max(0, y + h + 1), max(0, x - 1) : max(0, x + w + 1)] = False
        if not moved.any():
            return []
        labels, count = _label(_dilate(moved))
        arr = to_array(frame)
        packed, before = _packed(frame), _packed(prev)
        bg = self._catalogue.background
        found: list[ScreenObject] = []
        for i in range(1, count + 1):
            region = (labels == i) & moved
            ys, xs = np.nonzero(region)
            if len(ys) < self._min_pixels:
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            w, h = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
            if w > self._max_size or h > self._max_size:
                continue  # a big change (a screen flash, a new room) is not a thing
            # Its kind from the colours now drawn where it moved, the backdrop left out
            # (its trail shows the sky): a walker on the ground has a kind too (T-977).
            crop = arr[y0 : y0 + h, x0 : x0 + w]
            mask = region[y0 : y0 + h, x0 : x0 + w].copy()
            if bg is not None:
                mask &= (crop != np.array(bg, np.uint8)).any(axis=2)
            whole = None
            if self._whole and mask.any():
                own = np.zeros(region.shape, np.bool_)
                own[y0 : y0 + h, x0 : x0 + w] = mask
                whole = self._filled_out(packed, own, frame, hide)
            if whole is None or not self._was_elsewhere(packed, before, whole[0], whole[1]):
                kind = self._catalogue.classify(crop, mask)
                found.append(ScreenObject(x0, y0, w, h, len(ys), False, kind, cut_out=True))
                continue
            (bx, by, bw, bh), body, colours = whole
            kind = self._catalogue.classify(arr[by : by + bh, bx : bx + bw], body)
            if kind is not None and len(self._movers) < MOST_MOVERS:
                self._movers.setdefault(colours, kind)  # from now on known by its colours
            thing = ScreenObject(
                bx, by, bw, bh, int(np.count_nonzero(body)), False, kind, True, colours
            )
            if thing not in found:  # both edges of one thing fill out to the same thing
                found.append(thing)
        return found

    def _match(self, objects: list[ScreenObject], number: int) -> None:
        cx, cy = self._camera
        sx, sy = self._scroll
        pairs: list[tuple[float, int, int]] = []
        tracks = list(self._tracks.values())
        for ti, t in enumerate(tracks):
            # Where it should be on screen now: moved by its velocity, slid by the scroll
            # (what is on the glass does not slide).
            ex = t.centre[0] + t.velocity[0] * (t.missing + 1) - (0 if t.glass else sx)
            ey = t.centre[1] + t.velocity[1] * (t.missing + 1) - (0 if t.glass else sy)
            for oi, o in enumerate(objects):
                if (
                    not (t.cut_out or o.cut_out)
                    and t.kind is not None
                    and o.kind is not None
                    and t.kind != o.kind
                ):
                    continue  # a cut-out's colours include a little of what it uncovered
                d = float(np.hypot(o.centre[0] - ex, o.centre[1] - ey))
                if d <= self._max_jump:
                    pairs.append((d, ti, oi))
        pairs.sort()
        used_t: set[int] = set()
        used_o: set[int] = set()
        for _d, ti, oi in pairs:
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            t, o = tracks[ti], objects[oi]
            glass = self._on_glass(o.y, o.height)
            flipped = glass != t.glass
            if flipped:
                # Found to be on the glass (or no longer): where it has been was counted
                # the other way, so its past says nothing about whether or how it moves.
                t.glass = glass
                t.history.clear()
                t.velocity = (0.0, 0.0)
            if glass:  # the level it stands still in is the screen
                new_level = o.centre
                old_level = new_level if flipped else t.centre
            else:
                new_level = (o.centre[0] + cx, o.centre[1] + cy)
                old_level = (
                    new_level if flipped else (t.centre[0] + (cx - sx), t.centre[1] + (cy - sy))
                )
            steps = t.missing + 1
            vx = (new_level[0] - old_level[0]) / steps
            vy = (new_level[1] - old_level[1]) / steps
            t.velocity = (
                (vx, vy)
                if t.seen == 1
                else (
                    0.5 * t.velocity[0] + 0.5 * vx,
                    0.5 * t.velocity[1] + 0.5 * vy,
                )
            )
            t.x, t.y, t.width, t.height = o.x, o.y, o.width, o.height
            t.kind = o.kind if o.kind is not None and not (o.cut_out and t.kind) else t.kind
            t.cut_out = o.cut_out
            t.colours = o.colours or t.colours
            t.last_frame = number
            t.missing = 0
            t.seen += 1
            t.merged = False
            t.history.append((number, *new_level))
            del t.history[:-HISTORY]
        # Two things that touch are drawn as one patch, which looks like neither of them:
        # the things it covers stay seen, where their motion takes them, touching each
        # other, instead of a new thing being made of the pair (T-976).
        for oi, o in enumerate(objects):
            if oi in used_o:
                continue
            covered = [
                ti for ti, t in enumerate(tracks) if ti not in used_t and _inside(t, o, sx, sy)
            ]
            if len(covered) >= 2:
                used_o.add(oi)
                for ti in covered:
                    used_t.add(ti)
                    t = tracks[ti]
                    t.x = round(t.x + t.velocity[0] - sx)
                    t.y = round(t.y + t.velocity[1] - sy)
                    t.last_frame = number
                    t.missing = 0
                    t.seen += 1
                    t.merged = True
                    lx, ly = t.centre[0] + cx, t.centre[1] + cy
                    t.history.append((number, lx, ly))
                    del t.history[:-HISTORY]
        for ti, t in enumerate(tracks):
            if ti not in used_t:
                t.missing += 1
                # Out of sight it keeps sliding with the camera.
                if not t.glass:
                    t.x -= sx
                    t.y -= sy
                if t.missing > self._max_missing:
                    del self._tracks[t.id]
        for oi, o in enumerate(objects):
            if oi in used_o:
                continue
            t = Track(self._next_id, o.kind, o.x, o.y, o.width, o.height, number, number)
            t.cut_out = o.cut_out
            t.colours = o.colours
            t.glass = self._on_glass(o.y, o.height)
            if t.glass:
                t.history.append((number, *o.centre))
            else:
                t.history.append((number, o.centre[0] + cx, o.centre[1] + cy))
            self._tracks[t.id] = t
            self._next_id += 1


__all__ = [
    "ObjectTracker",
    "Track",
    "estimate_scroll",
    "moving_mask",
]
