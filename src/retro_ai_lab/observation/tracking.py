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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.objects import (
    DEFAULT_MAX_SIZE,
    DEFAULT_MIN_PIXELS,
    ObjectCatalogue,
    ScreenObject,
    _dilate,
    _label,
)
from retro_ai_lab.observation.screen import to_array

DEFAULT_MAX_SHIFT = 8  # pixels a frame the camera may move either way
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


def _overlap(
    a: NDArray[np.int32], b: NDArray[np.int32], dx: int, dy: int
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """The parts of ``a`` (after) and ``b`` (before) showing one spot: a[y,x] ~ b[y+dy,x+dx]."""
    h, w = a.shape
    ay0, ay1 = max(0, -dy), min(h, h - dy)
    ax0, ax1 = max(0, -dx), min(w, w - dx)
    return a[ay0:ay1, ax0:ax1], b[ay0 + dy : ay1 + dy, ax0 + dx : ax1 + dx]


def estimate_scroll(
    before: Frame,
    after: Frame,
    *,
    max_dx: int = DEFAULT_MAX_SHIFT,
    max_dy: int = DEFAULT_MAX_SHIFT,
) -> tuple[int, int]:
    """How far the camera moved: ``(dx, dy)`` with ``after[y, x] == before[y+dy, x+dx]``.

    Positive ``dx`` is the camera moving right (the level sliding left).
    """
    _check_pair(before, after)
    if not isinstance(max_dx, int) or not isinstance(max_dy, int) or max_dx < 0 or max_dy < 0:
        raise ConfigurationError("The largest shift must be a whole number, 0 or more.")
    max_dx = min(max_dx, before.width - 1)
    max_dy = min(max_dy, before.height - 1)
    a, b = _packed(after), _packed(before)
    best: tuple[float, int, int, int] | None = None
    for dy in range(-max_dy, max_dy + 1):
        for dx in range(-max_dx, max_dx + 1):
            aa, bb = _overlap(a, b, dx, dy)
            if aa.size == 0:
                continue
            mismatch = float(np.count_nonzero(aa != bb)) / aa.size
            key = (round(mismatch, 6), abs(dx) + abs(dy), dy, dx)
            if best is None or key < best:
                best = key
    assert best is not None
    return best[3], best[2]


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
    ) -> None:
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
        self._previous: Frame | None = None
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._camera = (0, 0)  # where the camera is in the level, summed scrolls
        self._scroll = (0, 0)

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

    def tracks(self) -> tuple[Track, ...]:
        """The things on screen now (seen this frame or missing only briefly)."""
        return tuple(sorted(self._tracks.values(), key=lambda t: t.id))

    def reset(self) -> None:
        """Forget everything followed (a new go, a loaded state); kinds are kept."""
        self._previous = None
        self._tracks.clear()
        self._camera = (0, 0)
        self._scroll = (0, 0)

    def update(self, frame: Frame) -> tuple[Track, ...]:
        if not isinstance(frame, Frame):
            raise ConfigurationError("update() needs a Frame.")
        prev = self._previous
        if prev is not None and (prev.width, prev.height, prev.channels) != (
            frame.width,
            frame.height,
            frame.channels,
        ):
            self.reset()
            prev = None
        objects = [o for o in self._catalogue.observe(frame) if not o.scenery]
        if prev is not None:
            self._scroll = estimate_scroll(
                prev, frame, max_dx=self._max_shift, max_dy=self._max_shift
            )
            self._camera = (self._camera[0] + self._scroll[0], self._camera[1] + self._scroll[1])
            objects.extend(self._movers_in_scenery(prev, frame, objects))
        else:
            self._scroll = (0, 0)
        self._match(objects, frame.number)
        self._previous = frame
        return self.tracks()

    # ------------------------------------------------------------ internals

    def _movers_in_scenery(
        self, prev: Frame, frame: Frame, things: list[ScreenObject]
    ) -> list[ScreenObject]:
        """Moving patches not already covered by a thing: a walker on the ground."""
        moved = moving_mask(prev, frame, self._scroll)
        sx, sy = self._scroll
        # What is already a thing needs no second look, and neither does the spot a
        # thing of a known kind has just left (its trail is a change too). A mover cut
        # out of the scenery keeps its trail: its patch is where it was and is, and
        # the middle of that moves as fast as it does.
        boxes = [(o.x, o.y, o.width, o.height) for o in things]
        boxes += [
            (t.x - sx, t.y - sy, t.width, t.height)
            for t in self._tracks.values()
            if t.kind is not None
        ]
        for x, y, w, h in boxes:
            moved[max(0, y - 1) : max(0, y + h + 1), max(0, x - 1) : max(0, x + w + 1)] = False
        if not moved.any():
            return []
        labels, count = _label(_dilate(moved))
        found: list[ScreenObject] = []
        for i in range(1, count + 1):
            ys, xs = np.nonzero((labels == i) & moved)
            if len(ys) < self._min_pixels:
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            w, h = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
            if w > self._max_size or h > self._max_size:
                continue  # a big change (a screen flash, a new room) is not a thing
            found.append(ScreenObject(x0, y0, w, h, len(ys), False, None))
        return found

    def _match(self, objects: list[ScreenObject], number: int) -> None:
        cx, cy = self._camera
        sx, sy = self._scroll
        pairs: list[tuple[float, int, int]] = []
        tracks = list(self._tracks.values())
        for ti, t in enumerate(tracks):
            # Where it should be on screen now: moved by its velocity, slid by the scroll.
            ex = t.centre[0] + t.velocity[0] * (t.missing + 1) - sx
            ey = t.centre[1] + t.velocity[1] * (t.missing + 1) - sy
            for oi, o in enumerate(objects):
                if t.kind is not None and o.kind is not None and t.kind != o.kind:
                    continue
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
            old_level = (t.centre[0] + (cx - sx), t.centre[1] + (cy - sy))
            new_level = (o.centre[0] + cx, o.centre[1] + cy)
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
            t.kind = o.kind if o.kind is not None else t.kind
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
                t.x -= sx
                t.y -= sy
                if t.missing > self._max_missing:
                    del self._tracks[t.id]
        for oi, o in enumerate(objects):
            if oi in used_o:
                continue
            t = Track(self._next_id, o.kind, o.x, o.y, o.width, o.height, number, number)
            t.history.append((number, o.centre[0] + cx, o.centre[1] + cy))
            self._tracks[t.id] = t
            self._next_id += 1


__all__ = [
    "ObjectTracker",
    "Track",
    "estimate_scroll",
    "moving_mask",
]
