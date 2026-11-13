"""The things on a screen, found without anyone pointing at them (T-971).

A retro game draws its sprites over a plain backdrop: the sky, a black
dungeon floor, the road. So the commonest colour of a frame is taken for the
backdrop, and every patch of other colours is a candidate thing. Parts of
one sprite a pixel apart (the eyes, the feet) are joined; patches bigger
than ``max_size`` either way are *scenery* (the ground, a wall, a big
building), kept apart from the things because they are the level, not
something in it.

An :class:`ObjectCatalogue` sorts the things it sees, screen after screen,
into *kinds* by the colours they are drawn in (T-977: every animation frame
and facing of a sprite is one kind) and names them in the order first seen:
"thing 1", "thing 2". Each kind keeps a
:class:`~retro_ai_lab.observation.sprites.SpriteTemplate`, so the fast finder
of T-970 can look for it on later screens.

What a kind *is* (the player, an enemy, a coin, a platform) is not decided
here: that comes from what the kind does, followed over time and set
against what memory says happened (DESIGN.md §3.2, ``docs/VISION.md``).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.screen import to_array
from retro_ai_lab.observation.sprites import SpriteTag, SpriteTemplate

DEFAULT_MIN_PIXELS = 3  # smaller is a speck (a star, a stray pixel), not a thing
DEFAULT_MAX_SIZE = 64  # wider or taller than this is scenery
DEFAULT_MOST_KINDS = 256


def background_colour(frame: Frame) -> tuple[int, ...]:
    """The commonest colour of the frame (ties go to the lowest colour value)."""
    arr = to_array(frame).reshape(-1, frame.channels).astype(np.int64)
    packed = arr[:, 0].copy()
    for ch in range(1, frame.channels):
        packed = (packed << 8) | arr[:, ch]
    colours, counts = np.unique(packed, return_counts=True)
    value = int(colours[int(np.argmax(counts))])
    return tuple((value >> (8 * (frame.channels - 1 - ch))) & 0xFF for ch in range(frame.channels))


@dataclass(frozen=True, slots=True)
class ScreenObject:
    """One thing on the screen: the box round its own pixels."""

    x: int
    y: int
    width: int
    height: int
    pixels: int  # how many of the box's pixels are its own
    scenery: bool  # too big to be a thing: part of the level
    kind: str | None = None  # set by an ObjectCatalogue; None for scenery
    cut_out: bool = False  # found by its motion against the scenery, not on a still screen
    # The colours (packed) a cut-out was filled out by, when it could be (T-1016).
    colours: tuple[int, ...] = ()

    @property
    def centre(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


def _dilate(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Grow the mask by one pixel each way (3x3), so parts a pixel apart touch."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    grown = out.copy()
    grown[:, 1:] |= out[:, :-1]
    grown[:, :-1] |= out[:, 1:]
    return grown


def _label(mask: NDArray[np.bool_]) -> tuple[NDArray[np.int32], int]:
    """Connected areas (8-neighbour) of a mask, numbered from 1; run-based union-find."""
    height = mask.shape[0]
    labels = np.zeros(mask.shape, np.int32)
    parent: list[int] = [0]

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    previous: list[tuple[int, int, int]] = []  # (start, stop, label) of the row above
    padded = np.zeros(mask.shape[1] + 2, np.int8)
    for y in range(height):
        padded[1:-1] = mask[y]
        edges = np.flatnonzero(np.diff(padded))
        current: list[tuple[int, int, int]] = []
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            s, e = int(start), int(stop)
            label = 0
            for ps, pe, pl in previous:
                if ps <= e and pe >= s:  # touches, diagonals included
                    root = find(pl)
                    if label == 0:
                        label = root
                    elif root != label:
                        parent[root] = label
            if label == 0:
                label = len(parent)
                parent.append(label)
            current.append((s, e, label))
            labels[y, s:e] = label
        previous = current
    roots = np.array([find(i) for i in range(len(parent))], np.int32)
    # Renumber the roots 1..n.
    unique = np.unique(roots[1:]) if len(parent) > 1 else np.array([], np.int32)
    renumber = np.zeros(len(parent), np.int32)
    renumber[unique] = np.arange(1, len(unique) + 1, dtype=np.int32)
    return renumber[roots][labels], len(unique)


def _foreground(
    frame: Frame, background: tuple[int, ...] | None
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    arr = to_array(frame)
    if background is None:
        background = background_colour(frame)
    elif len(background) != frame.channels:
        raise ConfigurationError(f"The background colour needs {frame.channels} values.")
    return arr, (arr != np.array(background, np.uint8)).any(axis=2)


def _objects(fg: NDArray[np.bool_], min_pixels: int, max_size: int) -> list[ScreenObject]:
    labels, count = _label(_dilate(fg))
    if count == 0:
        return []
    ys, xs = np.nonzero(fg)
    own = labels[ys, xs]
    sizes = np.bincount(own, minlength=count + 1)
    x0 = np.full(count + 1, fg.shape[1], np.int64)
    y0 = np.full(count + 1, fg.shape[0], np.int64)
    x1 = np.full(count + 1, -1, np.int64)
    y1 = np.full(count + 1, -1, np.int64)
    np.minimum.at(x0, own, xs)
    np.minimum.at(y0, own, ys)
    np.maximum.at(x1, own, xs)
    np.maximum.at(y1, own, ys)
    found: list[ScreenObject] = []
    for i in range(1, count + 1):
        if sizes[i] < min_pixels:
            continue
        w, h = int(x1[i] - x0[i] + 1), int(y1[i] - y0[i] + 1)
        found.append(
            ScreenObject(int(x0[i]), int(y0[i]), w, h, int(sizes[i]), w > max_size or h > max_size)
        )
    found.sort(key=lambda o: (o.y, o.x))
    return found


def _check(min_pixels: int, max_size: int) -> None:
    if not isinstance(min_pixels, int) or min_pixels < 1:
        raise ConfigurationError("min_pixels must be at least 1.")
    if not isinstance(max_size, int) or max_size < 1:
        raise ConfigurationError("max_size must be at least 1.")


def find_objects(
    frame: Frame,
    *,
    background: tuple[int, ...] | None = None,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_size: int = DEFAULT_MAX_SIZE,
) -> tuple[ScreenObject, ...]:
    """Every thing and every piece of scenery on ``frame``, top row first."""
    _check(min_pixels, max_size)
    _, fg = _foreground(frame, background)
    return tuple(_objects(fg, min_pixels, max_size))


@dataclass(slots=True)
class ObjectKind:
    """Things that look the same, and how often they have been seen."""

    id: str
    name: str
    template: SpriteTemplate
    sightings: int = 0  # copies seen, all screens together
    screens: int = 0  # screens it was on


PALETTE_SHARE = 0.04  # a colour is part of a thing's palette from this share of its pixels


def _palette(crop: NDArray[np.uint8], mask: NDArray[np.bool_]) -> bytes:
    """A thing's look as its colours (T-977): the same in every animation frame and facing.

    The user's game showed that an exact look made a new kind of every walking
    frame and every sprite half behind another (over 250 kinds in an hour);
    the colours a sprite is drawn in stay the same while it animates.
    """
    pixels = crop[mask].astype(np.int64)
    if pixels.size == 0:
        return b""
    packed = pixels[:, 0].copy()
    for ch in range(1, crop.shape[2]):
        packed = (packed << 8) | pixels[:, ch]
    colours, counts = np.unique(packed, return_counts=True)
    keep = colours[(counts >= 2) & (counts >= PALETTE_SHARE * len(packed))]
    if keep.size == 0:
        keep = colours[np.argsort(-counts)[:1]]
    h = hashlib.blake2b(digest_size=16)
    h.update(np.array([crop.shape[2]], np.int32).tobytes())
    h.update(np.sort(keep).astype(np.int64).tobytes())
    return h.digest()


class ObjectCatalogue:
    """Sorts the things seen screen after screen into kinds, by their colours (T-977)."""

    def __init__(
        self,
        *,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_size: int = DEFAULT_MAX_SIZE,
        most: int = DEFAULT_MOST_KINDS,
    ) -> None:
        _check(min_pixels, max_size)
        if not isinstance(most, int) or most < 1:
            raise ConfigurationError("most must be at least 1.")
        self._min_pixels = min_pixels
        self._max_size = max_size
        self._most = most
        self._kinds: dict[str, ObjectKind] = {}
        self._by_look: dict[bytes, str] = {}
        self._dropped: set[bytes] = set()
        self._background: tuple[int, ...] | None = None

    @property
    def dropped(self) -> int:
        """Looks not catalogued because the catalogue was full."""
        return len(self._dropped)

    @property
    def background(self) -> tuple[int, ...] | None:
        """The backdrop colour of the last frame observed."""
        return self._background

    def kinds(self) -> tuple[ObjectKind, ...]:
        return tuple(self._kinds.values())

    def kind(self, kind_id: str) -> ObjectKind:
        try:
            return self._kinds[kind_id]
        except KeyError as exc:
            raise ConfigurationError(f"No kind {kind_id!r} has been seen.") from exc

    def observe(
        self,
        frame: Frame,
        *,
        background: tuple[int, ...] | None = None,
        hide: tuple[int, int, int, int] | None = None,
    ) -> tuple[ScreenObject, ...]:
        """The things on ``frame`` with their kinds; new palettes become new kinds.

        ``hide`` is a box (x, y, width, height) to leave out of the picture: what is
        drawn there counts as backdrop (T-987: the player, known by its own looks, so
        that it is not one patch with whatever it stands on or touches).
        """
        if background is None:
            background = background_colour(frame)
        self._background = background
        arr, fg = _foreground(frame, background)
        if hide is not None:
            x, y, w, h = hide
            fg[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)] = False
        out: list[ScreenObject] = []
        on_screen: Counter[str] = Counter()
        for o in _objects(fg, self._min_pixels, self._max_size):
            if o.scenery:
                out.append(o)
                continue
            crop = arr[o.y : o.y + o.height, o.x : o.x + o.width]
            mask = fg[o.y : o.y + o.height, o.x : o.x + o.width]
            kind_id = self.classify(crop, mask)
            if kind_id is None:
                out.append(o)
                continue
            on_screen[kind_id] += 1
            out.append(ScreenObject(o.x, o.y, o.width, o.height, o.pixels, False, kind_id))
        for kind_id, n in on_screen.items():
            k = self._kinds[kind_id]
            k.sightings += n
            k.screens += 1
        return tuple(out)

    def classify(self, crop: NDArray[np.uint8], mask: NDArray[np.bool_]) -> str | None:
        """The kind of the thing drawn by ``crop``'s pixels under ``mask``; a new palette
        is a new kind (None once the catalogue is full)."""
        if crop.ndim != 3 or mask.shape != crop.shape[:2] or not mask.any():
            return None
        look = _palette(crop, mask)
        kind_id = self._by_look.get(look)
        if kind_id is None:
            kind_id = self._add(look, crop, mask)
        return kind_id

    def _add(self, look: bytes, crop: NDArray[np.uint8], mask: NDArray[np.bool_]) -> str | None:
        if len(self._kinds) >= self._most:
            self._dropped.add(look)
            return None
        number = len(self._kinds) + 1
        kind_id = f"thing-{number}"
        h, w = mask.shape
        template = SpriteTemplate(
            name=f"thing {number}",
            tag=SpriteTag.OTHER,
            width=w,
            height=h,
            channels=crop.shape[2],
            pixels=np.ascontiguousarray(crop).tobytes(),
            mask=mask.astype(np.uint8).tobytes(),
        )
        self._kinds[kind_id] = ObjectKind(kind_id, template.name, template)
        self._by_look[look] = kind_id
        return kind_id


__all__ = [
    "ObjectCatalogue",
    "ObjectKind",
    "ScreenObject",
    "background_colour",
    "find_objects",
]
