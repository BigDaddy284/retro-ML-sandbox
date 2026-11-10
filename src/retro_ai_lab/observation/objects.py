"""The things on a screen, found without anyone pointing at them (T-971).

A retro game draws its sprites over a plain backdrop: the sky, a black
dungeon floor, the road. So the commonest colour of a frame is taken for the
backdrop, and every patch of other colours is a candidate thing. Parts of
one sprite a pixel apart (the eyes, the feet) are joined; patches bigger
than ``max_size`` either way are *scenery* (the ground, a wall, a big
building), kept apart from the things because they are the level, not
something in it.

An :class:`ObjectCatalogue` sorts the things it sees, screen after screen,
into *kinds* by their exact look (a copy facing the other way is the same
kind) and names them in the order first seen: "thing 1", "thing 2". Each
kind keeps a :class:`~retro_ai_lab.observation.sprites.SpriteTemplate`, so
the fast finder of T-970 can look for it on later screens.

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
    arr = to_array(frame).reshape(-1, frame.channels)
    colours, counts = np.unique(arr, axis=0, return_counts=True)
    return tuple(int(v) for v in colours[int(np.argmax(counts))])


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


def _look(crop: NDArray[np.uint8], mask: NDArray[np.bool_]) -> bytes:
    shown = np.where(mask[..., None], crop, 0).astype(np.uint8)
    h = hashlib.blake2b(digest_size=16)
    h.update(np.array(crop.shape, np.int32).tobytes())
    h.update(shown.tobytes())
    h.update(mask.tobytes())
    return h.digest()


class ObjectCatalogue:
    """Sorts the things seen screen after screen into kinds, by look."""

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

    @property
    def dropped(self) -> int:
        """Looks not catalogued because the catalogue was full."""
        return len(self._dropped)

    def kinds(self) -> tuple[ObjectKind, ...]:
        return tuple(self._kinds.values())

    def kind(self, kind_id: str) -> ObjectKind:
        try:
            return self._kinds[kind_id]
        except KeyError as exc:
            raise ConfigurationError(f"No kind {kind_id!r} has been seen.") from exc

    def observe(
        self, frame: Frame, *, background: tuple[int, ...] | None = None
    ) -> tuple[ScreenObject, ...]:
        """The things on ``frame`` with their kinds; new looks become new kinds."""
        arr, fg = _foreground(frame, background)
        out: list[ScreenObject] = []
        on_screen: Counter[str] = Counter()
        for o in _objects(fg, self._min_pixels, self._max_size):
            if o.scenery:
                out.append(o)
                continue
            crop = arr[o.y : o.y + o.height, o.x : o.x + o.width]
            mask = fg[o.y : o.y + o.height, o.x : o.x + o.width]
            look = _look(crop, mask)
            kind_id = self._by_look.get(look)
            if kind_id is None:
                mirrored = _look(crop[:, ::-1], mask[:, ::-1])
                kind_id = self._by_look.get(mirrored)
                if kind_id is None:
                    kind_id = self._add(look, mirrored, crop, mask, frame.channels)
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

    def _add(
        self,
        look: bytes,
        mirrored: bytes,
        crop: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        channels: int,
    ) -> str | None:
        if len(self._kinds) >= self._most:
            self._dropped.add(min(look, mirrored))
            return None
        number = len(self._kinds) + 1
        kind_id = f"thing-{number}"
        h, w = mask.shape
        template = SpriteTemplate(
            name=f"thing {number}",
            tag=SpriteTag.OTHER,
            width=w,
            height=h,
            channels=channels,
            pixels=np.ascontiguousarray(crop).tobytes(),
            mask=mask.astype(np.uint8).tobytes(),
        )
        self._kinds[kind_id] = ObjectKind(kind_id, template.name, template)
        self._by_look[look] = kind_id
        self._by_look[mirrored] = kind_id
        return kind_id


__all__ = [
    "ObjectCatalogue",
    "ObjectKind",
    "ScreenObject",
    "background_colour",
    "find_objects",
]
