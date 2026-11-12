"""Knowing a thing by the colours it is drawn in (T-987).

Following a thing from frame to frame (T-972) loses it whenever the following
breaks: when it stands on the ground and is one patch with it, when it jumps
further than the tracker allows, when the camera is locked on it and nothing
about it moves on the screen. What does not change is how it looks. A
:class:`ColourSignature` is the handful of colours a thing is drawn in, how
many pixels of each, and the size of the patch they make; :meth:`find` picks
the thing out of a single screen with no memory of the screens before.

Nobody gives the colours. :meth:`ColourSignature.learn` is shown screens with
a box in which *some* of the thing was seen moving (a body of one colour
moving sideways only changes at its edges, so the boxes are often slivers)
and works out the rest:

* the colours in the boxes are the candidates;
* the patch of those colours joined to the box is the whole thing;
* a candidate that makes that patch bigger than a thing can be is scenery
  (the floor it stands on) and is dropped;
* a candidate that is mostly elsewhere on the screen says nothing about where
  the thing is, and is dropped too.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.objects import (
    DEFAULT_MAX_SIZE,
    PALETTE_SHARE,
    _dilate,
    _label,
)
from retro_ai_lab.observation.screen import to_array

MOSTLY = 0.5  # share of a colour's pixels on the screen that must be the thing's own
COMMON = 3  # a colour with more than this many times the thing's share floods the look
SEEN = 0.4  # share of its pixels that must be in one place for the thing to be seen
REACH = 48  # pixels from where it was last seen that are looked at first
MOST_SAMPLES = 24  # screens learnt from (more are thinned out evenly)

Box = tuple[int, int, int, int]  # x, y, width, height
Sample = tuple[Frame, Box]


def _packed(frame: Frame) -> NDArray[np.int64]:
    """One whole number per pixel (the channels packed)."""
    arr = to_array(frame).astype(np.int64)
    out = arr[..., 0].copy()
    for ch in range(1, frame.channels):
        out = (out << 8) | arr[..., ch]
    return out


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _median(values: Sequence[int]) -> int:
    return sorted(values)[len(values) // 2]


def _checked(samples: Iterable[Sample]) -> list[Sample]:
    out: list[Sample] = []
    for sample in samples:
        try:
            frame, box = sample
            x, y, w, h = box
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "A sample is a screen and the box (x, y, width, height) the thing was seen in."
            ) from exc
        if not isinstance(frame, Frame):
            raise ConfigurationError("A sample's screen must be a Frame.")
        if not all(_whole(v) for v in (x, y, w, h)) or w < 1 or h < 1:
            raise ConfigurationError("A sample's box needs whole numbers and a size of 1 or more.")
        if out and (frame.width, frame.height, frame.channels) != (
            out[0][0].width,
            out[0][0].height,
            out[0][0].channels,
        ):
            raise ConfigurationError("Every sample's screen must have the same size and channels.")
        out.append((frame, (x, y, w, h)))
    return out


@dataclass(frozen=True, slots=True)
class ColourSignature:
    """The colours one thing is drawn in, and the size of the patch they make."""

    colours: tuple[tuple[int, int], ...]  # (packed colour, how many of its pixels), by colour
    width: int
    height: int
    channels: int = 3

    def __post_init__(self) -> None:
        if not self.colours:
            raise ConfigurationError("A signature needs at least one colour.")
        for pair in self.colours:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not all(_whole(v) for v in pair)
                or pair[0] < 0
                or pair[1] < 1
            ):
                raise ConfigurationError("A signature's colours are (colour, pixels) pairs.")
        if len({c for c, _ in self.colours}) != len(self.colours):
            raise ConfigurationError("A signature names each colour once.")
        if not all(_whole(v) and v >= 1 for v in (self.width, self.height)):
            raise ConfigurationError("A signature's size must be 1 or more each way.")
        if self.channels not in (1, 3, 4):
            raise ConfigurationError("A signature's channels must be 1, 3 or 4.")

    @property
    def pixels(self) -> int:
        """How many of the thing's pixels are in its own colours."""
        return sum(n for _, n in self.colours)

    def describe(self) -> str:
        n = len(self.colours)
        return f"{n} colour{'s' if n != 1 else ''} of its own, {self.width} by {self.height} pixels"

    # ------------------------------------------------------------ learning

    @classmethod
    def learn(
        cls, samples: Iterable[Sample], *, max_size: int = DEFAULT_MAX_SIZE
    ) -> ColourSignature | None:
        """The signature of the thing seen, at least in part, in each sample's box.

        None when the thing has no colour of its own (everything it is drawn in
        is scenery, or mostly elsewhere on the screen).
        """
        if not _whole(max_size) or max_size < 1:
            raise ConfigurationError("max_size must be at least 1.")
        checked = _checked(samples)
        if not checked:
            return None
        if len(checked) > MOST_SAMPLES:
            step = len(checked) / MOST_SAMPLES
            checked = [checked[int(i * step)] for i in range(MOST_SAMPLES)]
        screens: list[tuple[NDArray[np.int64], tuple[int, int, int, int]]] = []
        in_boxes: Counter[int] = Counter()
        on_screen: Counter[int] = Counter()
        for frame, (x, y, w, h) in checked:
            packed = _packed(frame)
            values, counts = np.unique(packed, return_counts=True)
            backdrop = int(values[int(np.argmax(counts))])
            on_screen.update(dict(zip(values.tolist(), counts.tolist(), strict=True)))
            x0, y0, x1, y1 = max(0, x), max(0, y), max(0, x + w), max(0, y + h)
            seen, n = np.unique(packed[y0:y1, x0:x1], return_counts=True)
            for value, count in zip(seen.tolist(), n.tolist(), strict=True):
                if value != backdrop:
                    in_boxes[value] += count
            screens.append((packed, (x0, y0, x1, y1)))
        total = sum(in_boxes.values())
        candidates = [v for v, n in in_boxes.items() if n >= 2 and n >= PALETTE_SHARE * total]
        patches: list[tuple[NDArray[np.int64], Box]] = []
        while candidates:
            grown = _grow(screens, candidates, max_size)
            if grown is not None:
                patches = grown
                break
            # Scenery: the commonest of the candidates goes, and the rest is tried again.
            candidates.remove(max(candidates, key=lambda v: (on_screen[v], v)))
        if not candidates or not patches:
            return None
        # Of each colour: how much is the thing's own, on the screens it was made out on.
        own: Counter[int] = Counter()
        everywhere: Counter[int] = Counter()
        for packed, (x, y, w, h) in patches:
            crop = packed[y : y + h, x : x + w]
            for value in candidates:
                own[value] += int(np.count_nonzero(crop == value))
                everywhere[value] += int(np.count_nonzero(packed == value))
        keep = sorted(v for v in candidates if own[v] > 0 and own[v] >= MOSTLY * everywhere[v])
        if not keep:
            return None
        widths: list[int] = []
        heights: list[int] = []
        each: dict[int, list[int]] = {v: [] for v in keep}
        for packed, (x, y, w, h) in patches:
            crop = packed[y : y + h, x : x + w]
            mask = np.isin(crop, keep)
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            widths.append(int(np.max(xs) - np.min(xs)) + 1)
            heights.append(int(np.max(ys) - np.min(ys)) + 1)
            for value in keep:
                each[value].append(int(np.count_nonzero(crop == value)))
        if not widths:
            return None
        return cls(
            tuple((v, max(1, _median(each[v]))) for v in keep),
            _median(widths),
            _median(heights),
            checked[0][0].channels,
        )

    # ------------------------------------------------------------- finding

    def find(
        self, frame: Frame, *, near: tuple[int, int] | None = None, reach: int = REACH
    ) -> Box | None:
        """Where the thing is on ``frame``: the box round its own colours, or None.

        ``near`` is where its centre was last seen; the fullest place within
        ``reach`` of it wins when the thing can be seen there, so of two things
        alike the one being followed stays the one. Otherwise the fullest place
        on the whole screen is taken (a new life starts somewhere else).
        """
        if not isinstance(frame, Frame):
            raise ConfigurationError("find() needs a Frame.")
        if not _whole(reach) or reach < 0:
            raise ConfigurationError("reach must be a whole number, 0 or more.")
        if frame.channels != self.channels:
            return None
        packed = _packed(frame)
        # A colour that floods this screen (a wall in the player's shirt colour) would
        # put the thing anywhere: it is left out of this look, and the others decide.
        told = [(v, n) for v, n in self.colours if 0 < np.count_nonzero(packed == v) <= COMMON * n]
        if not told:
            return None
        mask = np.isin(packed, [v for v, _ in told])
        need = max(2, math.ceil(SEEN * sum(n for _, n in told)))
        rows, cols = mask.shape
        w, h = min(self.width, cols), min(self.height, rows)
        # How many of its pixels each w-by-h window holds (summed-area table).
        table = np.zeros((rows + 1, cols + 1), np.int32)
        table[1:, 1:] = mask.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
        sums = table[h:, w:] - table[:-h, w:] - table[h:, :-w] + table[:-h, :-w]
        spot: tuple[int, int] | None = None
        if near is not None:
            left, top = near[0] - w // 2, near[1] - h // 2
            x0, x1 = max(0, left - reach), min(sums.shape[1], left + reach + 1)
            y0, y1 = max(0, top - reach), min(sums.shape[0], top + reach + 1)
            if x0 < x1 and y0 < y1:
                part = sums[y0:y1, x0:x1]
                yy, xx = divmod(int(np.argmax(part)), part.shape[1])
                if part[yy, xx] >= need:
                    spot = (x0 + xx, y0 + yy)
        if spot is None:
            yy, xx = divmod(int(np.argmax(sums)), sums.shape[1])
            if sums[yy, xx] < need:
                return None
            spot = (xx, yy)
        # The box round its colours there, with room for it to be a little bigger
        # than when it was learnt (half its size each way).
        x, y = spot
        gx, gy = max(0, x - w // 2), max(0, y - h // 2)
        ys, xs = np.nonzero(mask[gy : y + h + h // 2, gx : x + w + w // 2])
        return (
            gx + int(np.min(xs)),
            gy + int(np.min(ys)),
            int(np.max(xs) - np.min(xs)) + 1,
            int(np.max(ys) - np.min(ys)) + 1,
        )

    # -------------------------------------------------------------- saving

    def to_dict(self) -> dict[str, Any]:
        return {
            "colours": [[v, n] for v, n in self.colours],
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ColourSignature:
        try:
            colours = tuple((v, n) for v, n in data["colours"])
            return cls(colours, data["width"], data["height"], data["channels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad saved colours: {exc}") from exc


def _grow(
    screens: Sequence[tuple[NDArray[np.int64], tuple[int, int, int, int]]],
    candidates: Sequence[int],
    max_size: int,
) -> list[tuple[NDArray[np.int64], Box]] | None:
    """On each screen, the patch of the candidate colours joined to the box: the whole
    thing. None when a patch is bigger than a thing can be (a candidate is scenery)."""
    out: list[tuple[NDArray[np.int64], Box]] = []
    for packed, (x0, y0, x1, y1) in screens:
        mask = np.isin(packed, candidates)
        if not mask[y0:y1, x0:x1].any():
            continue
        labels, _count = _label(_dilate(mask))
        joined = np.unique(labels[y0:y1, x0:x1][mask[y0:y1, x0:x1]])
        ys, xs = np.nonzero(np.isin(labels, joined) & mask)
        w, h = int(np.max(xs) - np.min(xs)) + 1, int(np.max(ys) - np.min(ys)) + 1
        if w > max_size or h > max_size:
            return None
        out.append((packed, (int(np.min(xs)), int(np.min(ys)), w, h)))
    return out


__all__ = ["COMMON", "MOSTLY", "REACH", "SEEN", "Box", "ColourSignature", "Sample"]
