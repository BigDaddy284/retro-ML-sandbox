"""Find what a person tagged on the screen: enemies, the player, items, ground (T-970).

The understanding engine learns a game from its memory, and memory says that
*something* at an address moves; it cannot say that the something is an
enemy. A person can: they draw a box round a thing on the screen and say what
it is ("enemy", "ground", "coin"). This module turns that box into a
:class:`SpriteTemplate` and finds every copy of it in later frames, so the
app can see its surroundings the way the person does.

Retro games draw sprites pixel for pixel from a fixed palette, never scaled
or rotated; only mirrored (a walker turning round) and drawn over different
backgrounds. So the matcher is a masked template match, not the image
pyramid or keypoint matching a phone-automation tool needs:

* the template keeps a *mask*: pixels of the crop's background colour (the
  most common colour on its border, or one the person names) are
  see-through, so an enemy tagged against the sky is still found in front of
  a bush;
* a pixel matches when every channel is within ``tolerance`` (emulators,
  palettes and filters shift colours a little);
* a spot's score is the share of the template's solid pixels that match;
  ``min_score`` below 1 allows a sprite partly covered by another;
* a few *anchor* pixels are checked across the whole frame first (one NumPy
  comparison each), and only spots that pass most of them are scored in
  full, so a 256x240 screen with several tags costs a few milliseconds;
* overlapping hits collapse to the best one, and the mirror image is tried
  too.

Only NumPy is used (ADR-008); see ``docs/VISION.md`` for why the Airtest
library and OpenCV were not taken on.
"""

from __future__ import annotations

import base64
import binascii
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.screen import to_array

DEFAULT_TOLERANCE = 24  # per channel, out of 255
DEFAULT_MIN_SCORE = 0.9
DEFAULT_ANCHORS = 8
OVERLAP_LIMIT = 0.3  # hits sharing more than this much of the smaller box are one thing
_VERIFY_BATCH = 4096  # spots scored at once, so memory stays bounded on a busy screen


class SpriteTag(StrEnum):
    """What a tagged thing is to the player (what the AI should make of it)."""

    PLAYER = "player"
    ENEMY = "enemy"  # moves and hurts
    HAZARD = "hazard"  # hurts but is part of the level: spikes, lava, pits
    ITEM = "item"  # worth picking up: coins, power-ups
    SOLID = "solid"  # ground, walls, platforms: stand on it, bump into it
    OTHER = "other"


TAG_WORDS: dict[SpriteTag, str] = {
    SpriteTag.PLAYER: "you",
    SpriteTag.ENEMY: "an enemy",
    SpriteTag.HAZARD: "something that hurts",
    SpriteTag.ITEM: "something to pick up",
    SpriteTag.SOLID: "ground, a wall or a platform",
    SpriteTag.OTHER: "something else",
}


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: object, what: str) -> bytes:
    if not isinstance(text, str):
        raise ConfigurationError(f"Sprite {what} must be text.")
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise ConfigurationError(f"Sprite {what} is not valid base64.") from exc


@dataclass(frozen=True, slots=True)
class SpriteTemplate:
    """A tagged picture: ``pixels`` row-major like a :class:`Frame`, ``mask`` 1 = solid."""

    name: str
    tag: SpriteTag
    width: int
    height: int
    channels: int
    pixels: bytes
    mask: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("A tagged sprite needs a name.")
        if not isinstance(self.tag, SpriteTag):
            raise ConfigurationError(f"Unknown sprite tag {self.tag!r}.")
        for value, what in ((self.width, "width"), (self.height, "height")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ConfigurationError(f"Sprite {what} must be a positive whole number.")
        if self.channels not in (1, 3, 4):
            raise ConfigurationError("Sprite channels must be 1, 3 or 4.")
        if not isinstance(self.pixels, bytes) or not isinstance(self.mask, bytes):
            raise ConfigurationError("Sprite pixels and mask must be bytes.")
        if len(self.pixels) != self.width * self.height * self.channels:
            raise ConfigurationError("Sprite pixels do not fill its size.")
        if len(self.mask) != self.width * self.height:
            raise ConfigurationError("Sprite mask does not fill its size.")
        if not any(self.mask):
            raise ConfigurationError("A tagged sprite needs at least one solid pixel.")

    # ------------------------------------------------------------ building

    @classmethod
    def from_frame(
        cls,
        frame: Frame,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        name: str,
        tag: SpriteTag,
        background: tuple[int, ...] | None = None,
    ) -> SpriteTemplate:
        """Cut the box ``(x, y, width, height)`` out of ``frame`` and tag it.

        Pixels of the ``background`` colour are see-through; by default that is
        the commonest colour on the box's border. When that leaves nothing
        solid (the box is all one colour, or the sprite fills it), every pixel
        counts.
        """
        for value, what in ((x, "x"), (y, "y"), (width, "width"), (height, "height")):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigurationError(f"The box's {what} must be a whole number.")
        if width < 1 or height < 1:
            raise ConfigurationError("The box must be at least one pixel across.")
        if x < 0 or y < 0 or x + width > frame.width or y + height > frame.height:
            raise ConfigurationError("The box must lie inside the screen.")
        crop = to_array(frame)[y : y + height, x : x + width]
        if background is None:
            border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]], axis=0)
            colours = Counter(tuple(int(v) for v in px) for px in border)
            bg = np.array(colours.most_common(1)[0][0], np.uint8)
        else:
            if len(background) != frame.channels:
                raise ConfigurationError(f"The background colour needs {frame.channels} values.")
            bg = np.array(background, np.uint8)
        solid = (crop != bg).any(axis=2)
        if not solid.any():
            solid[:] = True
        return cls(
            name=name,
            tag=tag,
            width=width,
            height=height,
            channels=frame.channels,
            pixels=np.ascontiguousarray(crop).tobytes(),
            mask=solid.astype(np.uint8).tobytes(),
        )

    # ------------------------------------------------------------ views

    @property
    def solid_pixels(self) -> int:
        return sum(1 for b in self.mask if b)

    def array(self) -> NDArray[np.uint8]:
        return np.frombuffer(self.pixels, np.uint8).reshape(self.height, self.width, self.channels)

    def mask_array(self) -> NDArray[np.bool_]:
        return np.frombuffer(self.mask, np.uint8).reshape(self.height, self.width).astype(bool)

    # ------------------------------------------------------------ storage

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tag": self.tag.value,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "pixels": _b64(self.pixels),
            "mask": _b64(self.mask),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpriteTemplate:
        if not isinstance(data, Mapping):
            raise ConfigurationError("A saved sprite must be a mapping.")
        try:
            tag = SpriteTag(data["tag"])
            return cls(
                name=data["name"],
                tag=tag,
                width=data["width"],
                height=data["height"],
                channels=data["channels"],
                pixels=_unb64(data["pixels"], "pixels"),
                mask=_unb64(data["mask"], "mask"),
            )
        except KeyError as exc:
            raise ConfigurationError(f"A saved sprite is missing {exc.args[0]!r}.") from exc
        except ValueError as exc:
            raise ConfigurationError(f"A saved sprite is broken: {exc}") from exc


@dataclass(frozen=True, slots=True)
class SpriteMatch:
    """One tagged thing found on the screen; ``x, y`` is its top-left pixel."""

    name: str
    tag: SpriteTag
    x: int
    y: int
    width: int
    height: int
    score: float  # share of the template's solid pixels that matched, 0..1
    mirrored: bool

    @property
    def centre(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass(frozen=True, slots=True)
class _Pattern:
    template: SpriteTemplate
    mirrored: bool
    pixels: NDArray[np.int16]  # h x w x c
    mask: NDArray[np.bool_]  # h x w
    anchors: tuple[tuple[int, int], ...]  # (row, column) of solid pixels checked first


def _anchors(
    mask: NDArray[np.bool_], pixels: NDArray[np.int16], most: int
) -> tuple[tuple[int, int], ...]:
    """Solid pixels spread over the sprite, rarest colours first so they say the most."""
    ys, xs = np.nonzero(mask)
    count = len(ys)
    colours = Counter(tuple(int(v) for v in pixels[y, x]) for y, x in zip(ys, xs, strict=True))
    order = sorted(
        range(count),
        key=lambda i: (colours[tuple(int(v) for v in pixels[ys[i], xs[i]])], i),
    )
    # From each colour (rarest first) take pixels spread across the sprite.
    picked: list[int] = []
    step = max(1, count // most)
    for start in range(step):
        for i in order[start::step]:
            if i not in picked:
                picked.append(i)
            if len(picked) == most:
                break
        if len(picked) >= min(most, count):
            break
    return tuple((int(ys[i]), int(xs[i])) for i in picked[:most])


def _overlap(a: SpriteMatch, b: SpriteMatch) -> float:
    w = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    h = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / min(a.width * a.height, b.width * b.height)


class SpriteFinder:
    """Finds every tagged sprite in a frame."""

    def __init__(
        self,
        templates: Iterable[SpriteTemplate],
        *,
        tolerance: int = DEFAULT_TOLERANCE,
        min_score: float = DEFAULT_MIN_SCORE,
        mirrored: bool = True,
        anchors: int = DEFAULT_ANCHORS,
    ) -> None:
        if not isinstance(tolerance, int) or not 0 <= tolerance <= 255:
            raise ConfigurationError("tolerance must be within 0..255.")
        if not isinstance(min_score, int | float) or not 0.0 < min_score <= 1.0:
            raise ConfigurationError("min_score must be above 0 and at most 1.")
        if not isinstance(anchors, int) or anchors < 1:
            raise ConfigurationError("anchors must be at least 1.")
        chosen = list(templates)
        if not chosen:
            raise ConfigurationError("Tag at least one sprite to look for.")
        if not all(isinstance(t, SpriteTemplate) for t in chosen):
            raise ConfigurationError("Only tagged sprites can be looked for.")
        self._tolerance = tolerance
        self._min_score = float(min_score)
        self._patterns: list[_Pattern] = []
        for t in chosen:
            px = t.array().astype(np.int16)
            mask = t.mask_array()
            self._patterns.append(_Pattern(t, False, px, mask, _anchors(mask, px, anchors)))
            if mirrored:
                mpx, mmask = px[:, ::-1], mask[:, ::-1]
                if not (np.array_equal(mpx, px) and np.array_equal(mmask, mask)):
                    self._patterns.append(
                        _Pattern(t, True, mpx, mmask, _anchors(mmask, mpx, anchors))
                    )

    @property
    def templates(self) -> tuple[SpriteTemplate, ...]:
        return tuple(p.template for p in self._patterns if not p.mirrored)

    def find(self, frame: Frame) -> tuple[SpriteMatch, ...]:
        """Every tagged thing on ``frame``, top row first then left to right."""
        if not isinstance(frame, Frame):
            raise ConfigurationError("find() needs a Frame.")
        arr = to_array(frame).astype(np.int16)
        near: dict[tuple[int, ...], NDArray[np.bool_]] = {}  # colour -> pixels close to it
        hits: list[SpriteMatch] = []
        for pattern in self._patterns:
            if pattern.template.channels != frame.channels:
                raise ConfigurationError(
                    f"{pattern.template.name!r} was tagged on a {pattern.template.channels}-"
                    f"channel screen, this one has {frame.channels}."
                )
            hits.extend(self._find_pattern(arr, pattern, near))
        hits.sort(key=lambda m: (-m.score, m.y, m.x, m.mirrored, m.name))
        kept: list[SpriteMatch] = []
        for m in hits:
            # A hit whose box is mostly explained by better hits already kept is the
            # same thing seen again (a shifted copy, the mirror, a repeating tile).
            if sum(_overlap(m, k) for k in kept) <= OVERLAP_LIMIT:
                kept.append(m)
        kept.sort(key=lambda m: (m.y, m.x, m.name))
        return tuple(kept)

    def find_tag(self, frame: Frame, tag: SpriteTag) -> tuple[SpriteMatch, ...]:
        return tuple(m for m in self.find(frame) if m.tag is tag)

    # ------------------------------------------------------------ internals

    def _near(
        self,
        arr: NDArray[np.int16],
        colour: NDArray[np.int16],
        cache: dict[tuple[int, ...], NDArray[np.bool_]],
    ) -> NDArray[np.bool_]:
        key = tuple(int(v) for v in colour)
        hit = cache.get(key)
        if hit is None:
            # Retro games use a small palette, so one map per colour serves every anchor.
            # Channel by channel on contiguous planes: much cheaper than .all(axis=2).
            hit = np.abs(arr[..., 0] - colour[0]) <= self._tolerance
            for ch in range(1, arr.shape[2]):
                hit &= np.abs(arr[..., ch] - colour[ch]) <= self._tolerance
            cache[key] = hit
        return hit

    def _find_pattern(
        self,
        arr: NDArray[np.int16],
        p: _Pattern,
        cache: dict[tuple[int, ...], NDArray[np.bool_]],
    ) -> list[SpriteMatch]:
        fh, fw = arr.shape[:2]
        h, w = p.mask.shape
        if h > fh or w > fw:
            return []
        rows, cols = fh - h + 1, fw - w + 1
        tol = self._tolerance
        solid = int(p.mask.sum())
        # Cheap pass: anchor pixels across every spot at once.
        votes = np.zeros((rows, cols), np.int16)
        for ay, ax in p.anchors:
            near = self._near(arr, p.pixels[ay, ax], cache)
            votes += near[ay : ay + rows, ax : ax + cols]
        # A spot may miss as many anchors as the score allows solid pixels to miss.
        need = max(1, math.ceil(len(p.anchors) * self._min_score) - 1)
        if self._min_score >= 1.0:
            need = len(p.anchors)
        ys, xs = np.nonzero(votes >= need)
        if len(ys) == 0:
            return []
        windows = sliding_window_view(arr, (h, w, arr.shape[2]))[:, :, 0]
        found: list[SpriteMatch] = []
        for start in range(0, len(ys), _VERIFY_BATCH):
            by, bx = ys[start : start + _VERIFY_BATCH], xs[start : start + _VERIFY_BATCH]
            cand = windows[by, bx]  # n x h x w x c
            ok = (np.abs(cand - p.pixels) <= tol).all(axis=3) & p.mask
            scores = ok.reshape(len(by), -1).sum(axis=1) / solid
            for y, x, s in zip(by, bx, scores, strict=True):
                if s >= self._min_score - 1e-9:
                    found.append(
                        SpriteMatch(
                            p.template.name,
                            p.template.tag,
                            int(x),
                            int(y),
                            w,
                            h,
                            round(float(s), 4),
                            p.mirrored,
                        )
                    )
        return found


__all__ = [
    "TAG_WORDS",
    "SpriteFinder",
    "SpriteMatch",
    "SpriteTag",
    "SpriteTemplate",
]
