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


def bar(a: np.ndarray, where: str, count: int = 0) -> None:
    """Paint a solid status bar along one edge of the screen ``a``: three lives and a
    counter that shows ``count``."""
    if where in ("top", "bottom"):
        y = 0 if where == "top" else H - TALL
        a[y : y + TALL] = BAR
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
) -> Frame:
    """The screen with the camera at ``camera`` (across, down) in ``world``, the things
    drawn where they are on the screen, and the bar over it all."""
    x, y = camera
    a = world[y : y + H, x : x + W].copy()
    for tx, ty, picture in things:
        h, w = picture.shape[:2]
        a[ty : ty + h, tx : tx + w] = picture
    if where is not None:
        bar(a, where, count)
    return Frame(W, H, 3, a.tobytes())


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
def searched(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """How many times the shifts were searched through: one entry a search."""
    calls: list[int] = []
    real = tracking._search

    def counted(*args: object, **kwargs: object) -> tuple[int, int]:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tracking, "_search", counted)
    yield calls


__all__ = ["ConfigurationError", "estimate_scroll", "time"]
