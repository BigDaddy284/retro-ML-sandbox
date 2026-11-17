"""The screen as a learner sees it (T-1055).

Every game shows a screen, whatever it keeps in its memory. :func:`shrink` turns a
captured screen of any console's size into one small grey picture of a fixed
size, so one network fits every game; a :class:`Sight` is what a situation shows:
the picture now, the picture a step earlier (what moved is the difference) and
the move just made (a game that wants a fresh press of a button looks the same
whether the button is still down or not).

Nothing here knows what is on the screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retro_ai_lab.core import ConfigurationError, Frame

PICTURE_WIDTH = 32
PICTURE_HEIGHT = 28
PICTURE_SIZE = PICTURE_WIDTH * PICTURE_HEIGHT
NO_MOVE = -1  # no move was made yet: a go's first sight


def _edges(source: int, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Where each of ``cells`` cells begins and ends along ``source`` pixels: in
    proportion, nothing cropped, and at least one pixel each (a screen smaller than
    the picture has its pixels repeated)."""
    marks = np.arange(cells + 1) * source // cells
    first = marks[:-1]
    last = np.maximum(marks[1:], first + 1)
    return first, last


def shrink(frame: Frame) -> bytes:
    """The screen as ``PICTURE_WIDTH`` by ``PICTURE_HEIGHT`` grey bytes, row by row.

    Grey is the mean of the colour channels (alpha is left out); a cell is the mean
    of the part of the screen it covers.
    """
    if not isinstance(frame, Frame):
        raise ConfigurationError("Only a captured screen can be shrunk to a picture.")
    pixels = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
        frame.height, frame.width, frame.channels
    )
    colour = pixels[:, :, :3] if frame.channels >= 3 else pixels
    grey = colour.astype(np.float64).mean(axis=2)
    # Sums over any block in four look-ups: the running total down and across.
    total = np.zeros((frame.height + 1, frame.width + 1))
    total[1:, 1:] = grey.cumsum(axis=0).cumsum(axis=1)
    top, bottom = _edges(frame.height, PICTURE_HEIGHT)
    left, right = _edges(frame.width, PICTURE_WIDTH)
    sums = (
        total[bottom[:, None], right[None, :]]
        - total[top[:, None], right[None, :]]
        - total[bottom[:, None], left[None, :]]
        + total[top[:, None], left[None, :]]
    )
    area = (bottom - top)[:, None] * (right - left)[None, :]
    return bytes(np.rint(sums / area).clip(0, 255).astype(np.uint8).tobytes())


@dataclass(frozen=True, slots=True)
class Sight:
    """What a situation shows: the picture now, the one a step earlier, the move made."""

    now: bytes
    before: bytes
    held: int = NO_MOVE  # the index of the move just made, NO_MOVE at a go's start

    def __post_init__(self) -> None:
        for picture, what in ((self.now, "now"), (self.before, "before")):
            if not isinstance(picture, bytes) or len(picture) != PICTURE_SIZE:
                raise ConfigurationError(
                    f"A sight's picture ({what}) must be {PICTURE_SIZE} bytes, "
                    f"{PICTURE_WIDTH} by {PICTURE_HEIGHT}."
                )
        if isinstance(self.held, bool) or not isinstance(self.held, int) or self.held < NO_MOVE:
            raise ConfigurationError("A sight's move must be a move's number, or NO_MOVE.")


__all__ = ["NO_MOVE", "PICTURE_HEIGHT", "PICTURE_SIZE", "PICTURE_WIDTH", "Sight", "shrink"]
