"""What is around the player, read off one screen (T-989).

The brain could see the nearest moving thing (T-978) and nothing of the ground:
a gap it was about to walk into looked the same as flat floor, so the only way
to learn where the gaps are was to remember where in the level it stood, and
that does not carry over to the next level or the next game.

:func:`read` takes one screen, the box the player was seen in and the colour of
the backdrop, and answers five small questions about the picture next to that
box. Nothing here knows a game: no colour is named, both sides are read alike
and every distance is counted from the player's own edges in the player's own
size.

Known limits. A game that draws a strip along the bottom of the screen never
shows an open column (the strip is under every gap). Scenery that is only
decoration (a bush, a far hill) reads as something in the way; the brain can
learn that a jump there costs nothing, and telling the two apart is a later
task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.screen import to_array

Box = tuple[int, int, int, int]

STEP = 8  # pixels in one step of distance
MOST = 6  # steps counted: this many or more, or nothing in sight, reads as MOST
MARGIN = 1  # pixels next to the box left unread (the player's own outline)
STANDING = 2  # pixels under its feet within which it stands on something

ON_SOMETHING = 0
NEAR_BELOW = 1  # something within its own height below it
FAR_BELOW = 2
NOTHING_BELOW = 3  # only backdrop from its feet to the bottom of the screen


@dataclass(frozen=True)
class Surroundings:
    """What :func:`read` found. Distances are steps of ``STEP`` pixels, 0 to ``MOST``."""

    floor_right: int  # floor before the first drop to its right
    floor_left: int
    clear_right: int  # clear way at its own height before something stands in it
    clear_left: int
    under: int  # ON_SOMETHING, NEAR_BELOW, FAR_BELOW or NOTHING_BELOW

    def parts(self) -> tuple[tuple[str, int], ...]:
        """The readings as state parts for the brain, in a fixed order."""
        return (
            ("screen:floor right", self.floor_right),
            ("screen:floor left", self.floor_left),
            ("screen:clear right", self.clear_right),
            ("screen:clear left", self.clear_left),
            ("screen:under", self.under),
        )


def read(
    frame: Frame, box: Box, backdrop: tuple[int, ...] | None, *, margin: int = MARGIN
) -> Surroundings | None:
    """Read what is around ``box`` (x, y, width, height) on ``frame``.

    None when there is no backdrop colour to tell things from (or one of another
    depth than the screen), or the box is wholly off the screen.
    """
    if not isinstance(frame, Frame):
        raise ConfigurationError("The surroundings are read off a screen.")
    if (
        not isinstance(box, tuple)
        or len(box) != 4
        or any(isinstance(v, bool) or not isinstance(v, int) for v in box)
        or box[2] < 1
        or box[3] < 1
    ):
        raise ConfigurationError("The player's box is (x, y, width, height) in whole pixels.")
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ConfigurationError("margin must be a whole number, 0 or more.")
    if backdrop is None or len(backdrop) != frame.channels:
        return None
    x, y, w, h = box
    if x >= frame.width or x + w <= 0 or y >= frame.height or y + h <= 0:
        return None
    thing: NDArray[np.bool_] = (to_array(frame) != np.array(backdrop, np.uint8)).any(axis=2)
    feet = max(0, y + h)
    # A column with anything in it from the player's feet down holds floor.
    floor = thing[feet:].any(axis=0)
    # Its own height, less the bottom quarter: a floor a little higher is no wall.
    top, bottom = max(0, y), max(0, y + h - h // 4)
    way = thing[top:bottom].any(axis=0) if bottom > top else np.zeros(frame.width, np.bool_)
    left, right = x - margin, x + w + margin  # the first columns read on either side
    below = thing[feet:, max(0, x) : max(0, x + w)].any(axis=1)
    return Surroundings(
        floor_right=_steps(~floor[max(0, right) :]),
        floor_left=_steps(~floor[: max(0, left)][::-1]),
        clear_right=_steps(way[max(0, right) :]),
        clear_left=_steps(way[: max(0, left)][::-1]),
        under=_under(below, h),
    )


def _steps(found: NDArray[np.bool_]) -> int:
    """Steps to the first True in ``found`` (nearest first); ``MOST`` when there is none
    within reach. Past the end of the screen nothing is found: it cannot be seen."""
    near = found[: MOST * STEP]
    if not near.any():
        return MOST
    return int(np.argmax(near)) // STEP


def _under(below: NDArray[np.bool_], height: int) -> int:
    if not below.any():
        return NOTHING_BELOW
    depth = int(np.argmax(below))
    if depth <= STANDING:
        return ON_SOMETHING
    return NEAR_BELOW if depth <= height else FAR_BELOW


__all__ = [
    "FAR_BELOW",
    "MARGIN",
    "MOST",
    "NEAR_BELOW",
    "NOTHING_BELOW",
    "ON_SOMETHING",
    "STANDING",
    "STEP",
    "Surroundings",
    "read",
]
