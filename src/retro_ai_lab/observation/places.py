"""Whether two screens show the same spot (T-1038).

A go can pass through more places than one: the next level, a room behind a
door. The ground it covers is counted place by place
(:mod:`retro_ai_lab.observation.ground`), and whoever counts must know, when
the screen has jumped, whether it is somewhere new, back somewhere it has been,
or where it was all along (a flash). Nothing in a game's memory is asked: the
screen at a place's door is how the place is known again.

:func:`same_view` compares two screens, allowing for a camera move between
them. What is the same plain backdrop on both has no say: two rooms drawn on
black are black alike over most of the screen, and so are two levels under one
sky. What counts is what is drawn on either, and how much of that differs.
"""

from __future__ import annotations

import numpy as np

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.tracking import _overlap, _packed, estimate_scroll

SAME_REACH = 32  # pixels the camera may have moved between two screens of one spot
SAME_MOST = 0.2  # share of what is drawn on either that may differ: things move, counters count


def _differing(a: Frame, b: Frame, shift: tuple[int, int]) -> float:
    """The share of what is drawn on either screen that differs with ``b`` laid over
    ``a`` moved by ``shift``; 0.0 where both are the same plain backdrop all over."""
    after, before = _packed(b), _packed(a)
    aa, bb = _overlap(after, before, *shift)
    if aa.size == 0:
        return 1.0
    colours, counts = np.unique(aa, return_counts=True)
    backdrop = colours[int(np.argmax(counts))]
    drawn = (aa != backdrop) | (bb != backdrop)
    said = int(np.count_nonzero(drawn))
    if said == 0:
        return 0.0
    return float(np.count_nonzero((aa != bb) & drawn)) / said


def same_view(a: Frame, b: Frame, *, reach: int = SAME_REACH, most: float = SAME_MOST) -> bool:
    """Whether screens ``a`` and ``b`` show the same spot: as they are, or with the
    camera moved by up to ``reach`` pixels either way, no more than ``most`` of what
    is drawn on either differs."""
    if not isinstance(a, Frame) or not isinstance(b, Frame):
        raise ConfigurationError("Two screens are compared.")
    if (a.width, a.height, a.channels) != (b.width, b.height, b.channels):
        raise ConfigurationError("Both screens must have the same size and channels.")
    if isinstance(reach, bool) or not isinstance(reach, int) or reach < 0:
        raise ConfigurationError("reach must be a whole number of pixels, 0 or more.")
    if isinstance(most, bool) or not isinstance(most, (int, float)) or not 0 <= most < 1:
        raise ConfigurationError("most must be a share from 0 to less than 1.")
    if a.pixels == b.pixels or _differing(a, b, (0, 0)) <= most:
        return True
    if reach == 0:
        return False
    shift = estimate_scroll(a, b, max_dx=reach, max_dy=reach)
    return shift != (0, 0) and _differing(a, b, shift) <= most


__all__ = ["SAME_MOST", "SAME_REACH", "same_view"]
