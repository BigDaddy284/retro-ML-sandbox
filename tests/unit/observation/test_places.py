"""T-1038: whether two screens show the same spot."""

from __future__ import annotations

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.places import SAME_MOST, SAME_REACH, same_view
from tests.unit.training.test_vision import HERO, WALKER, flat, indoors, level

HELD = [(80, 192, HERO)]


def plain(colour: tuple[int, int, int], width: int = 256, height: int = 240) -> Frame:
    a = np.zeros((height, width, 3), np.uint8)
    a[:] = colour
    return Frame(width, height, 3, a.tobytes())


def test_the_same_screen_is_the_same_view() -> None:
    assert SAME_REACH == 32 and SAME_MOST == 0.2
    assert same_view(level(100, HELD), level(100, HELD))
    assert same_view(indoors(400, HELD), indoors(400, HELD))
    # Things that moved, came or went do not make it another spot.
    assert same_view(level(100, HELD), level(100, [(120, 150, HERO), (200, 194, WALKER)]))
    assert same_view(level(100, []), level(100, HELD))
    # Two screens of one plain colour: nothing is drawn on either to tell them apart.
    assert same_view(plain((0, 0, 0)), plain((0, 0, 0)))


def test_a_view_the_camera_moved_over_is_the_same() -> None:
    for moved in (1, 7, 16, 32, -5, -32):
        assert same_view(level(100, HELD), level(100 + moved, HELD)), moved
        assert same_view(indoors(400, HELD), indoors(400 + moved, HELD)), moved
    # With no reach at all only the screen as it is counts.
    assert same_view(level(100, HELD), level(100, HELD), reach=0)
    assert not same_view(level(100, HELD), level(107, HELD), reach=0)


def test_another_room_is_another_view() -> None:
    assert not same_view(level(100, HELD), indoors(400, HELD))
    assert not same_view(indoors(400, HELD), level(100, HELD))
    assert not same_view(level(100, HELD), flat(0, HELD))
    # A blank screen is no room, and neither is a blank of another colour.
    assert not same_view(level(100, HELD), plain((0, 0, 0)))
    assert not same_view(plain((0, 0, 0)), indoors(400, HELD))
    assert not same_view(plain((0, 0, 0)), plain((252, 252, 252)))
    # Two rooms drawn on the same black are not one for all the black they share.
    lamps = np.zeros((240, 256, 3), np.uint8)
    lamps[40:60, 20:236:30] = (252, 188, 60)
    bars = np.zeros((240, 256, 3), np.uint8)
    bars[200:210, 10:246] = (0, 168, 0)
    one, other = Frame(256, 240, 3, lamps.tobytes()), Frame(256, 240, 3, bars.tobytes())
    assert not same_view(one, other) and same_view(one, one) and same_view(other, other)
    # How much may differ is the caller's to say.
    assert same_view(level(100, HELD), flat(0, HELD), most=0.99)
    assert not same_view(level(100, []), level(100, HELD), most=0.0)
    assert same_view(level(100, HELD), level(100, HELD), most=0.0)
    assert same_view(level(100, []), level(116, []), most=0.0)  # the strip brought in has no say


def test_nonsense_is_refused() -> None:
    a = level(100, HELD)
    with pytest.raises(ConfigurationError, match="same size"):
        same_view(a, plain((0, 0, 0), 128, 120))
    for bad in (None, a.pixels, "screen"):
        with pytest.raises(ConfigurationError, match="Two screens"):
            same_view(a, bad)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="Two screens"):
            same_view(bad, a)  # type: ignore[arg-type]
    for bad in (-1, 1.5, True, "8"):
        with pytest.raises(ConfigurationError, match="reach"):
            same_view(a, a, reach=bad)  # type: ignore[arg-type]
    for bad in (-0.1, 1, 1.5, True, False, "0.2"):
        with pytest.raises(ConfigurationError, match="most"):
            same_view(a, a, most=bad)  # type: ignore[arg-type]
