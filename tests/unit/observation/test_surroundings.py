"""T-989: what is around the player, read off one screen."""

from __future__ import annotations

import time

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.surroundings import (
    FAR_BELOW,
    MOST,
    NEAR_BELOW,
    NOTHING_BELOW,
    ON_SOMETHING,
    STEP,
    Surroundings,
    read,
)

SKY = (92, 148, 252)
BROWN = (136, 84, 24)
GREEN = (0, 168, 0)
GREY = (188, 188, 188)
RED = (228, 0, 88)

W, H = 256, 240
GROUND = 208
HERO_W, HERO_H = 12, 16
X = 100  # where the hero stands in most scenes


def scene(gaps: tuple[tuple[int, int], ...] = ()) -> np.ndarray:
    """Sky over ground from y=208 with a grass top; ``gaps`` are (first x, width)."""
    a = np.zeros((H, W, 3), np.uint8)
    a[:] = SKY
    a[GROUND:] = BROWN
    a[GROUND : GROUND + 4] = GREEN
    for x, wide in gaps:
        a[GROUND:, max(0, x) : max(0, x + wide)] = SKY
    return a


def hero(a: np.ndarray, x: int = X, y: int = GROUND - HERO_H) -> tuple[int, int, int, int]:
    a[max(0, y) : max(0, y + HERO_H), max(0, x) : max(0, x + HERO_W)] = RED
    return (x, y, HERO_W, HERO_H)


def frame(a: np.ndarray) -> Frame:
    return Frame(a.shape[1], a.shape[0], 3, a.tobytes())


def around(a: np.ndarray, box: tuple[int, int, int, int]) -> Surroundings:
    found = read(frame(a), box, SKY)
    assert found is not None
    return found


def test_flat_ground_is_floor_and_clear_both_ways() -> None:
    a = scene()
    assert around(a, hero(a)) == Surroundings(MOST, MOST, MOST, MOST, ON_SOMETHING)
    assert around(a, hero(a)).parts() == (
        ("screen:floor right", MOST),
        ("screen:floor left", MOST),
        ("screen:clear right", MOST),
        ("screen:clear left", MOST),
        ("screen:under", ON_SOMETHING),
    )


@pytest.mark.parametrize("apart", [0, 1, 7, 8, 9, 20, 40, 47, 48, 60])
def test_a_gap_is_read_at_each_distance_either_side(apart: int) -> None:
    # ``apart`` pixels of floor between the first column read (one past its own edge
    # and outline) and the gap: counted in steps of 8, and from 48 on it is "plenty".
    steps = min(apart // STEP, MOST)
    a = scene(gaps=((X + HERO_W + 1 + apart, 32),))
    found = around(a, hero(a))
    assert (found.floor_right, found.floor_left) == (steps, MOST)
    assert (found.clear_right, found.clear_left, found.under) == (MOST, MOST, ON_SOMETHING)
    b = scene(gaps=((X - 1 - apart - 32, 32),))
    found = around(b, hero(b))
    assert (found.floor_right, found.floor_left) == (MOST, steps)


@pytest.mark.parametrize("apart", [0, 5, 8, 23, 24, 47, 48, 90])
def test_a_wall_is_read_at_each_distance_either_side(apart: int) -> None:
    steps = min(apart // STEP, MOST)
    a = scene()
    wall = X + HERO_W + 1 + apart
    a[GROUND - 16 : GROUND, wall : wall + 16] = GREY
    found = around(a, hero(a))
    assert (found.clear_right, found.clear_left) == (steps, MOST)
    assert (found.floor_right, found.floor_left) == (MOST, MOST)  # a wall is not a drop
    b = scene()
    wall = X - 1 - apart
    b[GROUND - 16 : GROUND, wall - 16 : wall] = GREY
    found = around(b, hero(b))
    assert (found.clear_right, found.clear_left) == (MOST, steps)


def test_a_low_step_is_not_in_the_way() -> None:
    # Ground a quarter of its height higher just ahead (a slope, a kerb): it walks up
    # that. One pixel more and it is in the way.
    a = scene()
    a[GROUND - HERO_H // 4 : GROUND, X + 20 : X + 60] = BROWN
    assert around(a, hero(a)).clear_right == MOST
    a[GROUND - HERO_H // 4 - 1 : GROUND, X + 20 : X + 60] = BROWN
    assert around(a, hero(a)).clear_right == (20 - HERO_W - 1) // STEP
    # Something overhead, above its head, is not in the way either.
    b = scene()
    b[GROUND - 60 : GROUND - HERO_H, X + 20 : X + 60] = GREY
    assert around(b, hero(b)).clear_right == MOST
    # A coin hanging at head height is: the reading cannot tell what a thing is.
    b[GROUND - HERO_H : GROUND - HERO_H + 4, X + 30 : X + 34] = GREY
    assert around(b, hero(b)).clear_right == (30 - HERO_W - 1) // STEP


def test_under_it_standing_near_far_and_nothing() -> None:
    for up, under in (
        (0, ON_SOMETHING),
        (2, ON_SOMETHING),  # a sprite drawn a pixel or two above the ground stands on it
        (3, NEAR_BELOW),
        (HERO_H, NEAR_BELOW),  # within its own height
        (HERO_H + 1, FAR_BELOW),
        (120, FAR_BELOW),
    ):
        a = scene()
        found = around(a, hero(a, y=GROUND - HERO_H - up))
        assert found.under == under, up
        # Floor is floor however high above it it is.
        assert (found.floor_right, found.floor_left) == (MOST, MOST)
    # Standing on a wall block one tile up: on something, and the ground beside it
    # is still floor.
    a = scene()
    a[GROUND - 16 : GROUND, X : X + 16] = GREY
    found = around(a, hero(a, y=GROUND - 16 - HERO_H))
    assert found == Surroundings(MOST, MOST, MOST, MOST, ON_SOMETHING)


def test_over_a_gap_there_is_no_floor_under_or_beside() -> None:
    # In the air over the middle of a 64-wide gap: nothing under it, no floor at its
    # edges, and the far side 2 steps off (20 pixels past its right edge and outline).
    a = scene(gaps=((X - 19, 64),))
    found = around(a, hero(a, y=150))
    assert found.under == NOTHING_BELOW
    assert (found.floor_right, found.floor_left) == (0, 0)
    assert (found.clear_right, found.clear_left) == (MOST, MOST)
    # Half over the edge: its left half still has ground under it.
    b = scene(gaps=((X + 6, 64),))
    found = around(b, hero(b))
    assert (found.under, found.floor_right, found.floor_left) == (ON_SOMETHING, 0, MOST)
    # Down in the gap, below the ground's top: the gap's sides are walls beside it.
    c = scene(gaps=((X - 10, 40),))
    found = around(c, hero(c, y=GROUND + 4))
    assert found.under == NOTHING_BELOW
    assert (found.clear_right, found.clear_left) == (
        (40 - 10 - HERO_W - 1) // STEP,
        (10 - 1) // STEP,
    )


def test_the_edge_of_the_screen_counts_as_floor_and_clear() -> None:
    # It cannot see past the edge, so it reads no drop and nothing in the way there.
    a = scene()
    assert around(a, hero(a, x=W - HERO_W)) == Surroundings(MOST, MOST, MOST, MOST, ON_SOMETHING)
    assert around(a, hero(a, x=0)) == Surroundings(MOST, MOST, MOST, MOST, ON_SOMETHING)
    # A gap that runs off the screen is still read up to the edge.
    b = scene(gaps=((W - 20, 40),))
    found = around(b, hero(b, x=W - 60))
    assert found.floor_right == (60 - 20 - HERO_W - 1) // STEP


def test_at_the_bottom_or_partly_off_the_screen() -> None:
    # Falling out of the bottom: only the part in sight is its box, nothing is under it.
    a = scene(gaps=((X - 20, 64),))
    box = hero(a, y=H - 6)
    found = around(a, (box[0], box[1], HERO_W, 6))
    assert (found.under, found.floor_right, found.floor_left) == (NOTHING_BELOW, 0, 0)
    # A box that reaches past the bottom, the top or a side is read where it is in sight.
    assert around(a, (X, H - 6, HERO_W, HERO_H)).under == NOTHING_BELOW
    top = scene()
    assert around(top, hero(top, y=-8)).under == FAR_BELOW
    side = scene()
    assert around(side, hero(side, x=-6)) == Surroundings(MOST, MOST, MOST, MOST, ON_SOMETHING)
    # Wholly off the screen there is nothing to read.
    for gone in ((-12, 100, 12, 16), (W, 100, 12, 16), (100, H, 12, 16), (100, -16, 12, 16)):
        assert read(frame(scene()), gone, SKY) is None


def test_bad_input_is_refused() -> None:
    good = frame(scene())
    box = (X, GROUND - HERO_H, HERO_W, HERO_H)
    with pytest.raises(ConfigurationError):
        read("screen", box, SKY)  # type: ignore[arg-type]
    for bad in (
        (1, 2, 3),
        (1, 2, 0, 4),
        (1, 2, 3, 0),
        (1.5, 2, 3, 4),
        (True, 2, 3, 4),
        [1, 2, 3, 4],
    ):
        with pytest.raises(ConfigurationError):
            read(good, bad, SKY)  # type: ignore[arg-type]
    for margin in (-1, 1.5, True):
        with pytest.raises(ConfigurationError):
            read(good, box, SKY, margin=margin)  # type: ignore[arg-type]
    # No backdrop colour, or one of another depth than the screen: no reading.
    assert read(good, box, None) is None
    assert read(good, box, (0,)) is None
    # The margin is how much beside its box goes unread (its outline).
    a = scene()
    a[GROUND - 16 : GROUND, X + HERO_W] = (0, 0, 0)  # a black outline on its right
    assert around(a, hero(a)).clear_right == MOST
    assert read(frame(a), box, SKY, margin=0) == Surroundings(MOST, MOST, 0, MOST, ON_SOMETHING)


def test_reading_is_quick() -> None:
    a = scene(gaps=((150, 48),))
    box = hero(a)
    screen = frame(a)
    began = time.perf_counter()
    for _ in range(200):
        read(screen, box, SKY)
    assert (time.perf_counter() - began) / 200 < 0.01  # a look already costs ~15 ms
