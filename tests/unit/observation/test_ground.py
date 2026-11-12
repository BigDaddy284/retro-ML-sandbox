"""T-993: the new ground a go has covered, measured off the screen."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.ground import LARGEST, MOST_A_LOOK, SMALLEST, Ground

ME = (80, 200)  # where the camera holds the player
STILL = (0, 0)


def started(size: int = 16) -> Ground:
    """A go under way: seen once, standing where the go began."""
    ground = Ground()
    ground.begin()
    assert ground.look(STILL, ME, size) == 0
    return ground


def test_walking_on_pays_a_step_for_each_of_its_own_size() -> None:
    ground = started()
    # The camera follows it 8 pixels a look: a step of 16 every other look.
    paid = [ground.look((8, 0), ME, 16) for _ in range(8)]
    assert paid == [0, 1, 0, 1, 0, 1, 0, 1]
    assert ground.steps == 4 and ground.place == (64, 0)
    assert ground.reach == (64, 0, 0, 0)
    # Standing still pays nothing, however long.
    assert [ground.look(STILL, ME, 16) for _ in range(5)] == [0] * 5


def test_the_same_ground_pays_once() -> None:
    ground = started()
    assert sum(ground.look((16, 0), ME, 16) for _ in range(5)) == 5
    # Back the way it came (no further left than where the go began), and on again:
    # nothing until it is past the 80 pixels it had reached.
    assert sum(ground.look((-16, 0), ME, 16) for _ in range(5)) == 0
    assert [ground.look((16, 0), ME, 16) for _ in range(6)] == [0, 0, 0, 0, 0, 1]
    assert ground.steps == 6


def test_each_direction_is_counted_on_its_own() -> None:
    for move, reach in (
        ((16, 0), (48, 0, 0, 0)),
        ((-16, 0), (0, 48, 0, 0)),
        ((0, 16), (0, 0, 48, 0)),
        ((0, -16), (0, 0, 0, 48)),
    ):
        ground = started()
        assert [ground.look(move, ME, 16) for _ in range(3)] == [1, 1, 1], move
        assert ground.reach == reach
    # Up and along in one look is a step of each.
    ground = started()
    assert ground.look((16, -16), ME, 16) == 2
    # Left of where the go began is new ground as much as right of it.
    ground = started()
    assert ground.look((-32, 0), ME, 16) == 2
    assert ground.look((64, 0), ME, 16) == 2  # 32 to the right of the start now
    assert ground.reach == (32, 32, 0, 0)


def test_a_jump_pays_no_more_than_four_steps() -> None:
    ground = started()
    # A misread camera (or a door): 160 pixels in one look pays 4 steps, not 10 ...
    assert ground.look((160, 0), ME, 16) == MOST_A_LOOK == 4
    # ... and the rest is not owed at the next look.
    assert ground.look(STILL, ME, 16) == 0
    assert ground.look((16, 0), ME, 16) == 1
    assert Ground(most=2).look((0, 0), ME, 16) == 0
    two = Ground(most=2)
    two.look(STILL, ME, 16)
    assert two.look((160, 0), ME, 16) == 2


def test_back_where_the_go_began_nothing_pays_until_past_the_furthest() -> None:
    ground = started()
    assert sum(ground.look((16, 0), ME, 16) for _ in range(10)) == 10
    ground.back()  # a life lost: the screen jumped to the start
    assert ground.place == (0, 0) and ground.reach == (160, 0, 0, 0) and ground.steps == 10
    # Seen somewhere else on the screen after the jump: that is where it starts from.
    assert ground.look(STILL, (40, 180), 16) == 0
    walk = [ground.look((16, 0), (40, 180), 16) for _ in range(12)]
    assert walk == [0] * 10 + [1, 1]


def test_a_new_go_starts_from_nothing() -> None:
    ground = started()
    assert sum(ground.look((16, 0), ME, 16) for _ in range(10)) == 10
    ground.begin()
    assert ground.place == (0, 0) and ground.reach == (0, 0, 0, 0) and ground.steps == 0
    assert ground.unit == 16  # its own size is not forgotten
    assert [ground.look((16, 0), ME, 16) for _ in range(3)] == [1, 1, 1]


def test_the_camera_and_the_place_on_the_screen_add_up() -> None:
    # A still screen: it walks across it, 8 pixels a look.
    ground = started()
    assert [ground.look(STILL, (80 + 8 * n, 200), 16) for n in range(1, 5)] == [0, 1, 0, 1]
    # Then the camera starts to follow while it still gains on the screen.
    assert ground.look((8, 0), (120, 200), 16) == 1  # 40 on the screen + 8 of camera
    assert ground.place == (48, 0)
    # The camera slides on while it stands still in the level: on the screen it
    # slips back as far as the camera went, and nothing is new.
    assert ground.look((16, 0), (104, 200), 16) == 0
    assert ground.place == (48, 0)


def test_unseen_it_counts_the_camera_alone() -> None:
    ground = started()
    assert ground.look(STILL, (96, 200), 16) == 1
    # Out of sight for a look (behind something): it is taken to be where it was on
    # the screen, and the camera's travel still counts.
    assert ground.look((16, 0)) == 1
    assert ground.place == (32, 0)
    assert ground.look(STILL, (96, 200), 16) == 0
    # Never seen at all there is no size to count steps in: nothing is paid, and what
    # was reached is counted (four steps a look at most) once it is seen.
    blind = Ground()
    blind.begin()
    assert [blind.look((16, 0)) for _ in range(3)] == [0, 0, 0]
    assert blind.unit is None and blind.reach == (48, 0, 0, 0)
    assert blind.look(STILL, ME, 16) == 3


def test_the_step_is_its_own_size() -> None:
    small = started(size=8)
    assert small.unit == 8 and small.look((16, 0), ME, 8) == 2
    big = started(size=24)
    assert big.look((16, 0), ME, 24) == 0 and big.look((16, 0), ME, 24) == 1
    # A speck or a giant is kept within sensible steps, and the first size stays.
    assert started(size=1).unit == SMALLEST and started(size=200).unit == LARGEST
    kept = started(size=16)
    kept.look(STILL, ME, 30)
    assert kept.unit == 16


def test_nonsense_is_refused() -> None:
    ground = Ground()
    for moved in ((1,), (1, 2, 3), [1, 2], (1.5, 0), (True, 0), None, "right"):
        with pytest.raises(ConfigurationError):
            ground.look(moved)  # type: ignore[arg-type]
    for me in ((1,), (1.0, 2), [1, 2], (False, 1)):
        with pytest.raises(ConfigurationError):
            ground.look(STILL, me)  # type: ignore[arg-type]
    for size in (0, -3, 1.5, True):
        with pytest.raises(ConfigurationError):
            ground.look(STILL, ME, size)  # type: ignore[arg-type]
    for most in (0, -1, 1.5, True):
        with pytest.raises(ConfigurationError):
            Ground(most=most)  # type: ignore[arg-type]
