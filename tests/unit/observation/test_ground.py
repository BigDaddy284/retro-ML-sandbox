"""T-993: the new ground a go has covered, measured off the screen."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.ground import LARGEST, MOST_A_LOOK, SMALLEST, Ground, Kept

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


def test_on_a_still_screen_it_is_where_it_stands() -> None:
    # The screen never scrolls: it walks across it, 8 pixels a look.
    ground = started()
    assert not ground.scrolls
    assert [ground.look(STILL, (80 + 8 * n, 200), 16) for n in range(1, 5)] == [0, 1, 0, 1]
    assert ground.place == (32, 0) and ground.reach == (32, 0, 0, 0)
    # Up a ladder is new ground as much as along the floor.
    assert [ground.look(STILL, (112, 200 - 16 * n), 16) for n in range(1, 4)] == [1, 1, 1]
    assert ground.reach == (32, 0, 0, 48)


def test_once_the_screen_scrolls_only_the_cameras_travel_counts() -> None:
    ground = started()
    assert ground.look((16, 0), ME, 16) == 1 and ground.scrolls
    # A jump, and a fall into a pit: the player moves on the screen and the camera
    # does not. Neither is somewhere new.
    for y in (160, 130, 160, 200, 216, 232):
        assert ground.look(STILL, (80, y), 16) == 0
    assert ground.place == (16, 0) and ground.reach == (16, 0, 0, 0)
    # Gaining on the camera is not new ground either; the camera catching up is.
    assert ground.look(STILL, (120, 200), 16) == 0
    assert ground.look((16, 0), (120, 200), 16) == 1
    # It stays so for the goes after: the game scrolls, whatever this go has done yet.
    ground.begin()
    assert ground.scrolls
    assert ground.look(STILL, ME, 16) == 0 and ground.look(STILL, (80, 150), 16) == 0
    assert ground.reach == (0, 0, 0, 0)
    # What it reached on the screen before the first scroll stays reached.
    fresh = started()
    assert fresh.look(STILL, (112, 200), 16) == 2
    assert fresh.look((8, 0), (112, 200), 16) == 0  # 8 of camera is not past the 32
    assert fresh.place == (8, 0) and fresh.reach == (32, 0, 0, 0)


def test_unseen_it_counts_the_camera_alone() -> None:
    ground = started()
    # Out of sight for a look (behind something): the camera's travel still counts.
    assert ground.look((16, 0)) == 1
    assert ground.place == (16, 0)
    # Never seen at all there is no size to count steps in: nothing is paid, and what
    # was reached is counted (four steps a look at most) once it is seen.
    blind = Ground()
    blind.begin()
    assert [blind.look((16, 0)) for _ in range(3)] == [0, 0, 0]
    assert blind.unit is None and blind.reach == (48, 0, 0, 0)
    assert blind.look(STILL, ME, 16) == 3
    # On a still screen an unseen look leaves it where it was last seen.
    still = started()
    assert still.look(STILL, (96, 200), 16) == 1
    assert still.look(STILL) == 0 and still.place == (16, 0)


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


# ---------------------------------------------------------------- T-1000


def test_put_back_it_pays_only_past_its_furthest() -> None:
    ground = started()
    assert [ground.look((16, 0), ME, 16) for _ in range(2)] == [1, 1]
    kept = ground.keep()
    assert isinstance(kept, Kept) and ground.place == (32, 0)
    assert [ground.look((16, 0), ME, 16) for _ in range(4)] == [1, 1, 1, 1]
    assert ground.place == (96, 0) and ground.steps == 6
    # The game is put back to the kept moment: the place goes back, what was reached stays.
    ground.put_back(kept)
    assert ground.place == (32, 0)
    assert ground.reach == (96, 0, 0, 0) and ground.steps == 6
    # The same ground again pays nothing; the first step past the furthest does.
    assert [ground.look((16, 0), ME, 16) for _ in range(5)] == [0, 0, 0, 0, 1]
    assert ground.place == (112, 0) and ground.steps == 7
    # Put back more than once, and to a moment before it had moved at all.
    ground.put_back(kept)
    assert ground.place == (32, 0) and ground.steps == 7
    for bad in (None, (32, 0), "kept"):
        with pytest.raises(ConfigurationError):
            ground.put_back(bad)  # type: ignore[arg-type]


def test_put_back_on_a_screen_that_does_not_scroll() -> None:
    """There the place is where the player stands on the screen, from where it was
    first seen: that comes back too."""
    ground = Ground()
    ground.begin()
    ground.look(STILL, (40, 200), 16)
    ground.look(STILL, (72, 200), 16)
    kept = ground.keep()
    ground.look(STILL, (120, 200), 16)
    assert ground.place == (80, 0) and ground.steps == 5
    ground.put_back(kept)
    assert ground.place == (32, 0) and not ground.scrolls
    assert ground.look(STILL, (104, 200), 16) == 0  # 64 from where it was first seen
    assert ground.look(STILL, (136, 200), 16) == 1  # 96: a step past its furthest


def test_a_copy_is_its_own() -> None:
    ground = started()
    assert [ground.look((16, 0), ME, 16) for _ in range(3)] == [1, 1, 1]
    other = ground.copy()
    assert (other.place, other.reach, other.steps) == (ground.place, ground.reach, ground.steps)
    assert other.unit == 16 and other.scrolls
    assert [other.look((16, 0), ME, 16) for _ in range(2)] == [1, 1]
    assert other.steps == 5 and other.place == (80, 0)
    assert ground.steps == 3 and ground.place == (48, 0) and ground.reach == (48, 0, 0, 0)
    # And the other way round.
    assert ground.look((16, 0), ME, 16) == 1
    assert other.steps == 5
    # What was kept on the one can be put back on its copy.
    kept = ground.keep()
    other.put_back(kept)
    assert other.place == (64, 0) and other.reach == (80, 0, 0, 0)
