"""T-1038: the ground of a go, place by place."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.ground import Ground, Kept

ME = (80, 200)  # where the camera holds the player
STILL = (0, 0)
RIGHT, LEFT, DOWN, UP = (16, 0), (-16, 0), (0, 16), (0, -16)


def started() -> Ground:
    """A go under way: seen once, standing where the go began."""
    ground = Ground()
    ground.begin()
    assert ground.look(STILL, ME, 16) == 0
    return ground


def walk(ground: Ground, move: tuple[int, int], looks: int) -> list[int]:
    return [ground.look(move, ME, 16) for _ in range(looks)]


def test_a_new_place_pays_from_its_door() -> None:
    ground = started()
    assert walk(ground, RIGHT, 5) == [1] * 5
    assert (ground.place_number, ground.places) == (0, 1)
    # Somewhere it has not been: at the door, with nothing reached there.
    assert ground.new_place() == 1
    assert (ground.place_number, ground.places) == (1, 2)
    assert ground.place == (0, 0) and ground.reach == (0, 0, 0, 0)
    assert ground.steps == 5  # what the go counted elsewhere is still counted
    assert ground.look(STILL, ME, 16) == 0
    # The first step from the door pays, whatever was reached elsewhere, each way.
    assert walk(ground, RIGHT, 3) == [1, 1, 1]
    assert ground.reach == (48, 0, 0, 0) and ground.steps == 8
    assert walk(ground, LEFT, 4) == [0, 0, 0, 1]  # back to the door, and one past it
    assert walk(ground, DOWN, 2) == [1, 1] and walk(ground, UP, 3) == [0, 0, 1]
    assert ground.reach == (48, 16, 32, 16) and ground.steps == 12
    # And another: its number is the next.
    assert ground.new_place() == 2 and ground.places == 3
    assert walk(ground, LEFT, 2) == [1, 1] and ground.steps == 14


def test_a_place_it_has_been_in_pays_only_past_what_it_reached_there() -> None:
    ground = started()
    assert walk(ground, RIGHT, 5) == [1] * 5
    ground.new_place()
    assert walk(ground, RIGHT, 2) == [1, 1]
    # Back in the first place, at its door: what it reached there stays reached.
    ground.go_to(0)
    assert ground.place_number == 0 and ground.place == (0, 0)
    assert ground.reach == (80, 0, 0, 0)
    assert walk(ground, RIGHT, 6) == [0, 0, 0, 0, 0, 1]
    # And in the second: two steps were reached there.
    ground.go_to(1)
    assert ground.place_number == 1 and ground.reach == (32, 0, 0, 0)
    assert walk(ground, RIGHT, 3) == [0, 0, 1]
    assert ground.steps == 5 + 2 + 1 + 1 and ground.places == 2
    # To the place it is in: its door.
    ground.go_to(1)
    assert ground.place == (0, 0) and ground.reach == (48, 0, 0, 0)


def test_back_is_the_door_of_the_place_it_is_in() -> None:
    ground = started()
    assert walk(ground, RIGHT, 5) == [1] * 5
    ground.new_place()
    assert walk(ground, RIGHT, 2) == [1, 1]
    ground.back()
    assert ground.place_number == 1 and ground.place == (0, 0)
    assert ground.reach == (32, 0, 0, 0)
    assert walk(ground, RIGHT, 3) == [0, 0, 1]
    # On a screen that does not scroll, a new place is counted from where it is first
    # seen there.
    still = Ground()
    still.begin()
    still.look(STILL, (40, 200), 16)
    assert still.look(STILL, (72, 200), 16) == 2
    still.new_place()
    assert still.look(STILL, (200, 100), 16) == 0
    assert still.look(STILL, (184, 100), 16) == 1 and still.reach == (0, 16, 0, 0)


def test_a_new_go_has_one_place() -> None:
    ground = started()
    assert walk(ground, RIGHT, 3) == [1, 1, 1]
    ground.new_place()
    assert walk(ground, RIGHT, 2) == [1, 1]
    ground.begin()
    assert (ground.place_number, ground.places, ground.steps) == (0, 1, 0)
    assert ground.reach == (0, 0, 0, 0) and ground.unit == 16 and ground.scrolls
    ground.look(STILL, ME, 16)
    assert walk(ground, RIGHT, 2) == [1, 1]


def test_put_back_returns_to_the_place_kept() -> None:
    ground = started()
    assert walk(ground, RIGHT, 2) == [1, 1]
    kept = ground.keep()
    assert isinstance(kept, Kept) and kept.place == 0
    assert walk(ground, RIGHT, 2) == [1, 1]
    ground.new_place()
    assert walk(ground, RIGHT, 3) == [1, 1, 1]
    inside = ground.keep()
    assert inside.place == 1
    assert walk(ground, RIGHT, 1) == [1]
    # Put back to before the door: the first place again, at the spot kept.
    ground.put_back(kept)
    assert ground.place_number == 0 and ground.place == (32, 0)
    assert ground.reach == (64, 0, 0, 0) and ground.steps == 8 and ground.places == 2
    assert walk(ground, RIGHT, 3) == [0, 0, 1]
    # Through the door again is the caller's to say; the ground there pays nothing twice.
    ground.go_to(1)
    assert walk(ground, RIGHT, 5) == [0, 0, 0, 0, 1]
    # Put back to the moment inside.
    ground.put_back(inside)
    assert ground.place_number == 1 and ground.place == (48, 0)
    assert ground.reach == (80, 0, 0, 0)
    assert walk(ground, RIGHT, 3) == [0, 0, 1]


def test_the_story_of_each_place_goes_back() -> None:
    """What a go played straight would be paid, where the go itself was paid before it
    was put back (T-1028), place by place."""
    ground = started()
    assert walk(ground, RIGHT, 2) == [1, 1]
    kept = ground.keep()
    assert walk(ground, RIGHT, 2) == [1, 1]
    ground.new_place()
    assert walk(ground, RIGHT, 3) == [1, 1, 1]
    inside = ground.keep()
    assert walk(ground, RIGHT, 2) == [1, 1]
    # Back to before the door: the story had got two steps into the first place and
    # nowhere in the second.
    ground.put_back(kept)
    again = []
    for _ in range(2):
        assert ground.look(RIGHT, ME, 16) == 0
        again.append(ground.again)
    assert again == [1, 1]
    ground.go_to(1)
    again = []
    for _ in range(5):
        assert ground.look(RIGHT, ME, 16) == 0
        again.append(ground.again)
    assert again == [1] * 5
    assert ground.look(RIGHT, ME, 16) == 1 and ground.again == 0
    # Back to the moment inside: the story had three steps of the second place, and all
    # four of the first.
    ground.put_back(inside)
    again = []
    for _ in range(3):
        assert ground.look(RIGHT, ME, 16) == 0
        again.append(ground.again)
    assert again == [1, 1, 1]  # the three it walked on to after that moment
    ground.go_to(0)
    for _ in range(4):
        assert ground.look(RIGHT, ME, 16) == 0 and ground.again == 0
    # A moment kept in a go with more places than this one has: they are made, empty.
    other = Ground()
    other.begin()
    other.look(STILL, ME, 16)
    assert other.look(RIGHT, ME, 16) == 1
    other.put_back(inside)
    assert (other.place_number, other.places) == (1, 2)
    assert other.place == (48, 0) and other.reach == (0, 0, 0, 0) and other.steps == 1
    assert other.look(RIGHT, ME, 16) == 4 and other.again == 0  # 64 from its door, at most 4


def test_a_moment_written_by_hand() -> None:
    """A kept moment made without the list of places (as before T-1038) is one place's:
    its own reach and count are the story's there."""
    ground = started()
    assert walk(ground, RIGHT, 4) == [1] * 4
    ground.put_back(Kept((32, 0), None, (0, 0), (32, 0, 0, 0), (2, 0, 0, 0)))
    assert ground.place == (32, 0) and ground.place_number == 0
    again = []
    for _ in range(3):
        last = ground.look(RIGHT, ME, 16)
        again.append((last, ground.again))
    assert again == [(0, 1), (0, 1), (1, 0)]


def test_a_copy_keeps_its_places_to_itself() -> None:
    ground = started()
    assert walk(ground, RIGHT, 2) == [1, 1]
    ground.new_place()
    assert walk(ground, RIGHT, 1) == [1]
    other = ground.copy()
    assert (other.place_number, other.places, other.steps) == (1, 2, 3)
    assert other.place == ground.place and other.reach == ground.reach
    assert walk(other, RIGHT, 2) == [1, 1]
    other.go_to(0)
    assert walk(other, RIGHT, 3) == [0, 0, 1]
    assert other.new_place() == 2
    # None of it reached the one it was copied from.
    assert (ground.place_number, ground.places, ground.steps) == (1, 2, 3)
    assert ground.reach == (16, 0, 0, 0)
    ground.go_to(0)
    assert ground.reach == (32, 0, 0, 0)
    # What was kept on the one can be put back on its copy.
    kept = ground.keep()
    other.put_back(kept)
    assert other.place_number == 0 and other.place == (0, 0)


def test_nonsense_is_refused() -> None:
    ground = started()
    ground.new_place()
    for bad in (2, -1, True, 1.0, "1", None):
        with pytest.raises(ConfigurationError, match="no place"):
            ground.go_to(bad)  # type: ignore[arg-type]
    assert ground.place_number == 1
    for bad in (-1, True, 1.5, "0"):
        with pytest.raises(ConfigurationError):
            ground.put_back(Kept((0, 0), None, (0, 0), place=bad))  # type: ignore[arg-type]
