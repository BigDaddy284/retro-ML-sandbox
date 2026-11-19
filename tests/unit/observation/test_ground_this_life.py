"""T-1062: the ground count keeps a third count, for the life the go is on.

A search after a life lost takes a try for the way past when it stays alive and gets
somewhere the go had not been (T-1001). After the game has put the player back at a
place's door, everything within reach is ground the go has been on, so no try there
could ever be the way. The count now keeps how far the life it is on has reached as
well, starts that over when it is told the game put the player back, and says at
every look how many steps the life had not covered. What is paid is the go's new
ground, as it was.
"""

from __future__ import annotations

import dataclasses

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.ground import MOST_A_LOOK, Ground

ME = (80, 200)  # where the camera holds the player
STILL = (0, 0)
STEP = (16, 0)  # a step to the right, the player's own size
WAYS = [(16, 0), (-16, 0), (0, 16), (0, -16)]


def started() -> Ground:
    """A go under way: seen once, standing where the go began."""
    ground = Ground()
    ground.begin()
    assert ground.look(STILL, ME, 16) == 0 and ground.fresh == 0
    return ground


def looks(ground: Ground, move: tuple[int, int], count: int) -> list[tuple[int, int]]:
    """(steps paid, steps new to the life it is on) for each of ``count`` looks."""
    return [(ground.look(move, ME, 16), ground.fresh) for _ in range(count)]


@pytest.mark.parametrize("way", WAYS)
def test_until_the_game_puts_it_back_the_two_counts_are_one(way: tuple[int, int]) -> None:
    ground = started()
    there = looks(ground, way, 5)
    back = looks(ground, (-way[0], -way[1]), 8)  # back over it, and three steps beyond
    again = looks(ground, way, 9)
    assert there == [(1, 1)] * 5
    assert back == [(0, 0)] * 5 + [(1, 1)] * 3
    assert again == [(0, 0)] * 8 + [(1, 1)]
    assert ground.steps == 9
    # Faster than a step a look, and the most a look counts.
    assert looks(ground, (way[0] * 3, way[1] * 3), 1) == [(3, 3)]
    assert looks(ground, (way[0] * 9, way[1] * 9), 1) == [(MOST_A_LOOK, MOST_A_LOOK)]
    # Past the most, the rest is not owed: not to the go and not to the life.
    assert looks(ground, way, 1) == [(1, 1)]


@pytest.mark.parametrize("way", WAYS)
def test_put_back_by_the_game_walked_ground_is_new_to_the_life(way: tuple[int, int]) -> None:
    ground = started()
    assert looks(ground, way, 6) == [(1, 1)] * 6
    # The life goes and the game puts the player back at the door.
    ground.back()
    ground.new_life()
    assert ground.place == (0, 0) and ground.steps == 6 and ground.fresh == 1
    assert ground.look(STILL, ME, 16) == 0 and ground.fresh == 0
    # The same six steps: none of them paid, each of them new to this life.
    assert looks(ground, way, 6) == [(0, 1)] * 6
    # Past the furthest the go has been: paid as ever, and new to the life as well.
    assert looks(ground, way, 3) == [(1, 1)] * 3
    assert ground.steps == 9
    # Standing, and back over what this life has walked: nothing of either.
    assert looks(ground, STILL, 2) == [(0, 0)] * 2
    assert looks(ground, (-way[0], -way[1]), 4) == [(0, 0)] * 4
    assert looks(ground, way, 4) == [(0, 0)] * 4


def test_the_most_a_look_counts_holds_for_the_life_too() -> None:
    ground = started()
    assert looks(ground, STEP, 10) == [(1, 1)] * 10
    ground.back()
    ground.new_life()
    assert looks(ground, (16 * 7, 0), 1) == [(0, MOST_A_LOOK)]
    assert looks(ground, STEP, 1) == [(0, 1)]  # the three past the most are not owed
    other = Ground(most=2)
    other.begin()
    other.look(STILL, ME, 16)
    assert other.look((48, 0), ME, 16) == 2 and other.fresh == 2
    other.back()
    other.new_life()
    assert other.look((48, 0), ME, 16) == 0 and other.fresh == 2


def test_a_life_that_goes_twice() -> None:
    ground = started()
    assert looks(ground, STEP, 8) == [(1, 1)] * 8
    ground.back()
    ground.new_life()
    assert looks(ground, STEP, 3) == [(0, 1)] * 3
    ground.back()
    ground.new_life()
    # The third life has not been where the second was, nor where the first was.
    assert looks(ground, STEP, 8) == [(0, 1)] * 8
    assert looks(ground, STEP, 2) == [(1, 1)] * 2
    assert ground.steps == 10


def test_nothing_else_starts_the_lifes_count_over() -> None:
    ground = started()
    assert looks(ground, STEP, 5) == [(1, 1)] * 5
    kept = ground.keep()
    assert looks(ground, STEP, 3) == [(1, 1)] * 3
    # Put back to a saved moment: what the life reached stays reached, like the go's.
    ground.put_back(kept)
    assert ground.place == (80, 0)
    assert looks(ground, STEP, 3) == [(0, 0)] * 3
    assert looks(ground, STEP, 1) == [(1, 1)]
    # At the door again with no word of a life lost (a cut): the same.
    ground.back()
    assert looks(ground, STEP, 9) == [(0, 0)] * 9
    assert looks(ground, STEP, 1) == [(1, 1)]


def test_a_copy_has_the_lifes_count_and_keeps_it_to_itself() -> None:
    ground = started()
    assert looks(ground, STEP, 6) == [(1, 1)] * 6
    ground.back()
    ground.new_life()
    assert looks(ground, STEP, 2) == [(0, 1)] * 2
    kept = ground.keep()
    other = ground.copy()
    assert other.fresh == ground.fresh == 1
    other.put_back(kept)
    # The copy walks on: two steps this life has been on are behind it, four are not.
    assert looks(other, (-16, 0), 2) == [(0, 0)] * 2
    assert looks(other, STEP, 2) == [(0, 0)] * 2
    assert looks(other, STEP, 4) == [(0, 1)] * 4
    assert looks(other, STEP, 1) == [(1, 1)]
    # Nothing of that is in the count it was copied from.
    assert ground.fresh == 1 and ground.place == (32, 0) and ground.steps == 6
    assert looks(ground, STEP, 4) == [(0, 1)] * 4
    # A new life in the copy is the copy's.
    other.new_life()
    assert looks(ground, (-16, 0), 1) == [(0, 0)]
    assert looks(ground, STEP, 1) == [(0, 0)]


def test_the_life_starts_over_in_every_place() -> None:
    ground = started()
    assert looks(ground, STEP, 4) == [(1, 1)] * 4  # the first place
    assert ground.new_place() == 1
    ground.look(STILL, ME, 16)
    assert looks(ground, STEP, 6) == [(1, 1)] * 6  # the second: new from its door
    # The life goes in the second place and the game puts it back at that place's door.
    ground.back()
    ground.new_life()
    assert looks(ground, STEP, 6) == [(0, 1)] * 6
    assert looks(ground, STEP, 1) == [(1, 1)]
    # Back through the door into the first place: this life has not been there either.
    ground.go_to(0)
    assert ground.look(STILL, ME, 16) == 0 and ground.fresh == 0
    assert looks(ground, STEP, 4) == [(0, 1)] * 4
    assert looks(ground, STEP, 1) == [(1, 1)]
    assert ground.steps == 12
    # A place it comes to afterwards is new to the go and to the life alike.
    assert ground.new_place() == 2
    ground.look(STILL, ME, 16)
    assert looks(ground, STEP, 2) == [(1, 1)] * 2


def test_a_new_go_has_a_new_life() -> None:
    ground = started()
    assert looks(ground, STEP, 4) == [(1, 1)] * 4
    ground.back()
    ground.new_life()
    assert looks(ground, STEP, 2) == [(0, 1)] * 2
    assert ground.keep().life == 1
    ground.begin()
    assert ground.fresh == 0 and ground.keep().life == 0
    assert ground.look(STILL, ME, 16) == 0 and ground.fresh == 0
    assert looks(ground, STEP, 5) == [(1, 1)] * 5
    # A moment of the new go is a moment of its first life: put back to, it leaves
    # what that life has reached.
    kept = ground.keep()
    assert looks(ground, STEP, 2) == [(1, 1)] * 2
    ground.put_back(kept)
    assert looks(ground, STEP, 2) == [(0, 0)] * 2


def test_on_a_screen_that_never_scrolls() -> None:
    """Where it stands is the place, counted from where it was first seen: a life that
    starts over is first seen where the game put it."""
    ground = Ground()
    ground.begin()
    at = [(40 + 16 * i, 200) for i in range(7)]
    paid = [(ground.look(STILL, me, 16), ground.fresh) for me in at]
    assert paid == [(0, 0)] + [(1, 1)] * 6 and not ground.scrolls
    ground.back()
    ground.new_life()
    paid = [(ground.look(STILL, me, 16), ground.fresh) for me in at]
    assert paid == [(0, 0)] + [(0, 1)] * 6
    assert (ground.look(STILL, (40 + 16 * 7, 200), 16), ground.fresh) == (1, 1)


def test_before_the_player_is_seen_nothing_is_new_to_the_life() -> None:
    ground = Ground()
    ground.begin()
    for _ in range(3):
        assert ground.look(STEP) == 0 and ground.fresh == 0  # no step to count in yet
    ground.back()
    ground.new_life()
    assert ground.look(STEP) == 0 and ground.fresh == 0
    # What was reached is counted when there is a step: three by the go, and two by
    # the life, from where it started over.
    assert ground.look(STEP, ME, 16) == 3 and ground.fresh == 2


def test_a_moment_of_the_life_before_puts_that_life_back_on() -> None:
    """The game is put back to a moment saved before the life went: that life is on
    again, and what it had reached when it went is what it has reached. (A search
    for a way past plays its tries from such a moment after the game has put the
    player back: counted by the new life, every try that stayed alive would be on
    ground new to it.)"""
    ground = started()
    assert looks(ground, STEP, 4) == [(1, 1)] * 4
    kept = ground.keep()
    assert kept.life == 0
    assert looks(ground, STEP, 2) == [(1, 1)] * 2
    ground.back()
    ground.new_life()
    assert ground.keep().life == 1
    assert looks(ground, STEP, 2) == [(0, 1)] * 2
    ground.put_back(kept)
    assert ground.place == (64, 0) and ground.keep().life == 0
    # The two steps the life before had walked past the moment, and then on.
    assert looks(ground, STEP, 2) == [(0, 0)] * 2
    assert looks(ground, STEP, 2) == [(1, 1)] * 2
    # It goes again, and the life after it starts from nothing as the last one did.
    ground.back()
    ground.new_life()
    assert looks(ground, STEP, 8) == [(0, 1)] * 8
    assert looks(ground, STEP, 1) == [(1, 1)]


def test_a_copy_put_back_to_the_life_before_leaves_the_new_life_on() -> None:
    ground = started()
    assert looks(ground, STEP, 5) == [(1, 1)] * 5
    kept = ground.keep()
    ground.back()
    ground.new_life()
    assert looks(ground, STEP, 1) == [(0, 1)]
    # A try from the moment before the life went: where that life had been is
    # nothing new, the step past it is.
    for _ in range(2):
        other = ground.copy()
        other.put_back(kept)
        assert looks(other, (-16, 0), 3) == [(0, 0)] * 3
        assert looks(other, STEP, 3) == [(0, 0)] * 3
        assert looks(other, STEP, 1) == [(1, 1)]
    # The count it was copied from is still on the new life.
    assert ground.keep().life == 1
    assert looks(ground, STEP, 4) == [(0, 1)] * 4
    assert looks(ground, STEP, 1) == [(1, 1)]


def test_a_moment_of_the_life_it_is_on_changes_nothing_of_the_lifes_count() -> None:
    ground = started()
    assert looks(ground, STEP, 6) == [(1, 1)] * 6
    ground.back()
    ground.new_life()
    assert looks(ground, STEP, 2) == [(0, 1)] * 2
    kept = ground.keep()
    assert looks(ground, STEP, 2) == [(0, 1)] * 2
    ground.put_back(kept)
    assert ground.keep().life == 1
    assert looks(ground, STEP, 2) == [(0, 0)] * 2
    assert looks(ground, STEP, 2) == [(0, 1)] * 2
    assert looks(ground, STEP, 1) == [(1, 1)]


def test_a_moment_kept_in_a_later_life_of_another_go_changes_nothing() -> None:
    other = started()
    looks(other, STEP, 3)
    other.back()
    other.new_life()
    looks(other, STEP, 1)
    later = other.keep()
    assert later.life == 1
    ground = started()
    assert looks(ground, STEP, 4) == [(1, 1)] * 4
    ground.put_back(later)
    assert ground.keep().life == 0 and ground.place == (16, 0)
    assert looks(ground, STEP, 3) == [(0, 0)] * 3
    assert looks(ground, STEP, 1) == [(1, 1)]


def test_only_the_life_before_is_kept() -> None:
    """A moment two lives back is given what the life before had reached: nothing
    older is kept. (The go keeps no moment across two lives lost.)"""
    ground = started()
    assert looks(ground, STEP, 8) == [(1, 1)] * 8
    oldest = ground.keep()
    ground.back()
    ground.new_life()
    assert looks(ground, STEP, 3) == [(0, 1)] * 3
    ground.back()
    ground.new_life()
    assert looks(ground, STEP, 1) == [(0, 1)]
    ground.put_back(oldest)
    assert ground.keep().life == 0 and ground.place == (128, 0)
    ground.back()
    assert looks(ground, STEP, 3) == [(0, 0)] * 3  # the second life's three steps
    assert looks(ground, STEP, 5) == [(0, 1)] * 5
    assert looks(ground, STEP, 1) == [(1, 1)]


def test_the_life_before_is_kept_place_by_place() -> None:
    ground = started()
    assert looks(ground, STEP, 2) == [(1, 1)] * 2
    assert ground.new_place() == 1
    ground.look(STILL, ME, 16)
    assert looks(ground, STEP, 5) == [(1, 1)] * 5
    kept = ground.keep()
    ground.back()
    ground.new_life()
    assert looks(ground, STEP, 1) == [(0, 1)]
    ground.put_back(kept)
    assert ground.place_number == 1 and ground.place == (80, 0)
    assert looks(ground, STEP, 1) == [(1, 1)]
    # In the first place too the life before had been two steps along.
    ground.go_to(0)
    ground.look(STILL, ME, 16)
    assert looks(ground, STEP, 2) == [(0, 0)] * 2
    assert looks(ground, STEP, 1) == [(1, 1)]


@pytest.mark.parametrize("life", [-1, True, 1.0, "1", None])
def test_a_kept_moment_names_its_life_by_a_whole_number(life: object) -> None:
    ground = started()
    looks(ground, STEP, 2)
    kept = dataclasses.replace(ground.keep(), life=life)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="A kept life is numbered 0 or more"):
        ground.put_back(kept)
    # Refused before anything was changed.
    assert ground.place == (32, 0) and ground.keep().life == 0
    assert looks(ground, STEP, 1) == [(1, 1)]
