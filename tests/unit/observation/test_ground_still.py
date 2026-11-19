"""T-1058: the ground knows when the screen has stood still for a good while."""

from __future__ import annotations

from retro_ai_lab.observation.ground import STILL_LOOKS, Ground

ME = (80, 200)
NO_MOVE = (0, 0)


def looked(ground: Ground, times: int, moved: tuple[int, int] = NO_MOVE) -> Ground:
    for _ in range(times):
        ground.look(moved, ME, 16)
    return ground


def test_a_screen_is_not_still_until_it_has_been_looked_at_long_enough() -> None:
    assert STILL_LOOKS == 150  # 20 seconds of play at a look every 8 frames
    ground = Ground()
    ground.begin()
    assert not ground.still  # nothing seen yet says nothing
    looked(ground, STILL_LOOKS - 1)
    assert not ground.still and not ground.scrolls
    looked(ground, 1)
    assert ground.still and not ground.scrolls
    looked(ground, 500)
    assert ground.still


def test_the_player_moving_about_the_screen_is_not_the_screen_moving() -> None:
    ground = Ground()
    ground.begin()
    for n in range(STILL_LOOKS):
        ground.look(NO_MOVE, (20 + n, 200 - n % 7), 16)
    assert ground.still and ground.steps > 0  # it was paid for where it newly stood


def test_one_move_of_the_camera_ends_it_whichever_way() -> None:
    for moved in ((1, 0), (-1, 0), (0, 1), (0, -3), (8, 8)):
        ground = looked(Ground(), STILL_LOOKS)
        assert ground.still
        ground.look(moved, ME, 16)
        assert not ground.still and ground.scrolls
        # ... and it takes as long again to be still once more: a room that stands
        # still in a game that scrolls.
        looked(ground, STILL_LOOKS - 1)
        assert not ground.still
        looked(ground, 1)
        assert ground.still and ground.scrolls


def test_a_camera_that_keeps_moving_is_never_still() -> None:
    ground = Ground()
    for _ in range(10):
        looked(ground, STILL_LOOKS - 1)
        ground.look((2, 0), ME, 16)
    assert not ground.still


def test_a_look_that_does_not_see_the_player_counts_too() -> None:
    ground = Ground()
    for _ in range(STILL_LOOKS):
        ground.look(NO_MOVE)
    assert ground.still


def test_it_is_kept_from_go_to_go_and_when_the_screen_jumps() -> None:
    ground = looked(Ground(), 100)
    ground.begin()  # a new go
    looked(ground, 49)
    assert not ground.still
    ground.back()  # the screen jumped: it is only known to be somewhere else
    ground.new_place()
    looked(ground, 1)
    assert ground.still
    ground.begin()
    assert ground.still  # the next go knows it from its first step


def test_putting_the_story_back_does_not_put_it_back() -> None:
    ground = looked(Ground(), 100)
    kept = ground.keep()
    looked(ground, 50)
    assert ground.still
    ground.put_back(kept)  # the game was put back to a saved moment
    assert ground.still


def test_a_copy_for_a_try_knows_it_and_changes_nothing() -> None:
    ground = looked(Ground(), STILL_LOOKS - 1)
    other = ground.copy()
    assert not other.still
    looked(other, 1)
    assert other.still and not ground.still
    other.look((4, 0), ME, 16)
    looked(ground, 1)
    assert ground.still and not ground.scrolls and not other.still
    assert ground.copy().still
