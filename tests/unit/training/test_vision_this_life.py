"""T-1062: the watch starts the life's count of ground over when the game puts the
player back, and at nothing else.

The screens are the stand-ins of the watch's own tests: a level, a plain level where
a jump changes little, a dark room, and the practice hopper behind a connection for
the put-back that only the game's own count of its camera tells.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core import Frame
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training.vision import BACK_LOOKS, RESPAWN_LOOKS, ScreenWatch
from tests.unit.training import test_trial as trial
from tests.unit.training import test_vision_put_back_by_count as by_count
from tests.unit.training.test_trial import Hopper
from tests.unit.training.test_vision import HERO, flat, hero_with_colours
from tests.unit.training.test_vision_camera_memory import camera
from tests.unit.training.test_vision_places import (
    HELD,
    WHITE,
    along,
    inside,
    outside,
    plain,
)

hopper = trial.hopper  # the practice game behind a connection (pytest fixture)
no_jolt = by_count.no_jolt  # no change of the screen is a jolt (pytest fixture)

LOGGER = "retro_ai_lab.training.vision"
SAID = "A life was lost and the game put the AI back."
START = [(32, 192, HERO)]  # where the game puts it back: not where the camera holds it
Counted = tuple[int, int]  # what a look paid, and its steps new to the life


def see(watch: ScreenWatch, *frames: Frame, lost: int = 0) -> list[Counted]:
    """Look at each screen: what each look paid, and its steps new to the life."""
    counted = []
    for frame in frames:
        watch.look(frame, lost=lost)
        counted.append((watch.take_new_ground(), watch.take_life_ground()))
    return counted


def walked() -> ScreenWatch:
    """A go that has walked 64 pixels along the level, and been handed its steps."""
    watch = outside()
    assert watch.take_life_ground() == 4  # the steps it was paid: the same four
    return watch


def lives(watch: ScreenWatch) -> list[int]:
    """How often the life's count was started over, told by wrapping the ground's."""
    calls: list[int] = []
    real = watch.ground.new_life

    def counting() -> None:
        calls.append(watch.looks)
        real()

    watch.ground.new_life = counting  # type: ignore[method-assign]
    return calls


def test_until_a_life_is_lost_the_two_counts_are_one() -> None:
    watch = ScreenWatch(hero_with_colours((86, 200)))
    watch.begin_go()
    there = see(watch, *along(*range(100, 260, 16)))
    assert there == [(0, 0)] + [(1, 1)] * 9
    # Back over it pays nothing and is nothing new; on past the furthest is both.
    assert see(watch, *along(*range(228, 84, -16))) == [(0, 0)] * 9
    assert see(watch, *along(*range(116, 276, 16))) == [(0, 0)] * 9 + [(1, 1)]
    # Into a room and out again, with no life lost: each side once, for both. (Out
    # again it is at the level's door, 160 pixels short of the furthest it was there.)
    assert see(watch, *inside(400, 416, 432)) == [(0, 0), (1, 1), (1, 1)]
    assert see(watch, *along(260, 260, 276, 292)) == [(0, 0)] * 4
    assert watch.ground.place_number == 0 and watch.ground.reach == (160, 0, 0, 0)


@pytest.mark.parametrize("after", [1, 2, RESPAWN_LOOKS])
def test_a_cut_after_a_life_lost_starts_the_life_over(after: int) -> None:
    """The game takes a life and then shows the player somewhere else: at the door of
    the place it was in. What it walks from there pays nothing until it is past the
    furthest the go has been, and every step of it is new to the life."""
    watch = walked()
    began = lives(watch)
    if after == 1:
        assert see(watch, *inside(400), lost=1) == [(0, 0)]  # the life and the cut at once
    else:
        assert see(watch, *along(164), lost=1) == [(0, 0)]
        assert see(watch, *along(*[164] * (after - 2))) == [(0, 0)] * (after - 2)
        assert see(watch, *inside(400)) == [(0, 0)]
    assert began == []  # which place it is in is settled at the next look
    assert see(watch, *inside(416, 432, 448, 464)) == [(0, 1)] * 4
    assert len(began) == 1 and watch.ground.place_number == 0
    assert watch.ground.place == (64, 0) == watch.ground.reach[:2]
    assert see(watch, *inside(480, 496)) == [(1, 1), (1, 1)]
    # Back and on again: it has been there in this life.
    assert see(watch, *inside(464, 496)) == [(0, 0), (0, 0)]
    assert len(began) == 1


def test_a_cut_long_after_a_life_lost_is_a_new_place_and_no_new_life() -> None:
    watch = walked()
    began = lives(watch)
    assert see(watch, *along(164), lost=1) == [(0, 0)]
    assert see(watch, *along(*[164] * RESPAWN_LOOKS)) == [(0, 0)] * RESPAWN_LOOKS
    assert see(watch, *inside(400, 416, 432)) == [(0, 0), (1, 1), (1, 1)]
    assert began == [] and watch.ground.place_number == 1
    # Out again, at the level's door: ground this life has walked. The hit it took
    # there left it where it was.
    assert see(watch, *along(164, 164, 180, 196, 212)) == [(0, 0)] * 5
    assert watch.ground.place_number == 0 and began == []
    assert see(watch, *along(228, 244)) == [(0, 0), (1, 1)]


@pytest.mark.parametrize("late", [0, 2, -2])
def test_a_jump_of_the_screen_about_a_life_lost_starts_the_life_over(late: int) -> None:
    """Put back at the start on a level so plain that the jump changes little of the
    screen: seen somewhere else on it, with a life lost in the same look or a couple
    of looks before or after (T-993)."""
    watch = ScreenWatch(hero_with_colours((86, 200)))
    watch.begin_go()
    assert see(watch, *[flat(16 * i, HELD) for i in range(9)]) == [(0, 0)] + [(1, 1)] * 8
    began = lives(watch)
    for i in range(-3, 4):  # three looks either side of the jump at i == 0
        frame = flat(128, HELD) if i < 0 else flat(0, START)
        assert see(watch, frame, lost=1 if i == late else 0) == [(0, 0)]
    assert len(began) == 1 and watch.ground.place == (0, 0)
    # The same ground again: eight steps new to the life and not paid, the ninth both.
    again = see(watch, *[flat(16 * i, HELD) for i in range(1, 10)])
    assert again == [(0, 1)] * 8 + [(1, 1)]


def test_a_hit_that_leaves_it_where_it_was_starts_nothing_over() -> None:
    watch = walked()
    began = lives(watch)
    # A life (or some health) lost and the screen carries on as it was.
    assert see(watch, *along(180), lost=1) == [(1, 1)]
    assert see(watch, *along(196, 180, 164, 180, 196)) == [(1, 1)] + [(0, 0)] * 4
    # A crash: a jolt of the screen that leaves it where it was on the screen.
    crash = ScreenWatch(hero_with_colours((86, 200)))
    crash.begin_go()
    see(crash, *[flat(16 * i, HELD) for i in range(9)], *[flat(128, HELD)] * 3)
    crashed = lives(crash)
    assert see(crash, flat(0, HELD), lost=1) == [(0, 0)]
    assert 0.02 <= crash.tracker.unexplained < 0.25 and crash.ground.place == (128, 0)
    # A jolt that finds it somewhere else, long after the last life was lost.
    late = ScreenWatch(hero_with_colours((86, 200)))
    late.begin_go()
    for i in range(9):
        see(late, flat(16 * i, HELD), lost=1 if i == 2 else 0)
    belated = lives(late)
    see(late, *[flat(128, HELD)] * 3, flat(0, START))
    assert began == crashed == belated == []


@pytest.mark.parametrize(("later", "undone"), [(1, True), (BACK_LOOKS, True), (5, False)])
def test_a_place_undone_by_a_life_lost_just_after_starts_the_life_over(
    later: int, undone: bool
) -> None:
    """A game can show the player put back before it counts the life: the place it
    was taken to be in a few looks before the life went was that (T-1038)."""
    watch = walked()
    began = lives(watch)
    assert see(watch, *inside(400, 400)) == [(0, 0), (0, 0)]
    assert watch.ground.place_number == 1 and began == []
    assert see(watch, *inside(*[400] * (later - 1))) == [(0, 0)] * (later - 1)
    assert see(watch, *inside(400), lost=1) == [(0, 0)]
    assert len(began) == (1 if undone else 0)
    walk = see(watch, *inside(416, 432, 448, 464, 480))
    if undone:
        # At the door of the place it was in: four steps of walked ground, then new.
        assert walk == [(0, 1)] * 4 + [(1, 1)]
    else:
        # A room of its own, and a hit taken in it.
        assert walk == [(1, 1)] * 5
    # A life lost long after, or one that undid a place, undoes nothing more.
    assert see(watch, *inside(480), lost=1) == [(0, 0)]
    assert len(began) == (1 if undone else 0)


def test_a_flash_or_a_door_starts_nothing_over() -> None:
    watch = walked()
    began = lives(watch)
    # A flash leaves it at the door of the place it is in, as every cut does.
    assert see(watch, plain(WHITE), *along(164, 164, 180, 196, 212, 228)) == [(0, 0)] * 7
    assert see(watch, *along(244)) == [(1, 1)]
    # A door, there and back.
    assert see(watch, *inside(400, 416), *along(244, 244)) == [(0, 0), (1, 1), (0, 0), (0, 0)]
    assert see(watch, *along(260, 276, 292, 308, 324)) == [(0, 0)] * 5
    assert see(watch, *along(340)) == [(1, 1)]
    assert began == []


def test_going_back_to_a_saved_moment_starts_nothing_over() -> None:
    watch = walked()
    began = lives(watch)
    mark = watch.mark()
    assert see(watch, *along(180, 196, 212)) == [(1, 1)] * 3
    watch.back_to(mark)
    # What the life reached stays reached, like what the go reached.
    assert see(watch, *along(180, 196, 212)) == [(0, 0)] * 3
    assert see(watch, *along(228)) == [(1, 1)]
    assert began == []


def test_the_steps_are_handed_out_once_and_a_new_go_has_none() -> None:
    watch = walked()
    assert see(watch, *inside(400), lost=1) == [(0, 0)]
    for frame in inside(416, 432, 448):
        watch.look(frame)
    assert watch.take_life_ground() == 3 and watch.take_life_ground() == 0
    assert watch.take_new_ground() == 0
    watch.look(inside(464)[0])
    watch.begin_go()
    assert watch.take_life_ground() == 0
    # The new go's first life has been nowhere, and neither has the go.
    assert see(watch, *along(*range(100, 180, 16))) == [(0, 0)] + [(1, 1)] * 4


def test_a_try_counts_by_the_gos_life_and_keeps_its_own_to_itself() -> None:
    watch = walked()
    assert see(watch, *inside(400), lost=1) == [(0, 0)]
    assert see(watch, *inside(416, 432)) == [(0, 1), (0, 1)]
    mark = watch.mark()
    assert see(watch, *inside(448)) == [(0, 1)]
    before = (watch.looks, watch.ground.place, watch.ground.steps, watch.ground.fresh)
    # From the marked moment (a try's first look has nothing to be measured against):
    # a step this life has been on, one it has not, and one past the go's furthest.
    first = watch.trial(mark)
    assert see(first, *inside(448, 464, 480, 496)) == [(0, 0), (0, 0), (0, 1), (1, 1)]
    # A second try from there is not behind the first: each counts from the go's.
    second = watch.trial(mark)
    assert see(second, *inside(448, 464, 480)) == [(0, 0), (0, 0), (0, 1)]
    # A try in which the game puts it back starts its own life over, and says nothing.
    third = watch.trial(mark)
    began = lives(third)
    assert see(third, *along(164), lost=1) == [(0, 0)]
    assert see(third, *inside(400, 416, 432)) == [(0, 0), (0, 1), (0, 1)]
    assert len(began) == 1
    assert before == (watch.looks, watch.ground.place, watch.ground.steps, watch.ground.fresh)
    assert see(watch, *inside(464)) == [(0, 1)]


def test_the_log_says_once_that_the_game_put_it_back(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    watch = walked()
    assert SAID not in caplog.text
    see(watch, *inside(400), lost=1)
    assert SAID not in caplog.text  # not before it is settled where it is
    see(watch, *inside(416, 432))
    mark = watch.mark()
    # A try from here has nothing to say, whatever comes to it.
    other = watch.trial(mark)
    see(other, *along(164), lost=1)
    see(other, *inside(400, 416))
    # The next time, and in the next go, it is said under the breath.
    see(watch, *along(164), lost=1)
    see(watch, *inside(400, 416))
    watch.begin_go()
    see(watch, *along(100, 116, 132))
    see(watch, *inside(400), lost=1)
    see(watch, *inside(416, 432))
    said = [r for r in caplog.records if SAID in r.getMessage()]
    assert [r.levelno for r in said] == [logging.INFO, logging.DEBUG, logging.DEBUG]
    assert "somewhere this life has not been" in said[0].getMessage()
    # A hit that leaves it where it was says nothing.
    caplog.clear()
    see(watch, *inside(448), lost=1)
    see(watch, *inside(464, 480))
    assert SAID not in caplog.text


def test_a_put_back_told_by_the_games_own_count_starts_the_life_over(
    hopper: Hopper, no_jolt: None
) -> None:
    """Where the screen shows no jump, the game's own count of its camera tells the
    put-back (T-1049): back at what it was at the place's door, a life lost about
    then, and the player somewhere else on the screen."""
    given, unaided = watches = by_count.begun(hopper)
    far = by_count.far_off(hopper, watches)
    for watch in watches:
        watch.take_new_ground()
        watch.take_life_ground()
    began, unseen = lives(given), lives(unaided)
    by_count.put_back(hopper, watches)
    assert len(began) == 1 and unseen == [] and given.ground.place == (0, 0)
    # The same walk again, from the door to where it had got to: the watch that was
    # told is paid nothing, and counts every step of it as new to the life.
    by_count.walk(hopper, given, unaided, by_count.AFAR)
    assert camera(hopper.game) == far and given.ground.place == (far, 0)
    assert given.take_new_ground() == 0
    assert given.take_life_ground() == far // toy_hopper.PLAYER_H
    # Past its furthest the ground is new again, to the go and to the life.
    by_count.look(hopper, watches, ("right",), steps=4)
    assert given.take_new_ground() == given.take_life_ground() == 2
    # The watch without the count saw no put-back: its two counts never parted.
    assert unseen == [] and unaided.take_new_ground() == unaided.take_life_ground()
    assert len(began) == 1


def test_a_try_from_before_the_life_went_counts_by_that_life() -> None:
    """A search for a way past plays its tries from a moment before the life went,
    and by then the watch may have seen the game put the player back. Such a try is
    in the life that went: where that life had been is nothing new to it."""
    watch = walked()
    assert see(watch, *along(180, 196)) == [(1, 1)] * 2
    mark = watch.mark()
    assert see(watch, *along(212, 228)) == [(1, 1)] * 2
    # The life goes and the game puts it back; it walks a little of the level again.
    assert see(watch, *inside(400), lost=1) == [(0, 0)]
    assert see(watch, *inside(416, 432)) == [(0, 1)] * 2
    before = (watch.looks, watch.ground.place, watch.ground.steps, watch.ground.fresh)
    for _ in range(2):
        other = watch.trial(mark)
        # (A try's first look has nothing to be measured against.) Two steps the life
        # that went had walked, then one neither it nor the go had.
        assert see(other, *along(212, 228, 244, 260)) == [(0, 0)] * 3 + [(1, 1)]
    assert before == (watch.looks, watch.ground.place, watch.ground.steps, watch.ground.fresh)
    # The go itself is still on the new life.
    assert see(watch, *inside(448)) == [(0, 1)]


def test_going_back_to_before_the_life_went_puts_that_life_back_on() -> None:
    watch = walked()
    assert see(watch, *along(180, 196)) == [(1, 1)] * 2
    mark = watch.mark()
    assert see(watch, *along(212, 228)) == [(1, 1)] * 2
    assert see(watch, *inside(400), lost=1) == [(0, 0)]
    assert see(watch, *inside(416, 432)) == [(0, 1)] * 2
    began = lives(watch)
    # A way past was found: the game is put back to the moment, and the life is kept.
    watch.back_to(mark)
    assert watch.ground.place == (96, 0) and began == []
    assert see(watch, *along(212, 228)) == [(0, 0)] * 2
    assert see(watch, *along(244, 260)) == [(1, 1)] * 2
    # When that life does go, the next starts from nothing.
    assert see(watch, *inside(400), lost=1) == [(0, 0)]
    assert see(watch, *inside(416, 432)) == [(0, 1)] * 2
    assert len(began) == 1
