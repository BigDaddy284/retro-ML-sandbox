"""T-1038: the watch tells which place a go is in, and new ground is new there."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from retro_ai_lab.core import Frame
from retro_ai_lab.training.trial import PAST
from retro_ai_lab.training.vision import BACK_LOOKS, MOST_PLACES, RESPAWN_LOOKS, ScreenWatch
from tests.unit.training.test_vision import HERO, SKY, flat, hero_with_colours, indoors, level

HELD = [(80, 192, HERO)]  # the camera holds it here; its box is 10 by 15, so a step is 15
BLACK, WHITE = (0, 0, 0), (252, 252, 252)
CARD = (40, 40, 160)  # a card between two levels: one plain colour, like nothing else here
NEW_PLACE = "it shows nowhere this go has been"


def plain(colour: tuple[int, int, int]) -> Frame:
    a = np.zeros((240, 256, 3), np.uint8)
    a[:] = colour
    return Frame(256, 240, 3, a.tobytes())


def room(n: int) -> Frame:
    """One of many rooms, no two alike: bars of its own colour at its own height."""
    a = np.zeros((240, 256, 3), np.uint8)
    a[:] = (20 + 13 * n, 240 - 11 * n, 40 + 9 * n)  # its own walls: a cut from any other
    top = 10 + 12 * n
    a[top : top + 8, 8 + 3 * n : 248] = (250 - 10 * n, 12 * n, 200 - 7 * n)
    a[top + 9, ::5] = WHITE
    return Frame(256, 240, 3, a.tobytes())


def hall(seed: int, camera: int) -> Frame:
    """A long hall that looks different all along: a view further off than the camera's
    reach is not the same view. Each ``seed`` is another hall."""
    rng = np.random.default_rng(seed)
    a = np.zeros((240, 2048, 3), np.uint8)
    for x0 in range(0, 2048, 48):
        a[:200, x0 : x0 + 48] = rng.integers(0, 200, 3)
        a[int(rng.integers(20, 160)) :, x0 + 8 : x0 + 20] = rng.integers(0, 255, 3)[None]
    a[200:] = (90, 90, 90)
    a = a[:, camera : camera + 256].copy()
    x, y, spr = HELD[0]
    h, w = spr.shape[:2]
    solid = (spr != np.array(SKY, np.uint8)).any(axis=2)
    a[y : y + h, x : x + w][solid] = spr[solid]
    return Frame(256, 240, 3, a.tobytes())


def in_hall(seed: int, *cameras: int) -> list[Frame]:
    return [hall(seed, camera) for camera in cameras]


def in_the_hall(to: int = 160) -> ScreenWatch:
    """A go that has walked along the first hall, 16 pixels a look, to camera ``to``."""
    watch = ScreenWatch(hero_with_colours((86, 200)))
    watch.begin_go()
    paid = see(watch, *in_hall(1, *range(0, to + 16, 16)))
    assert sum(paid) == to // 15 and watch.ground.place == (to, 0)
    return watch


def outside() -> ScreenWatch:
    """A go that has walked 64 pixels along the level: four steps paid, at camera 164."""
    watch = ScreenWatch(hero_with_colours((86, 200)))
    watch.begin_go()
    for i in range(5):
        watch.look(level(100 + 16 * i, HELD))
    assert watch.take_new_ground() == 4 and watch.ground.places == 1
    return watch


def see(watch: ScreenWatch, *frames: Frame, lost: int = 0) -> list[int]:
    """Look at each screen; what each look paid."""
    paid = []
    for frame in frames:
        watch.look(frame, lost=lost)
        paid.append(watch.take_new_ground())
    return paid


def inside(*cameras: int) -> list[Frame]:
    return [indoors(camera, HELD) for camera in cameras]


def along(*cameras: int) -> list[Frame]:
    return [level(camera, HELD) for camera in cameras]


def test_in_and_out_of_a_door_pays_each_side_once() -> None:
    watch = outside()
    # In through a door. Nothing is paid at the cut, and it is nowhere yet.
    assert see(watch, *inside(400)) == [0]
    assert watch.ground.place_number == 0 and watch.ground.place == (0, 0)
    # The next look can be measured: a new place, and each step from its door pays.
    assert see(watch, *inside(416, 432, 448, 464)) == [1, 1, 1, 1]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    assert watch.ground.reach == (64, 0, 0, 0)
    # Out again, to the view it left by: the first place, where 64 pixels were reached.
    assert see(watch, *along(164, 164)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (0, 2)
    assert watch.ground.reach == (64, 0, 0, 0)
    assert see(watch, *along(180, 196, 212, 228, 244)) == [0, 0, 0, 0, 1]
    # In again, by the same door: the room, where 64 pixels were reached.
    assert see(watch, *inside(400, 400)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    assert see(watch, *inside(416, 432, 448, 464, 480)) == [0, 0, 0, 0, 1]
    # However often it goes in and out, standing or walking a little, nothing more.
    for _ in range(4):
        assert see(watch, *along(244, 244, 260, 244)) == [0, 0, 0, 0]
        assert watch.ground.place_number == 0
        assert see(watch, *inside(400, 400, 416, 400)) == [0, 0, 0, 0]
        assert watch.ground.place_number == 1
    assert watch.ground.places == 2 and watch.ground.steps == 4 + 4 + 1 + 1


@pytest.mark.parametrize(("after", "same"), [(1, True), (RESPAWN_LOOKS, True), (61, False)])
def test_a_cut_after_a_life_lost_is_the_same_place(after: int, same: bool) -> None:
    """The game takes a life and then puts the player somewhere else, over as many
    screens as it likes: that is no new place, and its ground pays nothing again."""
    assert RESPAWN_LOOKS == 60
    watch = outside()
    assert see(watch, *along(164), lost=1) == [0]
    # It settles at the look after the cut: ``after`` looks from the life lost.
    assert see(watch, *along(*[164] * (after - 2))) == [0] * (after - 2)
    assert see(watch, *inside(400, 416)) == ([0, 0] if same else [0, 1])
    assert watch.ground.place_number == (0 if same else 1)
    if same:
        # At the door of the place it was in: nothing until past its furthest there.
        assert watch.ground.places == 1 and watch.ground.reach == (64, 0, 0, 0)
        assert see(watch, *inside(432, 448, 464, 480)) == [0, 0, 0, 1]
    else:
        assert watch.ground.places == 2
        assert see(watch, *inside(432, 448)) == [1, 1]


def test_a_cut_right_after_a_life_lost() -> None:
    # The life and the cut in the same look.
    watch = outside()
    assert see(watch, *inside(400), lost=1) == [0]
    assert see(watch, *inside(416, 432, 448, 464, 480)) == [0, 0, 0, 0, 1]
    assert watch.ground.places == 1


@pytest.mark.parametrize("back", [164, 180, 150])
def test_a_flash_is_no_new_place(back: int) -> None:
    """The whole screen goes white and comes back to the view it left, or one the
    camera moved over a little: where it was. Counted from the door, as a cut always
    was: nothing is paid twice."""
    watch = outside()
    assert see(watch, plain(WHITE), *along(back, back)) == [0, 0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (0, 1)
    assert watch.ground.place == (0, 0) and watch.ground.reach == (64, 0, 0, 0)
    assert see(watch, *along(back + 16, back + 32, back + 48, back + 64, back + 80)) == [
        0,
        0,
        0,
        0,
        1,
    ]


def test_a_fade_is_one_arrival() -> None:
    # Three screens with nothing in common, and then the room: one arrival.
    watch = outside()
    assert see(watch, plain(BLACK), plain(WHITE), *inside(400)) == [0, 0, 0]
    assert watch.ground.places == 1
    assert see(watch, *inside(400, 416, 432)) == [0, 1, 1]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)


def test_a_card_between_two_levels_is_a_place_of_its_own() -> None:
    """A card that stands for a few looks can be measured against itself: it is a
    place, where nothing moves and nothing is paid. The level after it is another."""
    watch = outside()
    assert see(watch, plain(CARD), plain(CARD), plain(CARD)) == [0, 0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    assert see(watch, *inside(400, 400, 416, 432)) == [0, 0, 1, 1]
    assert (watch.ground.place_number, watch.ground.places) == (2, 3)
    # The same card again is the same place; what comes after it is new once more.
    assert see(watch, plain(CARD), plain(CARD)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (1, 3)
    assert see(watch, flat(0, HELD), flat(0, HELD), flat(16, HELD)) == [0, 0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (3, 4)


@pytest.mark.parametrize(("later", "undone"), [(1, True), (BACK_LOOKS, True), (5, False)])
def test_a_life_lost_just_after_undoes_the_place(later: int, undone: bool) -> None:
    """A game can show the player put back before it counts the life: the place it
    was taken to be in a few looks before the life went was that."""
    assert BACK_LOOKS == 4
    watch = outside()
    assert see(watch, *inside(400, 400)) == [0, 0]
    assert watch.ground.place_number == 1
    assert see(watch, *inside(*[400] * (later - 1))) == [0] * (later - 1)
    assert see(watch, *inside(400), lost=1) == [0]
    assert watch.ground.place_number == (0 if undone else 1)
    walk = see(watch, *inside(416, 432, 448, 464, 480))
    assert walk == ([0, 0, 0, 0, 1] if undone else [1, 1, 1, 1, 1])
    # A life lost long after, or one that undid a place, undoes nothing more.
    assert see(watch, *inside(480), lost=1) == [0]
    assert watch.ground.place_number == (0 if undone else 1)


def test_no_more_places_than_the_most() -> None:
    assert MOST_PLACES == 16
    watch = outside()
    for n in range(1, MOST_PLACES):
        assert see(watch, room(n), room(n)) == [0, 0]
        assert (watch.ground.place_number, watch.ground.places) == (n, n + 1)
    # The seventeenth is not taken for new: it stays at the door of the place it was
    # in, as every cut did before, and pays only past what was reached there.
    assert see(watch, *inside(400, 400, 416)) == [0, 0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (MOST_PLACES - 1, MOST_PLACES)
    assert see(watch, flat(0, HELD), flat(0, HELD), flat(16, HELD), flat(32, HELD)) == [0, 0, 0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (MOST_PLACES - 1, MOST_PLACES)
    # The places it has are still known again.
    assert see(watch, room(3), room(3)) == [0, 0]
    assert watch.ground.place_number == 3
    assert see(watch, *along(164, 164)) == [0, 0]
    assert watch.ground.place_number == 0


def test_gone_back_to_before_the_door() -> None:
    watch = ScreenWatch(hero_with_colours((86, 200)))
    watch.begin_go()
    see(watch, *along(100, 116, 132))
    assert watch.take_new_ground() == 0 and watch.ground.steps == 2
    mark = watch.mark()
    assert see(watch, *along(148, 164)) == [1, 1]
    assert see(watch, *inside(400, 416, 432)) == [0, 1, 1]
    assert watch.ground.place_number == 1
    inside_mark = watch.mark()
    assert see(watch, *inside(448)) == [1]
    # The game is put back to before the door: the first place, where it was then.
    watch.back_to(mark)
    assert (watch.ground.place_number, watch.ground.places) == (0, 2)
    assert watch.ground.place == (32, 0)
    assert see(watch, *along(148, 164, 180)) == [0, 0, 1]
    assert watch.take_ground_again() == 2  # a go played straight would be paid them
    # Through the door again: the room it knows, and its ground pays nothing twice.
    assert see(watch, *inside(400, 416, 432, 448, 464)) == [0, 0, 0, 0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    # Put back to a moment inside the room: in the room, where it was then.
    watch.back_to(inside_mark)
    assert watch.ground.place_number == 1 and watch.ground.place == (32, 0)
    assert see(watch, *inside(448, 464, 480)) == [0, 0, 1]
    # A jump that was not settled when the game was put back is forgotten with it.
    assert see(watch, *along(180)) == [0]
    watch.back_to(inside_mark)
    assert see(watch, *inside(448, 464)) == [0, 0]
    assert watch.ground.place_number == 1 and watch.ground.places == 2


def test_a_try_settles_a_door_as_the_go_would(caplog: pytest.LogCaptureFixture) -> None:
    watch = outside()
    mark = watch.mark()
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.vision"):
        other = watch.trial(mark)
        # The try goes through a door the go has not been through: new ground, enough
        # of it for the try to count as past (T-1001).
        assert see(other, *along(164), *inside(400, 416, 432)) == [0, 0, 1, 1]
        assert (other.ground.place_number, other.ground.places) == (1, 2)
        assert other.ground.steps - watch.ground.steps == 2 >= PAST
    assert not [r for r in caplog.records if NEW_PLACE in r.getMessage()]  # a try says nothing
    # Nothing of it reached the go's own watch.
    assert (watch.ground.place_number, watch.ground.places) == (0, 1)
    assert watch.take_new_ground() == 0
    # The go comes to the door itself: new for it too. A try from then on knows the room.
    assert see(watch, *inside(400, 416, 432)) == [0, 1, 1]
    inside_mark = watch.mark()
    # (A try's first look has no look before it to be measured against.)
    again = watch.trial(inside_mark)
    assert see(again, *inside(448, 464, 480)) == [0, 1, 1]
    assert again.ground.place_number == 1
    out = watch.trial(inside_mark)
    assert see(out, *inside(432), *along(164, 164, 180)) == [0, 0, 0, 0]
    assert out.ground.place_number == 0 and watch.ground.place_number == 1


def test_a_new_place_is_said_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.vision"):
        watch = outside()
        see(watch, *inside(400, 416), *along(164, 164), *inside(400, 400))
        see(watch, flat(0, HELD), flat(0, HELD))
    said = [r.getMessage() for r in caplog.records if NEW_PLACE in r.getMessage()]
    assert said == [
        "The screen changed altogether with no life lost, and it shows nowhere this go has "
        "been: the AI takes it for a new place (number 2 of this go). Ground there is new "
        "from where it came in.",
        "The screen changed altogether with no life lost, and it shows nowhere this go has "
        "been: the AI takes it for a new place (number 3 of this go). Ground there is new "
        "from where it came in.",
    ]


def test_a_new_go_knows_no_places() -> None:
    watch = outside()
    assert see(watch, *inside(400, 416)) == [0, 1]
    watch.begin_go()
    assert see(watch, *along(100, 116)) == [0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (0, 1)
    # The room is new again in the new go.
    assert see(watch, *inside(400, 416)) == [0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    # And where a go began is known again when the game puts it back there.
    assert see(watch, *along(100, 100, 116, 132)) == [0, 0, 0, 1]
    assert watch.ground.place_number == 0


def test_a_moment_kept_in_another_go() -> None:
    """A moment can outlive its go. Put back to one kept in a place this go has not
    come to, the ground makes that place with nothing reached, the watch has no
    screen for it, and the next places are counted on from there."""
    watch = outside()
    assert see(watch, *inside(400, 416)) == [0, 1]
    kept = watch.mark()
    watch.begin_go()
    assert see(watch, *along(100, 116)) == [0, 1]
    watch.back_to(kept)
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    # Nothing was reached there this go: the 32 pixels it stands from the door are two steps.
    assert see(watch, *inside(432, 448)) == [2, 1]
    # Somewhere else again: a third place, and the second is known by where it was left.
    assert see(watch, flat(0, HELD), flat(0, HELD), flat(16, HELD)) == [0, 0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (2, 3)
    assert see(watch, *inside(448, 448, 464)) == [0, 0, 0]
    assert watch.ground.place_number == 1
    assert see(watch, *along(100, 100)) == [0, 0]
    assert watch.ground.place_number == 0


def small(colour: tuple[int, int, int]) -> Frame:
    a = np.zeros((120, 128, 3), np.uint8)
    a[:] = colour
    a[100:, ::3] = WHITE
    return Frame(128, 120, 3, a.tobytes())


def test_a_screen_of_another_size_is_never_a_place_it_knows() -> None:
    """A game can change the size of its picture. The tracker starts afresh there, and
    a cut after that is compared with the doors it has: none of that size."""
    watch = outside()
    assert see(watch, small(CARD), small(CARD)) == [0, 0]
    assert watch.ground.places == 1  # no look to measure against: no cut seen
    assert see(watch, small(BLACK), small(BLACK)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    # Back at the size it began with, the level is known again.
    assert see(watch, *along(164, 164)) == [0, 0]
    assert see(watch, *inside(400, 400)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (2, 3)
    assert see(watch, *along(164, 164)) == [0, 0]
    assert watch.ground.place_number == 0


def test_a_place_is_known_by_the_view_it_was_left_by() -> None:
    """Coming out of a room it is where it went in, which looks nothing like where the
    go began. The screen it left by says which place that is."""
    watch = in_the_hall(320)
    # Out by a fade, into a room.
    assert see(watch, plain(BLACK), plain(WHITE), *inside(400, 400)) == [0, 0, 0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    # Back in the hall where it left it: the hall, by the view it was left by. Counted
    # from the door, so nothing there pays until past the 320 pixels it reached.
    assert see(watch, *in_hall(1, 320, 320)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (0, 2)
    assert watch.ground.reach == (320, 0, 0, 0)
    # Walking on, where it is is not asked again: each look is like nowhere it has a
    # screen of, and no new place comes of that.
    assert see(watch, *in_hall(1, *range(336, 496, 16))) == [0] * 10
    assert (watch.ground.place_number, watch.ground.places) == (0, 2)
    # A part of the hall it has never seen, reached by a cut, is somewhere new.
    assert see(watch, *in_hall(1, 1000, 1000, 1016)) == [0, 0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (2, 3)


def test_the_place_it_is_in_is_asked_first() -> None:
    watch = in_the_hall(160)
    assert see(watch, *inside(400, 400)) == [0, 0]
    # Into the hall again 60 pixels from its start: like neither its door nor where it
    # was left, so a place of its own.
    assert see(watch, *in_hall(1, 60, 60)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (2, 3)
    # A flash, and a view that is within the camera's reach of this place's door and of
    # the first place's too: it is where it was.
    assert see(watch, plain(WHITE), *in_hall(1, 30, 30)) == [0, 0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (2, 3)


def test_what_was_undone_is_forgotten() -> None:
    # A cut that was not settled when the game was put back.
    watch = in_the_hall(320)
    mark = watch.mark()
    see(watch, *in_hall(1, *range(336, 496, 16)), *inside(400))
    watch.back_to(mark)
    assert see(watch, *in_hall(1, 336, 352)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (0, 1)
    # A life lost and undone: a door soon after is a door, not the game putting it back.
    see(watch, *in_hall(1, 368), lost=1)
    watch.back_to(mark)
    assert see(watch, *in_hall(1, 336), *inside(400, 400, 416)) == [0, 0, 0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    # A place taken up and undone: a hit soon after undoes no place.
    watch.back_to(mark)
    assert watch.ground.place_number == 0
    see(watch, *in_hall(1, 336))
    see(watch, *in_hall(1, 352), lost=1)
    assert watch.ground.place_number == 0 and watch.ground.place == (352, 0)
    # The first look after going back can be a cut: the place is left by the screen
    # gone back to, and known again by it.
    watch.back_to(mark)
    assert see(watch, *in_hall(2, 0, 0)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (2, 3)
    assert see(watch, *in_hall(1, 320, 320)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (0, 3)


def test_a_new_go_forgets_what_was_under_way() -> None:
    # A cut not settled when the go ended.
    watch = outside()
    assert see(watch, *inside(400)) == [0]
    watch.begin_go()
    assert see(watch, *along(100, 116)) == [0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (0, 1)
    # A life lost at the end of the go before.
    watch = outside()
    see(watch, *along(164), lost=1)
    watch.begin_go()
    assert see(watch, *along(100, 116), *inside(400, 416)) == [0, 1, 0, 1]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    # A place taken up at the end of the go before.
    watch = outside()
    assert see(watch, *inside(400, 400)) == [0, 0]
    watch.begin_go()
    assert see(watch, *along(100, 116)) == [0, 1]
    see(watch, *along(132), lost=1)
    assert watch.ground.place_number == 0 and watch.ground.place == (32, 0)


def test_a_try_keeps_what_it_learns_of_places_to_itself() -> None:
    watch = in_the_hall(160)
    # Another hall and back: the second place, left 64 pixels from its door.
    assert see(watch, *in_hall(2, 0, 0, 16, 32, 48, 64)) == [0, 0, 1, 1, 1, 1]
    assert see(watch, *in_hall(1, 160, 160)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (0, 2)
    mark = watch.mark()
    # A try goes into the second hall by its door: the place the go knows, where its
    # first 64 pixels are walked ground. It walks far along and comes back out.
    other = watch.trial(mark)
    assert see(other, *in_hall(1, 160), *in_hall(2, 0, 0)) == [0, 0, 0]
    assert (other.ground.place_number, other.ground.places) == (1, 2)
    paid = see(other, *in_hall(2, *range(16, 336, 16)))
    assert paid[:4] == [0, 0, 0, 0] and sum(paid) == 320 // 15 - 64 // 15
    assert see(other, *in_hall(1, 160, 160)) == [0, 0]
    assert other.ground.place_number == 0
    # The try's watch knows the second hall by where *it* left it.
    assert see(other, *in_hall(2, 320, 320)) == [0, 0]
    assert (other.ground.place_number, other.ground.places) == (1, 2)
    # The go's does not: for the go that view is nowhere it has been. It knows the hall
    # by where it left it itself.
    assert see(watch, *in_hall(2, 64, 64)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (1, 2)
    assert see(watch, *in_hall(1, 160, 160)) == [0, 0]
    assert see(watch, *in_hall(2, 320, 320)) == [0, 0]
    assert (watch.ground.place_number, watch.ground.places) == (2, 3)


def test_an_undone_place_is_undone_once() -> None:
    watch = outside()
    assert see(watch, *inside(400, 400)) == [0, 0]
    assert see(watch, *inside(400), lost=1) == [0]
    assert watch.ground.place_number == 0
    # A second life two looks later, still within reach of the place that was undone:
    # it is where it walked to, not at the door again.
    assert see(watch, *inside(416)) == [0]
    assert see(watch, *inside(432), lost=1) == [0]
    assert watch.ground.place_number == 0 and watch.ground.place == (32, 0)


def test_a_flash_and_then_a_hit() -> None:
    """Staying where it was is taking up no place: a life lost just after a flash, with
    the screen carrying on, leaves it where it walked to."""
    watch = outside()
    assert see(watch, plain(WHITE), *along(164, 164, 180)) == [0, 0, 0, 0]
    assert see(watch, *along(196), lost=1) == [0]
    assert watch.ground.place_number == 0 and watch.ground.place == (32, 0)
