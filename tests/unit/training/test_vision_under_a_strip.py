"""T-1061: the watch on a road under a status bar.

The road is the stand-in of the tracker's own tests (a place with little in it, seen
through a screen of 256 by 240 with a solid bar along one edge), with a car the camera
holds near the bottom of the screen and the watch knows by its colours.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from retro_ai_lab.core import Frame
from retro_ai_lab.observation.signature import ColourSignature
from retro_ai_lab.observation.sprites import SpriteTag
from retro_ai_lab.observation.tracking import ObjectTracker
from retro_ai_lab.training.vision import ScreenWatch
from retro_ai_lab.understanding.knowledge import Confidence
from retro_ai_lab.understanding.player import PlayerFinding
from tests.unit.observation.test_tracking_under_a_strip import NIGHT, RICH, START, THIN, seen

RED = (248, 56, 0)
GREY = (124, 124, 124)
SAID = "A strip of the screen stands still"


def car() -> np.ndarray:
    a = np.zeros((16, 12, 3), np.uint8)
    a[:] = NIGHT
    a[1:, 1:11] = RED
    return a


def rock() -> np.ndarray:
    a = np.zeros((14, 14, 3), np.uint8)
    a[:] = GREY
    a[3:6, 3:9] = (60, 60, 60)
    return a


CAR, ROCK = car(), rock()
HELD = (120, 200)  # where the camera holds the car; its box is 10 by 15, so a step is 15


def found(at: tuple[int, int] = (126, 208)) -> PlayerFinding:
    red = (RED[0] << 16) | (RED[1] << 8) | RED[2]
    sign = ColourSignature(((red, 150),), 10, 15)
    return PlayerFinding(None, at, frozenset({"left", "right"}), 2, Confidence.MEDIUM, sign)


def road(
    camera: tuple[int, int],
    where: str | None,
    *,
    count: int = 0,
    at: tuple[int, int] = HELD,
    world: np.ndarray = THIN,
) -> Frame:
    return seen(world, camera, where=where, count=count, things=[(*at, CAR)])


def drive(
    watch: ScreenWatch,
    step: tuple[int, int],
    where: str | None,
    *,
    looks: int = 10,
    start: int = 0,
    world: np.ndarray = THIN,
) -> list[int]:
    """Looks at the road as the camera moves ``step`` between each and the next; what
    each look was paid for new ground."""
    paid = []
    for i in range(start, start + looks):
        camera = (START[0] + step[0] * i, START[1] + step[1] * i)
        watch.look(road(camera, where, count=i, world=world))
        paid.append(watch.take_new_ground())
    return paid


DRIVES = [
    ("top", (0, -8), (0, 0, 0, 72), 4),
    ("top", (0, -13), (0, 0, 0, 117), 7),
    ("top", (6, -4), (54, 0, 0, 36), 5),
    ("bottom", (0, 16), (0, 0, 144, 0), 9),
    ("left", (8, 0), (72, 0, 0, 0), 4),
    ("left", (6, -4), (54, 0, 0, 36), 5),
    ("right", (-12, 0), (0, 108, 0, 0), 7),
]


@pytest.mark.parametrize(("where", "step", "reach", "steps"), DRIVES)
def test_new_ground_under_a_bar_is_what_the_road_travelled(
    where: str, step: tuple[int, int], reach: tuple[int, int, int, int], steps: int
) -> None:
    watch = ScreenWatch(found())
    watch.begin_go()
    paid = drive(watch, step, where)
    # Nine moves, counted in the car's own size and each step paid once.
    assert watch.ground.reach == reach and watch.ground.unit == 15
    assert sum(paid) == steps == watch.ground.steps and paid[0] == 0
    assert watch.ground.place == (9 * step[0], 9 * step[1]) == watch.tracker.camera
    assert watch.tracker.read_as == (0, 9) and watch.follows_camera
    assert watch.me == (HELD[0] + 1, HELD[1] + 1, 10, 15)
    # As on the same road with no bar over it, look for look.
    bare = ScreenWatch(found())
    bare.begin_go()
    assert drive(bare, step, None) == paid
    assert bare.ground.reach == reach and bare.tracker.read_as == (9, 0)
    # Standing still, and going back, pay nothing.
    for i in (9, 9, 8, 7):
        watch.look(road((START[0] + step[0] * i, START[1] + step[1] * i), where))
        assert watch.take_new_ground() == 0


def test_with_the_reading_held_off_nothing_is_paid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fault this task mends: the bar outvotes the road, the camera is read as
    standing, and a go that drives 72 pixels is paid for none of them."""
    monkeypatch.setattr(ObjectTracker, "_under_a_strip", lambda *_: None)
    watch = ScreenWatch(found())
    watch.begin_go()
    assert sum(drive(watch, (0, -8), "top")) == 0
    assert watch.ground.reach == (0, 0, 0, 0) and watch.tracker.camera == (0, 0)
    assert watch.tracker.read_as == (0, 0)
    # Nothing tells it that it misses anything: so little changes that no look is a jolt.
    assert watch.follows_camera


def test_the_car_steering_across_the_road_changes_nothing() -> None:
    watch = ScreenWatch(found())
    watch.begin_go()
    paid = []
    for i in range(10):
        at = (HELD[0] + 5 * i, HELD[1])
        watch.look(road((START[0], START[1] - 8 * i), "top", count=i, at=at))
        paid.append(watch.take_new_ground())
        assert watch.me == (at[0] + 1, at[1] + 1, 10, 15)
    # On a screen that scrolls the ground is the camera's travel: 72 pixels up the road
    # are four steps of 15, and steering across it is not somewhere new.
    assert watch.tracker.camera == (0, -72) and watch.ground.reach == (0, 0, 0, 72)
    assert sum(paid) == 4 and watch.tracker.read_as == (0, 9)


def test_the_log_says_once_that_a_strip_stands_still(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="retro_ai_lab.training.vision")
    watch = ScreenWatch(found())
    watch.begin_go()
    drive(watch, (0, -8), "top", looks=1)
    assert SAID not in caplog.text  # nothing has moved yet
    drive(watch, (0, -8), "top", looks=5, start=1)
    mark = watch.mark()
    # A try from here reads the screen as the go does, and has nothing to say of it.
    trial = watch.trial(mark)
    assert trial.tracker.read_as == watch.tracker.read_as == (0, 5)
    drive(trial, (0, -8), "top", looks=5, start=5)
    assert trial.tracker.read_as == (0, 9) and watch.tracker.read_as == (0, 5)
    # Nor has the next go.
    watch.begin_go()
    drive(watch, (0, -8), "top", looks=6)
    assert watch.tracker.read_as == (0, 10)
    said = [r for r in caplog.records if SAID in r.getMessage()]
    assert len(said) == 1 and said[0].levelno == logging.INFO
    assert "status bar" in said[0].getMessage()


@pytest.mark.parametrize(("where", "world"), [(None, THIN), ("top", RICH), (None, RICH)])
def test_the_log_says_nothing_where_the_whole_screen_tells(
    where: str | None, world: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="retro_ai_lab.training.vision")
    watch = ScreenWatch(found())
    watch.begin_go()
    paid = drive(watch, (0, -8), where, world=world)
    assert sum(paid) == 4 and watch.tracker.read_as == (9, 0)
    assert SAID not in caplog.text


def test_a_try_is_paid_for_what_it_gets_past_the_go() -> None:
    watch = ScreenWatch(found())
    watch.begin_go()
    drive(watch, (0, -8), "top", looks=4)
    mark = watch.mark()
    drive(watch, (0, -8), "top", looks=4, start=4)
    assert watch.ground.reach == (0, 0, 0, 56)
    before = (watch.looks, watch.ground.place, watch.ground.steps, watch.tracker.read_as)
    # From the marked moment the try drives the same road, and 40 pixels more of it.
    far = watch.trial(mark)
    paid = drive(far, (0, -8), "top", looks=9, start=3)
    assert far.ground.place == (0, -88) and far.ground.reach == (0, 0, 0, 88)
    assert sum(paid) == 88 // 15 - 56 // 15 == 2
    # A try that gets no further than the go did is paid nothing.
    short = watch.trial(mark)
    assert sum(drive(short, (0, -8), "top", looks=4, start=3)) == 0
    after = (watch.looks, watch.ground.place, watch.ground.steps, watch.tracker.read_as)
    assert after == before


def crash(watch: ScreenWatch, where: str | None) -> None:
    """A go: the road slides 8 rows a look and a rock that stands on it comes down the
    screen to the car; the look after it touches, a life is gone."""
    watch.begin_go()
    for i in range(12):
        camera = (START[0], START[1] - 8 * i)
        rock_at = (HELD[0] - 1, HELD[1] - 14 - 80 + 8 * i)  # standing still on the road
        things = [(*HELD, CAR)]
        if i <= 10:
            things.append((rock_at[0], min(rock_at[1], HELD[1] - 14), ROCK))
        frame = seen(THIN, camera, where=where, count=i, things=things)
        watch.look(frame, lost=1 if i == 11 else 0)


@pytest.mark.parametrize("where", ["top", None])
def test_what_stands_on_the_road_and_costs_a_life_twice_is_found_to_hurt(
    where: str | None,
) -> None:
    watch = ScreenWatch(found())
    crash(watch, where)
    crash(watch, where)
    hurts = [r for r in watch.roles() if r.tag in (SpriteTag.ENEMY, SpriteTag.HAZARD)]
    assert len(hurts) == 1 and hurts[0].confidence is Confidence.MEDIUM
    assert watch.lives_lost == 2 and watch.lives_after_touch == 2
