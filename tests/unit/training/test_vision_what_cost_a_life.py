"""T-1051: the screen watch says of every life lost what it was put down to, looks
back 45 looks, and blames the closest thing where no touch was seen."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from retro_ai_lab.observation.sprites import SpriteTag
from retro_ai_lab.training.vision import SOON_LOOKS, ScreenWatch
from retro_ai_lab.understanding.roles import Way
from tests.unit.training.test_vision import (
    COIN,
    HERO,
    WALKER,
    fall,
    hero_with_colours,
    into_a_walker,
    screen,
)

VISION = "retro_ai_lab.training.vision"
Things = list[tuple[int, int, np.ndarray]]


def a_watch() -> ScreenWatch:
    watch = ScreenWatch(hero_with_colours((46, 108)))
    watch.begin_go()
    return watch


def touch(watch: ScreenWatch, *, then_wait: int) -> None:
    """The hero walks into a walker and stays against it; ``then_wait`` looks after
    the look they were first seen touching at, a life is taken."""
    watch.look(screen([(40, 100, HERO), (120, 100, WALKER)]))
    watch.look(screen([(109, 100, HERO), (120, 100, WALKER)]))
    for _ in range(then_wait - 1):
        watch.look(screen([(109, 100, HERO), (120, 100, WALKER)]))
    watch.look(screen([(109, 100, HERO), (120, 100, WALKER)]), lost=1)


def comes_close(watch: ScreenWatch, *, then_wait: int = 4, lost: bool = True) -> None:
    """A walker comes at the hero, 16 pixels a look, and is last seen 21 pixels off;
    the game then shows the hero alone, and ``then_wait`` looks later takes a life."""
    for x in (120, 104, 88, 72):
        watch.look(screen([(40, 100, HERO), (x, 100, WALKER)]))
    for _ in range(then_wait - 1):
        watch.look(screen([(40, 100, HERO)]))
    if lost:
        watch.look(screen([(40, 100, HERO)]), lost=1)


def said(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("A life was lost")]


NOTHING = (
    f"A life was lost with nothing seen coming to touch the AI or close to it in the "
    f"{SOON_LOOKS} looks before: nothing is blamed."
)


# ------------------------------------------------------------ the look back


def test_the_look_back_is_45_looks() -> None:
    assert SOON_LOOKS == 45
    watch = a_watch()
    touch(watch, then_wait=40)  # 5.3 s at a look every 8 frames: 30 looks was 4 s
    assert (watch.lives_lost, watch.lives_after_touch) == (1, 1)
    (role,) = watch.roles()
    assert role.tag is SpriteTag.HAZARD and role.ways == (Way.SIDE,)
    late = a_watch()
    touch(late, then_wait=50)
    assert (late.lives_lost, late.lives_after_touch, late.lives_after_closest) == (1, 0, 0)
    assert [r.tag for r in late.roles()] == [SpriteTag.OTHER]
    # The same for what was close, and for a watch told how long to wait.
    close = a_watch()
    comes_close(close, then_wait=40)
    assert close.lives_after_closest == 1
    gone = a_watch()
    comes_close(gone, then_wait=50)
    assert gone.lives_after_closest == 0 and gone.roles() == ()
    quick = ScreenWatch(hero_with_colours((46, 108)), soon=10)
    quick.begin_go()
    comes_close(quick, then_wait=12)
    assert quick.lives_after_closest == 0


# ------------------------------------------------------------ the closest thing


def test_a_life_with_no_touch_seen_is_put_down_to_the_closest_thing() -> None:
    watch = a_watch()
    comes_close(watch)
    assert (watch.lives_lost, watch.lives_after_touch, watch.lives_after_closest) == (1, 0, 1)
    (role,) = watch.roles()
    assert role.tag is SpriteTag.ENEMY and role.ways == (Way.SIDE,)  # it hurts, and it moves
    assert role.describe().endswith(
        "A life went after 1 of 1 from the side (1 of them with no touch seen: it was "
        "the closest thing)."
    )
    # A fall explains a life without it: what was close is let go.
    watch = a_watch()
    comes_close(watch, lost=False)
    fall(watch)
    watch.look(screen([(40, 100, HERO)]), lost=1)
    assert (watch.lives_after_fall, watch.lives_after_closest) == (1, 0) and watch.roles() == ()
    # A life told with no look to go with it (the go ended) goes to the closest too.
    watch = a_watch()
    comes_close(watch, lost=False)
    watch.told(lost=1)
    assert (watch.lives_lost, watch.lives_after_closest) == (1, 1)
    watch.told(lost=1)  # the next, with nothing new seen, is nobody's
    assert (watch.lives_lost, watch.lives_after_closest) == (2, 1)
    watch.told()  # and none lost is nothing told
    assert watch.lives_lost == 2


def test_a_walker_never_seen_touching_is_blamed_as_the_closest() -> None:
    """The go of T-1016's test seen the old way, in which the walker standing on the
    ground beside the hero is not seen at all: nothing was put down to it. It was seen
    coming, and is blamed for that now."""
    from retro_ai_lab.observation.tracking import ObjectTracker
    from retro_ai_lab.training.vision import LOOK_SHIFT

    old = ScreenWatch(hero_with_colours((86, 200)))
    old._tracker = ObjectTracker(max_shift=LOOK_SHIFT, whole=False)
    into_a_walker(old)
    assert (old.lives_lost, old.lives_after_touch, old.lives_after_closest) == (1, 0, 1)
    (role,) = old.roles()
    assert role.tag is SpriteTag.ENEMY and role.ways == (Way.SIDE,)
    # Seen as it is now, the touch itself is seen, and that is what is counted.
    watch = ScreenWatch(hero_with_colours((86, 200)))
    into_a_walker(watch)
    assert (watch.lives_after_touch, watch.lives_after_closest) == (1, 0)


# ------------------------------------------------------------ what the log says


def test_the_log_says_what_each_life_was_put_down_to(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=VISION):
        touch(a_watch(), then_wait=3)
        comes_close(a_watch())
        falling = a_watch()
        falling.look(screen([(40, 100, HERO)]))
        fall(falling)
        falling.look(screen([(40, 100, HERO)]), lost=1)
        alone = a_watch()
        alone.look(screen([(40, 100, HERO)]))
        alone.look(screen([(40, 100, HERO)]), lost=2)  # two at once: one line
    assert said(caplog) == [
        "A life was lost. Put down to thing-2, touched from the side 3 looks before.",
        f"A life was lost with no touch seen in the {SOON_LOOKS} looks before. Put down to "
        "thing-2, the closest thing in that time: 22 pixels off, met from the side, 4 looks "
        "before.",
        "A life was lost falling off the bottom of the screen: nothing it touched is blamed.",
        NOTHING,
    ]


def test_the_log_names_what_else_was_touched_and_counts_one_look(
    caplog: pytest.LogCaptureFixture,
) -> None:
    watch = a_watch()
    watch.look(screen([(40, 100, HERO), (120, 100, WALKER), (70, 60, COIN)]))
    watch.look(screen([(64, 68, HERO), (120, 100, WALKER), (70, 60, COIN)]))  # under the coin
    watch.look(screen([(109, 100, HERO), (120, 100, WALKER), (70, 60, COIN)]))  # at the walker
    with caplog.at_level(logging.INFO, logger=VISION):
        watch.look(screen([(109, 100, HERO), (120, 100, WALKER), (70, 60, COIN)]), lost=1)
    (line,) = said(caplog)
    assert line == (
        "A life was lost. Put down to thing-3, touched from the side 1 look before. Also "
        "touched in that time: thing-2 from below (2 looks before)."
    )


def test_a_life_told_with_no_look_is_said_too(caplog: pytest.LogCaptureFixture) -> None:
    watch = a_watch()
    watch.look(screen([(40, 100, HERO), (120, 100, WALKER)]))
    watch.look(screen([(109, 100, HERO), (120, 100, WALKER)]))
    with caplog.at_level(logging.INFO, logger=VISION):
        watch.told(lost=1)
        watch.told(lost=1)
        watch.told()
    assert said(caplog) == [
        "A life was lost. Put down to thing-2, touched from the side 0 looks before.",
        NOTHING,
    ]


def test_a_trys_watch_says_nothing(caplog: pytest.LogCaptureFixture) -> None:
    watch = a_watch()
    watch.look(screen([(40, 100, HERO), (120, 100, WALKER)]))
    trial = watch.trial(watch.mark())
    with caplog.at_level(logging.INFO, logger=VISION):
        trial.look(screen([(40, 100, HERO), (120, 100, WALKER)]))
        trial.look(screen([(109, 100, HERO), (120, 100, WALKER)]), lost=1)
        trial.told(lost=1)
    assert said(caplog) == [] and trial.lives_lost == 2


# ------------------------------------------------------------ the summary


def test_the_summary_counts_the_lives_put_down_to_the_closest_thing() -> None:
    watch = a_watch()
    comes_close(watch)
    assert (
        "1 lives lost, 0 of them soon after touching something, 1 with no touch seen and "
        "put down to the closest thing; 1 kinds found to hurt" in watch.describe()
    )
    # With a fall as well, each is named.
    watch.begin_go()
    watch.look(screen([(40, 100, HERO)]))
    fall(watch)
    watch.look(screen([(40, 100, HERO)]), lost=1)
    assert (
        "2 lives lost, 0 of them soon after touching something, 1 with no touch seen and "
        "put down to the closest thing and 1 after falling off the bottom of the screen; "
        "1 kinds found to hurt" in watch.describe()
    )
    # With none, the summary is as it was.
    plain = a_watch()
    touch(plain, then_wait=3)
    assert "1 lives lost, 1 of them soon after touching something; 1 kinds" in plain.describe()
