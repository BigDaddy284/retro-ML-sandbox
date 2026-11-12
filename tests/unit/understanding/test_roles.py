"""T-975: what each kind of thing is, from what touching it does to the player."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.sprites import SpriteTag
from retro_ai_lab.observation.tracking import Track
from retro_ai_lab.understanding.knowledge import Confidence
from retro_ai_lab.understanding.roles import Happening, RoleLearner

ME = 1


def track(tid: int, kind: str | None, x: int, y: int = 100, *, moving: bool = False) -> Track:
    t = Track(tid, kind, x, y, 16, 16, 0, 0)
    t.history = [(0, x, y), (1, x + (5 if moving else 0), y)]
    return t


def me(x: int) -> Track:
    return track(ME, "thing-1", x)


def touch_and(
    learner: RoleLearner,
    other: Track,
    then: Happening | None,
    *,
    start: int = 0,
    vanish: bool = False,
    wait: int = 5,
) -> int:
    """Walk the player into ``other`` and report ``then`` a few frames later."""
    f = start
    learner.observe(f, [me(other.x - 40), other], ME)  # apart
    f += 1
    learner.observe(f, [me(other.x - 10), other], ME)  # touching
    for _ in range(wait):
        f += 1
        shown = [me(other.x - 10)] if vanish else [me(other.x - 10), other]
        learner.observe(f, shown, ME)
    f += 1
    shown = [me(other.x - 40)] if vanish else [me(other.x - 40), other]
    learner.observe(f, shown, ME, [then] if then else [])
    return f + 1


def role_of(learner: RoleLearner, kind: str) -> tuple[SpriteTag, Confidence]:
    r = next(r for r in learner.roles() if r.kind == kind)
    return r.tag, r.confidence


def test_a_moving_thing_that_costs_a_life_is_an_enemy() -> None:
    lr = RoleLearner()
    f = 0
    for i in range(2):
        f = touch_and(lr, track(10 + i, "thing-2", 200, moving=True), Happening.LIFE_LOST, start=f)
    assert role_of(lr, "thing-2") == (SpriteTag.ENEMY, Confidence.MEDIUM)
    assert "an enemy" in lr.roles()[0].describe()


def test_a_still_thing_that_costs_a_life_is_a_hazard() -> None:
    lr = RoleLearner()
    f = 0
    for i in range(4):
        f = touch_and(lr, track(10 + i, "thing-3", 200), Happening.LIFE_LOST, start=f)
    assert role_of(lr, "thing-3") == (SpriteTag.HAZARD, Confidence.HIGH)


def test_a_thing_that_vanishes_as_the_score_goes_up_is_an_item() -> None:
    lr = RoleLearner()
    f = 0
    for i in range(2):
        f = touch_and(lr, track(10 + i, "coin", 200), Happening.SCORE_UP, start=f, vanish=True)
    assert role_of(lr, "coin") == (SpriteTag.ITEM, Confidence.MEDIUM)


def test_score_without_it_vanishing_does_not_make_an_item() -> None:
    lr = RoleLearner()
    block = track(10, "block", 200)  # the same block, touched twice; it stays
    f = touch_and(lr, block, Happening.SCORE_UP)
    touch_and(lr, block, Happening.SCORE_UP, start=f)
    assert role_of(lr, "block")[0] is SpriteTag.OTHER


def test_touching_with_nothing_after_is_harmless() -> None:
    lr = RoleLearner()
    f = 0
    for i in range(3):
        f = touch_and(lr, track(10 + i, "cloud", 200), None, start=f)
    assert role_of(lr, "cloud") == (SpriteTag.OTHER, Confidence.MEDIUM)


def test_one_unlucky_touch_among_many_harmless_ones_is_not_a_hazard() -> None:
    lr = RoleLearner()
    f = touch_and(lr, track(10, "cloud", 200), Happening.LIFE_LOST)
    for i in range(3):
        f = touch_and(lr, track(11 + i, "cloud", 200), None, start=f)
    assert role_of(lr, "cloud")[0] is SpriteTag.OTHER


def test_a_life_lost_long_after_the_touch_is_not_put_down_to_it() -> None:
    lr = RoleLearner(soon=10)
    touch_and(lr, track(10, "thing-2", 200, moving=True), Happening.LIFE_LOST, wait=20)
    assert role_of(lr, "thing-2")[0] is SpriteTag.OTHER


def test_a_long_touch_counts_once() -> None:
    lr = RoleLearner()
    other = track(10, "thing-2", 200)
    for f in range(10):
        lr.observe(f, [me(190), other], ME)
    assert lr.evidence["thing-2"].touches == 1


def test_one_life_is_put_down_to_one_touch() -> None:
    lr = RoleLearner()
    a, b = track(10, "a", 200), track(11, "b", 200, y=110)
    lr.observe(0, [me(150), a, b], ME)
    lr.observe(1, [me(190), a, b], ME)  # touching both
    lr.observe(2, [me(150), a, b], ME, [Happening.LIFE_LOST])
    hurt = {k: e.hurt for k, e in lr.evidence.items()}
    assert sum(hurt.values()) == 1


def test_untouched_and_kindless_things_get_no_role() -> None:
    lr = RoleLearner()
    lr.observe(0, [me(10), track(10, "far", 200), track(11, None, 20)], ME)
    assert lr.roles() == ()


def test_no_player_means_no_touches() -> None:
    lr = RoleLearner()
    lr.observe(0, [me(190), track(10, "x", 200)], None)
    lr.observe(1, [me(190), track(10, "x", 200)], 99)
    assert lr.roles() == ()


def test_settings_are_checked() -> None:
    with pytest.raises(ConfigurationError):
        RoleLearner(soon=0)


# ------------------------------------------------------------------ T-987: a box for "me"


def test_touches_are_measured_from_the_box_when_one_is_given() -> None:
    """T-987: the player known by its looks is a box, not a track. Touches are measured
    from the box, and the tracks lying in it are the player and not a thing it touched."""
    lr = RoleLearner()
    spike = track(7, "thing-3", 200)
    f = 0
    for _ in range(2):
        # Its own track (a moving patch of its colours) sits in the box all along.
        lr.observe(f, [track(3, "thing-1", 150, moving=True), spike], None, me=(150, 100, 16, 16))
        lr.observe(
            f + 1, [track(3, "thing-1", 186, moving=True), spike], None, me=(186, 100, 16, 16)
        )
        lr.observe(f + 2, [track(3, "thing-1", 186), spike], None, [Happening.LIFE_LOST], me=None)
        f += 40
    assert [r.kind for r in lr.roles()] == ["thing-3"]  # its own kind was never "touched"
    assert role_of(lr, "thing-3") == (SpriteTag.HAZARD, Confidence.MEDIUM)
    assert "thing-1" not in lr.evidence  # nor counted as a thing that moves by itself
    # No box and no track: nothing is touched, whatever is on the screen.
    quiet = RoleLearner()
    quiet.observe(0, [track(3, "thing-1", 186), spike], None)
    quiet.observe(1, [track(3, "thing-1", 186), spike], None, me=None)
    assert quiet.roles() == () and not quiet.waiting
    # A track only partly in the box (its middle outside) is something else.
    near = RoleLearner()
    near.observe(0, [track(4, "thing-5", 162)], None, me=(150, 100, 16, 16))
    assert near.evidence["thing-5"].touches == 1 and near.waiting
    # A box given beside a track id wins: the id's track is not looked up.
    both = RoleLearner()
    both.observe(0, [me(0), spike], ME, me=(186, 100, 16, 16))
    assert both.evidence["thing-3"].touches == 1
