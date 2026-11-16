"""T-1051: what cost a life, by how the thing was met and by who was closest.

Boss, 16 Nov: "it should go oh whats that tag it, oh it killed me when it touch me,
best keep the fuck away from them in the future. And then if the same enemy kills
them in a different way, oh they can do that too, best remember that." In his logs no
touch at all was seen before more than half of the lives lost, and the kinds that
were blamed were outvoted by harmless touches.
"""

from __future__ import annotations

import pytest

from retro_ai_lab.observation.signature import Box
from retro_ai_lab.observation.sprites import SpriteTag
from retro_ai_lab.observation.tracking import Track
from retro_ai_lab.understanding.knowledge import Confidence
from retro_ai_lab.understanding.roles import (
    REACH,
    Blame,
    Happening,
    KindEvidence,
    Role,
    RoleLearner,
    Way,
    WayEvidence,
    way_of,
)

LOST = [Happening.LIFE_LOST]
SCORED = [Happening.SCORE_UP]
GROUND = 100  # the top of a 16 by 16 thing standing on the ground


def thing(
    tid: int,
    kind: str | None,
    x: int,
    y: int = GROUND,
    *,
    w: int = 16,
    h: int = 16,
    moving: bool = False,
    missing: int = 0,
) -> Track:
    t = Track(tid, kind, x, y, w, h, 0, 0)
    t.history = [(0, x, y), (1, x + (5 if moving else 0), y)]
    t.missing = missing
    return t


def me(x: int, y: int = GROUND, *, w: int = 16, h: int = 16) -> Box:
    return (x, y, w, h)


class Meeting:
    """Looks handed to a learner one after another."""

    def __init__(self, learner: RoleLearner | None = None) -> None:
        self.learner = learner or RoleLearner()
        self.look = 0

    def see(self, box: Box | None, *things: Track, then: list[Happening] | None = None) -> None:
        self.learner.observe(self.look, list(things), None, then or [], me=box)
        self.look += 1

    def wait(self, looks: int, box: Box | None = None) -> None:
        for _ in range(looks):
            self.see(box)

    def of(self, kind: str) -> KindEvidence:
        return self.learner.evidence[kind]

    def role(self, kind: str) -> Role:
        return next(r for r in self.learner.roles() if r.kind == kind)


# ------------------------------------------------------------ the way of a touch


@pytest.mark.parametrize(
    ("mine", "its", "way"),
    [
        # The same size, the thing's top at 100.
        (me(184), me(200), Way.SIDE),
        (me(190, 84), me(200), Way.ABOVE),  # standing on it
        (me(190, 92), me(200), Way.ABOVE),  # half-way into its top: still come down on
        (me(190, 93), me(200), Way.SIDE),
        (me(190, 116), me(200), Way.BELOW),  # its head against the thing's underside
        (me(190, 108), me(200), Way.BELOW),
        (me(190, 107), me(200), Way.SIDE),
        # A player twice as tall: beside it on the same ground, and coming down on it.
        (me(184, 84, h=32), me(200), Way.SIDE),
        (me(190, 76, h=32), me(200), Way.ABOVE),
        (me(190, 77, h=32), me(200), Way.SIDE),
        # A thing three times as tall: beside it wherever along it, on it, under it.
        (me(184, 132), me(200, 100, h=48), Way.SIDE),
        (me(184, 100), me(200, 100, h=48), Way.SIDE),
        (me(190, 92), me(200, 100, h=48), Way.ABOVE),
        (me(190, 140), me(200, 100, h=48), Way.BELOW),
        (me(190, 139), me(200, 100, h=48), Way.SIDE),
    ],
)
def test_the_way_of_a_touch_is_told_from_the_boxes(mine: Box, its: Box, way: Way) -> None:
    assert way_of(mine, its) is way
    # A touch made so is kept so.
    m = Meeting()
    x, y, w, h = its
    m.see(mine, thing(7, "thing-2", x, y, w=w, h=h))
    assert m.of("thing-2").ways == {way: WayEvidence(touches=1)}


def test_the_way_is_that_of_the_look_the_touch_began_at() -> None:
    m = Meeting()
    walker = thing(7, "thing-2", 200, moving=True)
    m.see(me(184), walker)  # met from the side
    m.see(me(190, 84), walker)  # and then on top of it, still touching: one touch
    m.see(me(190, 84), walker, then=LOST)
    assert m.of("thing-2").ways == {Way.SIDE: WayEvidence(touches=1, hurt=1)}
    assert m.learner.last_blame == Blame("thing-2", Way.SIDE, 2, 0, True)


# ------------------------------------------------------------ what hurts


def come_down_on(m: Meeting, tid: int, kind: str = "thing-2") -> None:
    """It lands on the thing, which vanishes as the score goes up."""
    walker = thing(tid, kind, 200, moving=True)
    m.see(me(120), walker)
    m.see(me(194, 84), walker)
    m.see(me(194, 100), then=SCORED)
    m.wait(m.learner.soon + 1, me(120))


def walk_into(m: Meeting, tid: int, kind: str = "thing-2", *, costs: bool = True) -> None:
    walker = thing(tid, kind, 200, moving=True)
    m.see(me(120), walker)
    m.see(me(184), walker)
    m.see(me(184), walker, then=LOST if costs else None)
    m.wait(m.learner.soon + 1, me(120))


def test_an_enemy_come_down_on_often_still_hurts_from_the_side() -> None:
    m = Meeting()
    walk_into(m, 10)
    for i in range(6):
        come_down_on(m, 20 + i)
    ev = m.of("thing-2")
    assert (ev.touches, ev.hurt, ev.collected) == (7, 1, 6)
    assert ev.ways == {
        Way.SIDE: WayEvidence(touches=1, hurt=1),
        Way.ABOVE: WayEvidence(touches=6, collected=6),
    }
    # One life among seven touches: the old rule called it something to pick up.
    role = m.role("thing-2")
    assert (role.tag, role.confidence, role.touches) == (SpriteTag.ENEMY, Confidence.LOW, 7)
    assert role.ways == (Way.SIDE,)
    # A second life from the side and the guess is firmer, however often it is landed on.
    walk_into(m, 11)
    come_down_on(m, 30)
    role = m.role("thing-2")
    assert (role.tag, role.confidence, role.ways) == (
        SpriteTag.ENEMY,
        Confidence.MEDIUM,
        (Way.SIDE,),
    )


def test_one_unlucky_touch_among_harmless_ones_the_same_way_is_no_hazard() -> None:
    m = Meeting()
    walk_into(m, 10, "cloud")
    for i in range(2):
        walk_into(m, 11 + i, "cloud", costs=False)
    # One of three from the side: not half. (One of two would be.)
    role = m.role("cloud")
    assert (role.tag, role.ways) == (SpriteTag.OTHER, ())
    assert m.of("cloud").ways == {Way.SIDE: WayEvidence(touches=3, hurt=1)}
    other = Meeting()
    walk_into(other, 10, "cloud")
    walk_into(other, 11, "cloud", costs=False)
    assert other.role("cloud").tag is SpriteTag.ENEMY and other.role("cloud").ways == (Way.SIDE,)


def test_what_hurt_by_the_old_rule_still_hurts() -> None:
    # Half of all its touches, with nothing kept of the way they were made.
    learner = RoleLearner()
    learner.evidence["thing-3"] = KindEvidence(touches=4, hurt=2)
    (role,) = learner.roles()
    assert (role.tag, role.confidence, role.ways) == (SpriteTag.HAZARD, Confidence.MEDIUM, ())
    # By the touches together its lives are counted together, as they were.
    mixed = KindEvidence(touches=5, hurt=3)
    mixed.ways = {
        Way.SIDE: WayEvidence(touches=2, hurt=2),
        Way.ABOVE: WayEvidence(touches=3, hurt=1),
    }
    learner.evidence["thing-3"] = mixed
    (role,) = learner.roles()
    assert (role.confidence, role.ways) == (Confidence.MEDIUM, (Way.SIDE,))  # three: not four
    # In one way only: the lives of that way.
    mixed.touches, mixed.ways[Way.ABOVE] = 12, WayEvidence(touches=10, hurt=1)
    (role,) = learner.roles()
    assert (role.tag, role.confidence, role.ways) == (
        SpriteTag.HAZARD,
        Confidence.MEDIUM,
        (Way.SIDE,),
    )
    # And two ways that both hurt are counted together.
    mixed.hurt, mixed.ways[Way.ABOVE] = 7, WayEvidence(touches=10, hurt=5)
    (role,) = learner.roles()
    assert (role.confidence, role.ways) == (Confidence.HIGH, (Way.ABOVE, Way.SIDE))


# ------------------------------------------------------------ the closest thing


def test_with_no_touch_seen_the_closest_thing_is_blamed() -> None:
    m = Meeting()
    walker = thing(7, "thing-2", 140, moving=True)
    m.see(me(100), walker)  # 24 pixels off, coming: the look before the hit
    assert not m.learner.waiting and m.learner.close
    m.wait(5, me(100))  # what the game shows of the hit shows no walker at all
    m.see(me(100), then=LOST)
    ev = m.of("thing-2")
    assert (ev.touches, ev.hurt, ev.unseen) == (1, 1, 1)
    assert ev.ways == {Way.SIDE: WayEvidence(touches=1, hurt=1)}
    assert m.learner.last_blame == Blame("thing-2", Way.SIDE, 6, 24, False)
    role = m.role("thing-2")
    assert (role.tag, role.confidence, role.ways) == (SpriteTag.ENEMY, Confidence.LOW, (Way.SIDE,))
    # It is used up with the life: the next life, with nothing new, is nobody's.
    assert not m.learner.close
    m.see(me(100), then=LOST)
    assert m.of("thing-2").hurt == 1 and m.learner.last_blame is None


def test_the_closest_and_of_equals_the_latest() -> None:
    m = Meeting()
    m.see(me(100), thing(7, "far", 146), thing(8, "near", 126))  # 30 and 10 pixels off
    m.see(me(100), thing(7, "far", 140))  # the far one comes on to 24
    m.see(me(100), then=LOST)
    assert m.learner.last_blame == Blame("near", Way.SIDE, 2, 10, False)
    assert "far" not in m.learner.evidence or m.of("far").hurt == 0
    # As close, later: the later one.
    m.see(me(100), thing(7, "first", 126))
    m.see(me(100), thing(8, "second", 74))  # 10 pixels off on the other side
    m.see(me(100), then=LOST)
    assert m.learner.last_blame == Blame("second", Way.SIDE, 1, 10, False)
    # Above it and under it are ways too.
    m.see(me(100), thing(9, "bat", 104, GROUND - 40))
    m.see(me(100), then=LOST)
    assert m.learner.last_blame == Blame("bat", Way.BELOW, 1, 24, False)
    m.see(me(100, GROUND - 40), thing(9, "spike", 104, GROUND))
    m.see(me(100, GROUND - 40), then=LOST)
    assert m.learner.last_blame == Blame("spike", Way.ABOVE, 1, 24, False)


def test_what_is_not_noted_as_close() -> None:
    m = Meeting()
    # Further off than a look's travel; out of sight; of no kind; the player's own.
    m.see(
        me(100),
        thing(1, "far", 100 + 16 + REACH + 1),
        thing(2, "gone", 130, missing=1),
        thing(3, None, 130),
        thing(4, "mine", 102, moving=True),
    )
    assert not m.learner.close
    m.see(me(100), then=LOST)
    assert m.learner.last_blame is None and m.learner.roles() == ()
    # Just within reach is noted.
    m.see(me(100), thing(1, "far", 100 + 16 + REACH))
    assert m.learner.close
    m.see(me(100), then=LOST)
    assert m.learner.last_blame == Blame("far", Way.SIDE, 1, REACH, False)
    # Longer ago than a touch would wait: forgotten.
    quick = Meeting(RoleLearner(soon=4))
    quick.see(me(100), thing(1, "walker", 130))
    quick.wait(4, me(100))
    assert quick.learner.close
    quick.wait(1, me(100))
    assert not quick.learner.close
    quick.see(me(100), then=LOST)
    assert quick.learner.last_blame is None
    # With the player out of sight nothing is close to it.
    blind = Meeting()
    blind.see(None, thing(1, "walker", 130))
    assert not blind.learner.close


def test_a_touch_waiting_is_preferred_to_a_closer_thing() -> None:
    m = Meeting()
    block = thing(5, "block", 104, GROUND - 17)  # touched from below
    m.see(me(100), block)
    assert m.learner.waiting and not m.learner.close
    m.see(me(60), thing(6, "walker", 86))  # away from the block; a walker 10 pixels off
    m.see(me(60), then=LOST)
    assert m.learner.last_blame == Blame("block", Way.BELOW, 2, 0, True)
    assert "walker" not in m.learner.evidence
    # What was close went with that life too.
    assert not m.learner.close and not m.learner.waiting
    m.see(me(60), then=LOST)
    assert m.learner.last_blame is None
    # What is seen at the look a life is told with belongs to what comes after it.
    m.see(me(60), thing(6, "walker", 86), then=LOST)
    assert m.learner.last_blame is None and m.learner.close
    m.see(me(60), then=LOST)
    assert m.learner.last_blame == Blame("walker", Way.SIDE, 1, 10, False)


def test_the_other_touches_waiting_are_named_with_it() -> None:
    m = Meeting()
    kinds = ["a", "b", "c", "d", "e"]
    for upto in range(1, len(kinds) + 1):  # one more comes to touch it at each look
        m.see(me(100), *(thing(10 + i, kind, 116) for i, kind in enumerate(kinds[:upto])))
    m.see(me(100), then=LOST)
    # The latest takes it; the three before it are named, latest first.
    assert m.learner.last_blame == Blame(
        "e", Way.SIDE, 1, 0, True, (("d", Way.SIDE, 2), ("c", Way.SIDE, 3), ("b", Way.SIDE, 4))
    )
    assert sum(ev.hurt for ev in m.learner.evidence.values()) == 1


def test_a_fall_lets_the_close_things_go() -> None:
    m = Meeting()
    m.see(me(100), thing(7, "walker", 130))
    assert m.learner.close
    m.learner.let_go()
    assert not m.learner.close
    m.learner.life_lost()
    assert m.learner.last_blame is None and m.learner.roles() == ()


def test_a_life_told_directly_goes_to_the_closest_too() -> None:
    m = Meeting()
    m.see(me(100), thing(7, "walker", 130))
    m.wait(3, me(100))
    m.learner.life_lost()  # the go ended: no new look comes with it
    assert m.learner.last_blame == Blame("walker", Way.SIDE, 3, 14, False)
    m.learner.life_lost()  # once only
    assert m.of("walker").hurt == 1 and m.learner.last_blame is None


# ------------------------------------------------------------ what a role says


def test_what_a_role_says_of_its_ways() -> None:
    m = Meeting()
    walk_into(m, 10)
    walk_into(m, 11, costs=False)
    for i in range(3):
        come_down_on(m, 20 + i)
    assert m.role("thing-2").describe() == (
        "thing-2 is an enemy (low, 5 touches). A life went after 0 of 3 from above, "
        "1 of 2 from the side."
    )
    # With a life put down to it as the closest thing.
    m.see(me(100), thing(30, "thing-2", 140, moving=True))
    m.see(me(100), then=LOST)
    assert m.role("thing-2").describe() == (
        "thing-2 is an enemy (medium, 6 touches). A life went after 0 of 3 from above, "
        "2 of 3 from the side (1 of them with no touch seen: it was the closest thing)."
    )
    # A kind no life was ever put down to is said as it was.
    come_down_on(m, 40, "coin")
    assert m.role("coin").describe() == "coin is something to pick up (low, 1 touches)."
    assert m.role("coin").detail == ""
    # One that is outvoted says so too: the log shows what was counted.
    for i in range(4):
        walk_into(m, 50 + i, "post", costs=i == 0)
    assert m.role("post").describe() == (
        "post is harmless (medium, 4 touches). A life went after 1 of 4 from the side."
    )
