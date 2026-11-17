"""T-1052: the screen watch says which things in sight are of a kind that has cost a
life, for the runner to look ahead for."""

from __future__ import annotations

from retro_ai_lab.observation.sprites import SpriteTag
from retro_ai_lab.training.vision import PART_REACH, ScreenWatch
from tests.unit.training.test_vision import COIN, HERO, WALKER, hero_found, screen
from tests.unit.training.test_vision_what_cost_a_life import a_watch, touch

HERO_AT = 40  # where the hero is drawn in these screens


def cost_a_life(watch: ScreenWatch) -> str:
    """A walker costs the hero a life, the game puts the hero back and the walker is
    gone. Returns the walker's kind."""
    touch(watch, then_wait=2)
    (role,) = watch.roles()
    assert role.tag is SpriteTag.HAZARD
    for _ in range(4):
        watch.look(screen([(HERO_AT, 100, HERO)]))
    assert watch.threats() == ()  # nothing of that kind is in sight
    return role.kind


def test_a_thing_of_a_kind_that_cost_a_life_is_a_threat() -> None:
    watch = a_watch()
    assert watch.threats() == ()  # before any look
    watch.look(screen([(HERO_AT, 100, HERO), (120, 100, WALKER)]))
    assert watch.threats() == ()  # no life has been put down to anything
    kind = cost_a_life(watch)
    # Another of its kind comes into sight: a threat, with its kind.
    watch.look(screen([(HERO_AT, 100, HERO), (150, 100, WALKER)]))
    ((track, named),) = watch.threats()
    assert named == kind
    # The same thing at the next look is the same threat, to be looked ahead for once.
    watch.look(screen([(HERO_AT, 100, HERO), (146, 100, WALKER)]))
    assert watch.threats() == ((track, kind),)
    # A kind stays one a life was put down to however often it is then met safely.
    for _ in range(3):
        watch.look(screen([(HERO_AT, 100, HERO)]))
        watch.look(screen([(HERO_AT, 100, HERO), (60, 100, WALKER)]))
        watch.look(screen([(49, 100, HERO), (60, 100, WALKER)]))  # touched: nothing lost
        for _ in range(50):
            watch.look(screen([(49, 100, HERO), (60, 100, WALKER)]))
    assert [r.tag for r in watch.roles() if r.kind == kind] == [SpriteTag.OTHER]
    assert [named for _, named in watch.threats()] == [kind]


def test_what_is_no_threat() -> None:
    watch = a_watch()
    kind = cost_a_life(watch)
    # Further off across than the brain's view reaches: not yet. (The hero is seen with
    # its middle at 46 and a walker drawn at x with its middle at x + 7.)
    assert PART_REACH == 128
    far = 46 + PART_REACH + 1 - 7
    watch.look(screen([(HERO_AT, 100, HERO), (far, 100, WALKER)]))
    assert watch.threats() == ()
    watch.look(screen([(HERO_AT, 100, HERO), (far - 1, 100, WALKER)]))
    assert [named for _, named in watch.threats()] == [kind]
    # However far above or below: across is what counts.
    watch = a_watch()
    kind = cost_a_life(watch)
    watch.look(screen([(HERO_AT, 100, HERO), (90, 20, WALKER)]))
    assert [named for _, named in watch.threats()] == [kind]
    # Out of sight at this look: no threat, though the tracker still keeps it.
    watch.look(screen([(HERO_AT, 100, HERO)]))
    assert watch.threats() == ()
    assert any(t.kind == kind and t.missing for t in watch.tracker.tracks())
    # A thing of a kind no life was put down to.
    watch.look(screen([(HERO_AT, 100, HERO), (90, 100, COIN)]))
    assert watch.threats() == ()
    # The player's own track, and nothing at all where the player is not seen.
    watch.look(screen([(HERO_AT, 100, HERO), (90, 100, WALKER)]))
    assert len(watch.threats()) == 1
    watch.look(screen([(90, 100, WALKER)]))
    assert watch.me is None and watch.threats() == ()


def test_threats_are_nearest_first() -> None:
    watch = a_watch()
    kind = cost_a_life(watch)
    watch.look(screen([(HERO_AT, 100, HERO), (150, 100, WALKER), (90, 100, WALKER)]))
    first = watch.threats()
    assert [named for _, named in first] == [kind, kind]
    by_track = {t.id: t for t in watch.tracker.tracks()}
    assert [by_track[track].centre[0] for track, _ in first] == [97, 157]
    # Behind it counts as in front: by how far, across and up or down together.
    watch = a_watch()
    kind = cost_a_life(watch)
    watch.look(screen([(100, 100, HERO), (40, 100, WALKER), (130, 60, WALKER)]))
    by_track = {t.id: t for t in watch.tracker.tracks()}
    assert [by_track[track].centre[0] for track, _ in watch.threats()] == [47, 137]


def test_the_player_is_no_threat_to_itself() -> None:
    """A thing that looks just as the player does costs a life: the kind a life was put
    down to is then the player's own kind too. (A watch that follows the player as a
    track of its own, with no colours to know it by.)"""
    watch = ScreenWatch(hero_found())
    watch.begin_go()
    for x in (120, 104, 88, 72):
        watch.look(screen([(HERO_AT, 100, HERO), (x, 100, HERO)]))
    for _ in range(3):
        watch.look(screen([(HERO_AT, 100, HERO)]))
    watch.look(screen([(HERO_AT, 100, HERO)]), lost=1)
    (role,) = watch.roles()
    assert role.tag is SpriteTag.ENEMY and watch.lives_after_closest == 1
    for _ in range(4):
        watch.look(screen([(HERO_AT, 100, HERO)]))
    own = [t for t in watch.tracker.tracks() if t.kind == role.kind]
    assert len(own) == 1 and own[0].missing == 0  # the player's track is of that kind
    assert watch.threats() == ()
    watch.look(screen([(HERO_AT, 100, HERO), (150, 100, HERO)]))
    ((track, kind),) = watch.threats()
    assert kind == role.kind and track != own[0].id
