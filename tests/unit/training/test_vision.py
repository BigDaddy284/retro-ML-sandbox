"""T-976: the AI watches the screen while it plays and learns what things are."""

from __future__ import annotations

import numpy as np
import pytest

from retro_ai_lab.core import Frame
from retro_ai_lab.observation.sprites import SpriteTag
from retro_ai_lab.training.vision import ScreenWatch
from retro_ai_lab.understanding.knowledge import Confidence
from retro_ai_lab.understanding.player import PlayerFinding

SKY = (92, 148, 252)
RED = (248, 56, 0)
DARK = (120, 40, 0)
GOLD = (252, 188, 60)
WHITE = (252, 252, 252)


def sprite(colour: tuple[int, int, int], w: int, h: int) -> np.ndarray:
    s = np.zeros((h, w, 3), np.uint8)
    s[:] = SKY
    s[1:, 1 : w - 1] = colour
    s[2, 2] = WHITE
    return s


HERO, WALKER, COIN = sprite(RED, 12, 16), sprite(DARK, 14, 14), sprite(GOLD, 6, 8)


def screen(things: list[tuple[int, int, np.ndarray]]) -> Frame:
    a = np.zeros((240, 256, 3), np.uint8)
    a[:] = SKY
    a[0:4, ::7] = (0, 0, 0)  # a little texture so the scroll reads as none
    for x, y, spr in things:
        h, w = spr.shape[:2]
        solid = (spr != np.array(SKY, np.uint8)).any(axis=2)
        a[y : y + h, x : x + w][solid] = spr[solid]
    return Frame(256, 240, 3, a.tobytes())


def hero_found() -> PlayerFinding:
    return PlayerFinding(None, (46, 108), frozenset({"left", "right"}), 2, Confidence.MEDIUM)


def walk_into(watch: ScreenWatch, other: np.ndarray, *, then: str) -> None:
    """A go: the hero walks right into ``other`` (by the 8th look) and stops there;
    a coin vanishes once touched; memory reports ``then`` at the 14th look."""
    watch.begin_go()
    ox = 120
    touched_at = None
    for i in range(15):
        hero_x = min(40 + 10 * i, ox - 11)
        if touched_at is None and hero_x == ox - 11:
            touched_at = i
        things = [(hero_x, 100, HERO)]
        gone = then == "score" and touched_at is not None and i > touched_at + 1
        if not gone:
            things.append((ox, 100, other))
        lost = 1 if then == "life" and i == 14 else 0
        scored = 1 if then == "score" and i == 14 else 0
        watch.look(screen(things), lost=lost, scored=scored)


def test_the_hero_is_picked_out_at_the_first_look() -> None:
    watch = ScreenWatch(hero_found())
    watch.begin_go()
    watch.look(screen([(40, 100, HERO), (150, 100, WALKER)]))
    me = watch.player_track
    assert me is not None
    t = next(t for t in watch.tracker.tracks() if t.id == me)
    assert abs(t.centre[0] - 46) <= 2


def test_something_that_costs_a_life_twice_is_found_to_hurt() -> None:
    watch = ScreenWatch(hero_found())
    walk_into(watch, WALKER, then="life")
    walk_into(watch, WALKER, then="life")
    roles = watch.roles()
    hurts = [r for r in roles if r.tag in (SpriteTag.ENEMY, SpriteTag.HAZARD)]
    assert len(hurts) == 1 and hurts[0].confidence is Confidence.MEDIUM
    assert watch.looks == 30


def test_something_that_vanishes_as_the_score_rises_is_an_item() -> None:
    watch = ScreenWatch(hero_found())
    walk_into(watch, COIN, then="score")
    walk_into(watch, COIN, then="score")
    assert [r.tag for r in watch.roles()] == [SpriteTag.ITEM]


def test_news_tells_each_change_once() -> None:
    watch = ScreenWatch(hero_found())
    walk_into(watch, WALKER, then="life")
    first = watch.news()
    assert len(first) == 1 and first[0].confidence is Confidence.LOW
    assert watch.news() == ()
    walk_into(watch, WALKER, then="life")
    again = watch.news()
    assert len(again) == 1 and again[0].confidence is Confidence.MEDIUM


def test_no_player_far_from_where_it_was_found() -> None:
    watch = ScreenWatch(hero_found())
    watch.begin_go()
    watch.look(screen([(200, 30, WALKER)]))
    assert watch.player_track is None


# ------------------------------------------------------------ in the training loop


def test_score_rises_are_counted_for_the_watch() -> None:
    from retro_ai_lab.core.events import EventBus
    from retro_ai_lab.services.discovery import DiscoveryUpdated
    from retro_ai_lab.training.episode import EpisodeSignals
    from retro_ai_lab.understanding.events import EventKind, GameEvent

    bus = EventBus()
    signals = EpisodeSignals(bus)
    signals.start()
    signals.watch_scores(frozenset({"score"}))

    def update(*events: GameEvent) -> DiscoveryUpdated:
        return DiscoveryUpdated((), (), tuple(events), ())

    up = GameEvent(EventKind.UP, 5, "score", "score went up")
    other = GameEvent(EventKind.UP, 5, "x", "x went up")
    bus.publish(update(up, up))
    bus.publish(update(other))
    assert signals.take_scores() == 1
    assert signals.take_scores() == 0


def test_the_player_is_found_again_after_its_track_is_lost() -> None:
    """T-977: cut out of the ground or half behind something, the player's track can
    break; the nearest thing to where it was last seen is it again."""
    watch = ScreenWatch(hero_found())
    watch.begin_go()
    watch.look(screen([(40, 100, HERO)]))
    first = watch.player_track
    assert first is not None
    watch.look(screen([]))  # gone for a moment
    for _ in range(8):  # long enough for the old track to be dropped
        watch.look(screen([]))
    watch.look(screen([(48, 100, HERO)]))
    assert watch.player_track is not None


def test_the_summary_says_how_the_looking_went() -> None:
    watch = ScreenWatch(hero_found())
    assert "not looked" in watch.describe()
    walk_into(watch, WALKER, then="life")
    text = watch.describe()
    assert "15 looks" in text and "1 lives lost, 1 of them soon after touching" in text
    assert "1 kinds found to hurt" in text


# ------------------------------------------------------------ T-978: the brain sees it


def test_the_brain_reads_how_far_the_nearest_moving_thing_is() -> None:
    watch = ScreenWatch(hero_found())
    watch.begin_go()
    assert watch.parts() == ()  # nothing looked at yet
    for i in range(4):
        watch.look(screen([(40, 100, HERO), (100 - 4 * i, 102, WALKER)]))
    parts = dict(watch.parts())
    # What is around it comes first (T-989), then the nearest moving thing.
    assert list(parts)[:5] == [k for k, _ in watch.around.parts()]
    assert list(parts)[5:] == ["screen:mover dx", "screen:mover dy"]
    assert parts["screen:mover dx"] > 0  # to the right of the hero
    assert abs(parts["screen:mover dy"]) <= 1  # level with it


def test_the_brain_reads_how_far_the_nearest_thing_that_hurts_is() -> None:
    watch = ScreenWatch(hero_found())
    walk_into(watch, WALKER, then="life")
    walk_into(watch, WALKER, then="life")
    watch.begin_go()
    watch.look(screen([(40, 100, HERO), (100, 100, WALKER)]))
    watch.look(screen([(40, 100, HERO), (100, 100, WALKER)]))
    parts = dict(watch.parts())
    assert "screen:hurts dx" in parts and parts["screen:hurts dx"] > 0


def test_far_things_and_a_missing_player_give_nothing() -> None:
    watch = ScreenWatch(hero_found())
    watch.begin_go()
    for i in range(3):
        watch.look(screen([(40, 100, HERO), (240 - i, 100, WALKER)]))  # 200 px away
    assert all(not k.startswith("screen:mover") for k, _ in watch.parts())
    watch.begin_go()
    watch.look(screen([(200, 30, WALKER)]))
    assert watch.parts() == ()


def test_the_state_holds_the_screen_parts_in_seats_of_their_own() -> None:
    from retro_ai_lab.learning.state import SCREEN_SEATS, StateEncoder
    from retro_ai_lab.observation.memory import MemorySnapshot
    from retro_ai_lab.understanding.knowledge import Knowledge

    enc = StateEncoder(Knowledge("g"), most=8)
    snap = MemorySnapshot(0, {"ram": bytes(16)})
    assert enc.encode(snap).parts == ()
    enc.set_screen(lambda: (("screen:mover dx", 3), ("screen:mover dy", 0)))
    state = enc.encode(snap)
    assert state.parts == (("screen:mover dx", 3), ("screen:mover dy", 0))
    assert "nearest moving thing across=3" in state.describe()
    assert enc.sees_screen
    enc.set_screen(None)
    assert enc.encode(snap).parts == () and not enc.sees_screen
    assert SCREEN_SEATS == 12  # T-989: five for what is around it, four for things, room


@pytest.mark.parametrize(
    ("apart", "near", "want"),
    [
        (0, 48, 0),
        (8, 48, 0),
        (-8, 48, 0),
        (9, 48, 1),
        (48, 48, 1),
        (49, 48, 2),
        (-30, 48, -1),
        (-200, 48, -2),
        (100, None, 1),
        (-9, None, -1),
    ],
)
def test_distances_become_a_few_coarse_answers(apart: int, near: int | None, want: int) -> None:
    """Exact pixels would split the brain's table into thousands of states (T-978)."""
    from retro_ai_lab.training.vision import side

    assert side(apart, near=near) == want


# ------------------------------------------------------------ T-987: it knows itself by its colours

BROWN = (200, 76, 12)
BLACK = (0, 0, 0)
RED_ONLY = (RED[0] << 16) | (RED[1] << 8) | RED[2]


def hero_with_colours(start: tuple[int, int] = (46, 200)) -> PlayerFinding:
    from retro_ai_lab.observation.signature import ColourSignature

    # The hero sprite: red 10 by 15, less its one white mark.
    sign = ColourSignature(((RED_ONLY, 149),), 10, 15)
    return PlayerFinding(None, start, frozenset({"left", "right"}), 2, Confidence.MEDIUM, sign)


def level(camera: int, things: list[tuple[int, int, np.ndarray]]) -> Frame:
    """Brick ground from y=208 and hills behind, seen from ``camera``; the things are
    given where they are on the screen."""
    a = np.zeros((240, 1024, 3), np.uint8)
    a[:] = SKY
    for x0 in range(0, 1024, 97):
        a[150:208, x0 : x0 + 30] = (0, 168, 0)
    for x0 in range(0, 1024, 16):
        a[208:240, x0 : x0 + 16] = BROWN
        a[208, x0 : x0 + 16] = BLACK
        a[208:240, x0] = BLACK
    a = a[:, camera : camera + 256].copy()
    for x, y, spr in things:
        h, w = spr.shape[:2]
        solid = (spr != np.array(SKY, np.uint8)).any(axis=2)
        a[y : y + h, x : x + w][solid] = spr[solid]
    return Frame(256, 240, 3, a.tobytes())


def a_go(watch: ScreenWatch) -> list[tuple[int, int]]:
    """Standing still on the ground, a jump of 36 pixels a look, then walking with the
    camera locked on it (it stays put on the screen while the level slides by 16
    pixels a look). Returns where the hero's box really was at each look."""
    watch.begin_go()
    spots: list[tuple[int, int]] = []
    for _ in range(4):  # standing still: nothing on the screen changes
        spots.append((40, 192))
    for dy in (36, 60, 36, 0):  # up and down again, far between two looks
        spots.append((52, 192 - dy))
    for _ in range(6):  # camera locked
        spots.append((52, 192))
    camera = 0
    for i, (x, y) in enumerate(spots):
        if i >= 8:
            camera += 16
        watch.look(level(camera, [(x, y, HERO)]))
        spots[i] = (x, y)
        yield_me = watch.me
        if yield_me is not None:
            assert yield_me == (x + 1, y + 1, 10, 15)
    return spots


def test_with_colours_it_has_itself_through_a_jump_and_standing_still() -> None:
    watch = ScreenWatch(hero_with_colours())
    spots = a_go(watch)
    # In every look, exactly where it was (a_go checks the box each time it has one).
    assert watch.looks == len(spots) == 14 and watch.seen_me == 14
    assert watch.player_track is None  # no track is taken for it
    assert watch.me == (53, 193, 10, 15)
    # Out of sight (a death) it says so, and has itself again when it is back: at a
    # new go, the other end of the screen from where it was found.
    watch.look(level(96, []))
    assert watch.me is None and watch.parts() == () and watch.seen_me == 14
    watch.begin_go()
    watch.look(level(0, [(200, 100, HERO)]))
    assert watch.me == (201, 101, 10, 15) and watch.seen_me == 15
    assert "itself followed in 94% of them" in watch.describe()
    # The same go with no colours to go by: standing on the ground it is one patch with
    # it, and it is lost for most of the go (what T-987 is for).
    blind = ScreenWatch(PlayerFinding(None, (46, 200), frozenset({"a"}), 2, Confidence.MEDIUM))
    blind.begin_go()
    for i in range(14):
        x, y = (40, 192) if i < 4 else (52, 192 - (36, 60, 36, 0)[i - 4] if i < 8 else 192)
        blind.look(level(max(0, 16 * (i - 7)), [(x, y, HERO)]))
    assert blind.seen_me < 7


def test_with_colours_a_touch_counts_and_its_own_track_does_not() -> None:
    watch = ScreenWatch(hero_with_colours((46, 108)))
    walk_into(watch, WALKER, then="life")
    walk_into(watch, WALKER, then="life")
    assert watch.seen_me == watch.looks == 30
    roles = watch.roles()
    # One kind was touched, the walker's, and it hurts; the hero's own patch is not a
    # thing it touched.
    assert len(roles) == 1 and roles[0].tag in (SpriteTag.ENEMY, SpriteTag.HAZARD)
    assert roles[0].confidence is Confidence.MEDIUM and roles[0].touches == 2
    assert watch.lives_lost == 2 and watch.lives_after_touch == 2
    # The brain's view is measured from the box.
    watch.begin_go()
    for i in range(3):
        watch.look(screen([(40, 100, HERO), (100 - 4 * i, 100, WALKER)]))
    parts = dict(watch.parts())
    assert parts["screen:mover dx"] > 0 and parts["screen:hurts dx"] > 0


def test_the_findings_kind_name_is_not_taken_for_the_watchs_own() -> None:
    """The finding's kind is a name from the finder's catalogue; in the watch's own
    catalogue the same name is whatever it happened to see first."""
    from retro_ai_lab.observation.objects import ObjectCatalogue

    things = [(150, 30, COIN), (40, 100, HERO)]  # the coin is met first: "thing-1"
    found = PlayerFinding("thing-1", (46, 108), frozenset({"left", "right"}), 2, Confidence.MEDIUM)
    watch = ScreenWatch(found)
    watch.begin_go()
    watch.look(screen(things))
    assert watch.me is not None and abs(watch.me[0] - 41) <= 1  # the hero, where it was found
    # With the finder's own catalogue handed over, the name means what it says: the
    # coin's kind is followed even though the finding's start is nowhere near it.
    catalogue = ObjectCatalogue()
    kinds = {o.x: o.kind for o in catalogue.observe(screen(things)) if not o.scenery}
    far = PlayerFinding(kinds[41], (250, 230), frozenset({"left", "right"}), 2, Confidence.MEDIUM)
    shared = ScreenWatch(far, catalogue=catalogue)
    shared.begin_go()
    shared.look(screen(things))
    shared.look(screen(things))
    assert shared.me is not None and abs(shared.me[0] - 41) <= 1


# ------------------------------------------------------------ T-988: a fall


def fall(watch: ScreenWatch, *, extra: list[tuple[int, int, np.ndarray]] | None = None) -> None:
    """The hero drops out of the bottom of the screen: seen coming down at y=150 and
    y=190 with nothing but sky under it, then gone."""
    things = extra or []
    watch.look(screen([(60, 150, HERO), *things]))
    watch.look(screen([(60, 190, HERO), *things]))
    watch.look(screen(things))


def test_heading_off_the_bottom_then_a_life_lost_is_a_fall() -> None:
    watch = ScreenWatch(hero_with_colours((46, 108)))
    watch.begin_go()
    watch.look(screen([(40, 100, HERO)]))
    fall(watch)
    for _ in range(4):  # the dying pause: nothing to see
        watch.look(screen([]))
    watch.look(screen([(40, 100, HERO)]), lost=1)  # back at the start, a life short
    assert (watch.lives_lost, watch.lives_after_fall, watch.lives_after_touch) == (1, 1, 0)
    assert watch.roles() == ()  # nothing was touched, nothing is blamed
    # Counted once: the next life, lost standing where it is, is no fall.
    watch.look(screen([(40, 100, HERO)]), lost=1)
    assert (watch.lives_lost, watch.lives_after_fall, watch.lives_after_touch) == (2, 1, 0)
    assert "2 lives lost, 0 of them soon after touching something and 1 after falling off" in (
        watch.describe()
    )
    # A new go forgets a fall that no life followed.
    fall(watch)
    watch.begin_go()
    watch.look(screen([(40, 100, HERO)]), lost=1)
    assert watch.lives_after_fall == 1


def test_landing_near_the_bottom_or_vanishing_otherwise_is_no_fall() -> None:
    def lost_after(looks: list[list[tuple[int, int, np.ndarray]]], **kw: int) -> ScreenWatch:
        watch = ScreenWatch(hero_with_colours((46, 108)), **kw)
        watch.begin_go()
        for things in looks:
            watch.look(screen(things))
        watch.look(screen([(40, 100, HERO)]), lost=1)
        assert watch.lives_lost == 1
        return watch

    hero = lambda y, x=60: [(x, y, HERO)]  # noqa: E731
    # The same drop as a fall, to be sure the scene is one.
    assert lost_after([hero(150), hero(190), []]).lives_after_fall == 1
    # Coming down just as fast, then seen standing where it landed before it goes.
    assert lost_after([hero(150), hero(190), hero(190), []]).lives_after_fall == 0
    # Seen again after it went out of sight coming down (behind something, a flicker):
    # it did not fall out of the screen.
    assert lost_after([hero(150), hero(190), [], hero(190)]).lives_after_fall == 0
    # Standing still, then gone (a door, a flash).
    assert lost_after([hero(190), hero(190), []]).lives_after_fall == 0
    # Going up, then gone.
    assert lost_after([hero(220), hero(190), []]).lives_after_fall == 0
    # Coming down only a little, high up the screen, then gone: with nothing under it
    # that is a fall all the same (a fall starts slowly).
    assert lost_after([hero(50), hero(59), []]).lives_after_fall == 1
    # Seen once only before it went: no telling which way it was going.
    assert lost_after([hero(190), []]).lives_after_fall == 0
    # Out of sight between the two sightings: they are not one movement.
    assert lost_after([hero(150), [], hero(190), []]).lives_after_fall == 0
    # A fall, but the life went long after: something else took it.
    late = lost_after([hero(150), hero(190), [], [], [], [], [], []], soon=4)
    assert late.lives_after_fall == 0
    assert lost_after([hero(150), hero(190), [], [], [], []], soon=4).lives_after_fall == 1
    # A watch that follows a track (no colours) knows no falls, as before.
    blind = ScreenWatch(hero_found())
    blind.begin_go()
    blind.look(screen([(40, 100, HERO)]))
    for things in ([(40, 150, HERO)], [(40, 190, HERO)], []):
        blind.look(screen(things))
    blind.look(screen([(40, 100, HERO)]), lost=1)
    assert (blind.lives_lost, blind.lives_after_fall) == (1, 0)
    assert "falling" not in blind.describe()


def over_ground(things: list[tuple[int, int, np.ndarray]]) -> Frame:
    """The plain test screen with a floor across the bottom (y=224 down)."""
    floor = np.zeros((16, 256, 3), np.uint8)
    floor[:] = BROWN
    return screen([(0, 224, floor), *things])


def test_coming_down_through_the_floor_is_no_fall() -> None:
    """Many games throw a player that was hit off the bottom of the screen, through
    the ground. That is not a fall: there was something under it."""
    watch = ScreenWatch(hero_with_colours((46, 108)))
    watch.begin_go()
    watch.look(over_ground([(40, 100, HERO)]))
    watch.look(over_ground([(60, 150, HERO)]))
    watch.look(over_ground([(60, 190, HERO)]))  # coming down, the floor under it
    watch.look(over_ground([]))
    watch.look(over_ground([(40, 100, HERO)]), lost=1)
    assert (watch.lives_lost, watch.lives_after_fall, watch.lives_after_touch) == (1, 0, 0)
    # The same over a gap in the floor: a fall.
    gap = np.zeros((16, 40, 3), np.uint8)
    gap[:] = SKY

    def holed(things: list[tuple[int, int, np.ndarray]]) -> Frame:
        a = np.frombuffer(over_ground(things).pixels, np.uint8).reshape(240, 256, 3).copy()
        a[224:, 50:90] = SKY
        return Frame(256, 240, 3, a.tobytes())

    watch.begin_go()
    watch.look(holed([(40, 100, HERO)]))
    watch.look(holed([(60, 150, HERO)]))
    watch.look(holed([(60, 190, HERO)]))
    watch.look(holed([]))
    watch.look(holed([(40, 100, HERO)]), lost=1)
    assert (watch.lives_lost, watch.lives_after_fall, watch.lives_after_touch) == (2, 1, 0)
    # Half over the edge of the gap there is still floor under it: no fall.
    watch.begin_go()
    watch.look(holed([(40, 100, HERO)]))
    watch.look(holed([(45, 150, HERO)]))
    watch.look(holed([(45, 190, HERO)]))
    watch.look(holed([]))
    watch.look(holed([(40, 100, HERO)]), lost=1)
    assert (watch.lives_lost, watch.lives_after_fall) == (3, 1)


def test_a_fall_is_not_blamed_on_what_was_touched_before_it() -> None:
    """On the practice game the AI stood on a narrow pillar, fell into the gap beside
    it, and the pillar was "found to hurt"."""
    walker = [(120, 100, WALKER)]

    def touch_then(watch: ScreenWatch, *, floor: bool) -> None:
        """Touches the walker, then goes down out of the screen beside it; a life goes."""
        draw = over_ground if floor else screen
        watch.begin_go()
        watch.look(draw([(40, 100, HERO), *walker]))
        watch.look(draw([(109, 100, HERO), *walker]))  # touching
        watch.look(draw([(109, 150, HERO), *walker]))
        watch.look(draw([(109, 200, HERO), *walker]))
        watch.look(draw(walker))
        watch.look(draw([(40, 100, HERO), *walker]), lost=1)

    # Nothing under it: the fall explains the life, and the walker is not blamed.
    fell = ScreenWatch(hero_with_colours((46, 108)))
    touch_then(fell, floor=False)
    assert (fell.lives_after_touch, fell.lives_after_fall) == (0, 1)
    assert [r.tag for r in fell.roles()] == [SpriteTag.OTHER]  # touched, and harmless
    # The touch that was waiting was let go: the next life lost is not put down to it.
    fell.look(screen([(40, 100, HERO), *walker]), lost=1)
    assert (fell.lives_lost, fell.lives_after_touch, fell.lives_after_fall) == (2, 0, 1)
    # Thrown off the screen through the floor after the touch: the touch did it.
    hit = ScreenWatch(hero_with_colours((46, 108)))
    touch_then(hit, floor=True)
    assert (hit.lives_after_touch, hit.lives_after_fall) == (1, 0)
    assert len(hit.roles()) == 1 and hit.roles()[0].tag is not SpriteTag.OTHER


def test_told_without_a_screen() -> None:
    """The go ended before the next look: a life lost is still counted and explained."""
    watch = ScreenWatch(hero_with_colours((46, 108)))
    watch.begin_go()
    watch.look(screen([(40, 100, HERO)]))
    fall(watch)
    watch.told(lost=1)
    assert (watch.looks, watch.lives_lost, watch.lives_after_fall) == (4, 1, 1)
    watch.told(lost=0)
    watch.told()
    assert watch.lives_lost == 1
    # With a touch waiting, the touch is blamed.
    watch.begin_go()
    watch.look(screen([(40, 100, HERO), (120, 100, WALKER)]))
    watch.look(screen([(109, 100, HERO), (120, 100, WALKER)]))
    watch.told(lost=1)
    assert (watch.lives_lost, watch.lives_after_touch, watch.lives_after_fall) == (2, 1, 1)
    assert len(watch.roles()) == 1 and watch.roles()[0].tag is not SpriteTag.OTHER
    # No touch, no fall: counted, and nobody's fault.
    watch.begin_go()
    watch.look(screen([(40, 100, HERO)]))
    watch.told(lost=2)
    assert (watch.lives_lost, watch.lives_after_touch, watch.lives_after_fall) == (4, 1, 1)


def test_the_summary_names_the_falls() -> None:
    watch = ScreenWatch(hero_with_colours((46, 108)))
    walk_into(watch, WALKER, then="life")
    assert "1 lives lost, 1 of them soon after touching something; 1 kinds" in watch.describe()
    watch.begin_go()
    watch.look(screen([(40, 100, HERO)]))
    fall(watch)
    watch.look(screen([(40, 100, HERO)]), lost=1)
    assert (
        "2 lives lost, 1 of them soon after touching something and 1 after falling off the "
        "bottom of the screen; 1 kinds found to hurt" in watch.describe()
    )


# ------------------------------------------------------------ T-989: what is around it


def gap_ahead(hero_x: int, gap_x: int, wide: int, *, y: int = 192) -> Frame:
    """Plain ground from y=208 with a gap in it, and the hero."""
    a = np.zeros((240, 256, 3), np.uint8)
    a[:] = SKY
    a[208:] = BROWN
    a[208] = BLACK
    a[208:, gap_x : gap_x + wide] = SKY
    solid = (np.array(SKY, np.uint8) != HERO).any(axis=2)
    a[y : y + 16, hero_x : hero_x + 12][solid] = HERO[solid]
    return Frame(256, 240, 3, a.tobytes())


def test_the_brain_is_given_what_is_around_it() -> None:
    from retro_ai_lab.observation.surroundings import (
        MOST,
        NOTHING_BELOW,
        ON_SOMETHING,
        Surroundings,
    )

    watch = ScreenWatch(hero_with_colours())
    watch.begin_go()
    assert watch.around is None and watch.parts() == ()
    # Walking up to a gap 20 pixels a look: its box is (x+1, 193, 10, 15), so the first
    # column read to its right is x+12, and the gap starts at 150.
    for x, floor in ((40, MOST), (80, MOST), (100, 4), (120, 2), (136, 0)):
        watch.look(gap_ahead(x, 150, 48))
        assert watch.around == Surroundings(floor, MOST, MOST, MOST, ON_SOMETHING), x
        assert watch.parts() == watch.around.parts()  # nothing moves: these are all it reads
    # In the air over the gap.
    watch.look(gap_ahead(160, 150, 48, y=150))
    assert watch.around is not None
    assert (watch.around.under, watch.around.floor_right, watch.around.floor_left) == (
        NOTHING_BELOW,
        0,
        0,
    )
    # With hills behind the ground (decoration it cannot tell from a wall, a recorded
    # limit): the hill 10 pixels behind it and the one 45 ahead read as in the way.
    watch.begin_go()
    watch.look(level(0, [(40, 192, HERO)]))
    assert watch.around == Surroundings(MOST, MOST, 5, 1, ON_SOMETHING)
    # A watch that follows a track (no colours to go by) reads the same from its box.
    blind = ScreenWatch(hero_found())
    blind.begin_go()
    blind.look(screen([(40, 100, HERO)]))
    assert blind.around == Surroundings(0, 0, MOST, MOST, NOTHING_BELOW)
    assert blind.parts() == blind.around.parts()


def test_nothing_is_given_when_it_does_not_see_itself() -> None:
    watch = ScreenWatch(hero_with_colours())
    watch.begin_go()
    watch.look(gap_ahead(40, 150, 48))
    assert watch.around is not None and len(watch.parts()) == 5
    # Gone from the screen: no reading, and none left over from the look before.
    watch.look(level(0, []))
    assert watch.me is None and watch.around is None and watch.parts() == ()
    # Back in sight, then a new go: nothing is carried over from the last one.
    watch.look(gap_ahead(40, 150, 48))
    assert watch.around is not None
    watch.begin_go()
    assert watch.around is None and watch.parts() == ()


# ------------------------------------------------------------ T-992: the camera at its pace


def test_the_watch_follows_a_fast_camera() -> None:
    """The AI looks every 8 frames; in that time a camera moves well past the 8 pixels
    the tracker allowed for one frame. The player, held still on the screen by the
    camera, has no say in how far it moved."""
    watch = ScreenWatch(hero_with_colours())
    watch.begin_go()
    camera = 0
    for step in (0, 16, 16, 24, 31, 0, -20, -16, 3):
        camera += step
        watch.look(level(camera + 100, [(80, 192, HERO)]))
        assert watch.tracker.scroll == (step, 0), step
        assert watch.me == (81, 193, 10, 15)
    assert watch.tracker.camera == (54, 0)
    # A new go starts the camera's count again.
    watch.begin_go()
    assert watch.tracker.camera == (0, 0)


# ------------------------------------------------------------ T-993: new ground


def indoors(camera: int, things: list[tuple[int, int, np.ndarray]]) -> Frame:
    """Somewhere else altogether: a dark room with lamps and a grey floor."""
    a = np.zeros((240, 1024, 3), np.uint8)
    a[60:180:40, 30::70] = WHITE
    a[90:96, 11::53] = GOLD
    a[200:] = (60, 60, 60)
    a[200, ::9] = WHITE
    a = a[:, camera : camera + 256].copy()
    for x, y, spr in things:
        h, w = spr.shape[:2]
        solid = (spr != np.array(SKY, np.uint8)).any(axis=2)
        a[y : y + h, x : x + w][solid] = spr[solid]
    return Frame(256, 240, 3, a.tobytes())


def flat(camera: int, things: list[tuple[int, int, np.ndarray]]) -> Frame:
    """Plain ground with nothing to tell one place from another but two wide gaps (at
    300 and 700) and a tall post near the start: a jump from one place to another
    changes little of the screen."""
    a = np.zeros((240, 1024, 3), np.uint8)
    a[:] = SKY
    a[208:] = BROWN
    a[208:212] = (0, 168, 0)
    for gap in (300, 700):
        a[208:, gap : gap + 72] = SKY
    a[128:208, 150:166] = (188, 188, 188)
    a = a[:, camera : camera + 256].copy()
    for x, y, spr in things:
        h, w = spr.shape[:2]
        solid = (spr != np.array(SKY, np.uint8)).any(axis=2)
        a[y : y + h, x : x + w][solid] = spr[solid]
    return Frame(256, 240, 3, a.tobytes())


HELD = [(80, 192, HERO)]  # the camera holds it here; its box is 10 by 15, so a step is 15


def test_scrolling_scenery_pays_the_cameras_travel() -> None:
    watch = ScreenWatch(hero_with_colours((86, 200)))
    watch.begin_go()
    paid = []
    for i in range(10):
        watch.look(level(100 + 16 * i, HELD))
        paid.append(watch.take_new_ground())
    # 144 pixels in nine moves of 16: a step for every 15 of them, each paid once.
    assert sum(paid) == 144 // 15 == 9 and paid[0] == 0 and max(paid) <= 2
    assert watch.ground.reach == (144, 0, 0, 0) and watch.ground.unit == 15
    assert watch.ground.steps == 9
    # Standing still, and walking back, pay nothing.
    for camera in (244, 244, 228, 212, 196):
        watch.look(level(camera, HELD))
        assert watch.take_new_ground() == 0


@pytest.mark.parametrize("colours", [True, False])
def test_walking_across_a_still_screen_pays(colours: bool) -> None:
    # No camera move at all: it walks 8 pixels a look across the screen, known by its
    # colours or followed as a track.
    y = 192 if colours else 100
    watch = ScreenWatch(hero_with_colours((46, 200)) if colours else hero_found())
    watch.begin_go()
    paid = []
    for i in range(10):
        things = [(40 + 8 * i, y, HERO)]
        watch.look(level(100, things) if colours else screen(things))
        assert watch.tracker.scroll == (0, 0)
        paid.append(watch.take_new_ground())
    assert sum(paid) == 72 // (15 if colours else 16) == 4
    assert watch.ground.reach[0] == 72


def test_a_cut_puts_it_back_where_the_go_began() -> None:
    watch = ScreenWatch(hero_with_colours((86, 200)))
    watch.begin_go()
    for i in range(5):
        watch.look(level(100 + 16 * i, HELD))
    assert watch.take_new_ground() == 64 // 15 == 4
    # Through a door: nothing of the screen is as it was, and no camera move explains it.
    watch.look(indoors(400, HELD))
    assert watch.tracker.unexplained >= 0.25
    assert watch.ground.place == (0, 0) and watch.ground.reach == (64, 0, 0, 0)
    assert watch.take_new_ground() == 0
    # On through the room: nothing until it is further from the door than the go's
    # furthest (the fifth move makes 80 pixels: the fifth step).
    walk = []
    for i in range(1, 7):
        watch.look(indoors(400 + 16 * i, HELD))
        walk.append(watch.take_new_ground())
    assert walk == [0, 0, 0, 0, 1, 1]


START = [(32, 192, HERO)]  # where the game puts it back: not where the camera holds it


@pytest.mark.parametrize("late", [0, 2, -2])
def test_a_life_lost_with_a_jump_of_the_screen_puts_it_back(late: int) -> None:
    """Put back at the start after dying, on a level so plain that the jump changes
    little of the screen: it is seen somewhere else on the screen than it was, and
    with the life lost in the same look, or a couple of looks before or after, it is
    taken to be back where the go began."""
    watch = ScreenWatch(hero_with_colours((86, 200)))
    watch.begin_go()
    for i in range(9):
        watch.look(flat(16 * i, HELD))
    assert watch.take_new_ground() == 128 // 15 == 8
    for i in range(-3, 4):  # three looks either side of the jump at i == 0
        frame = flat(128, HELD) if i < 0 else flat(0, START)
        watch.look(frame, lost=1 if i == late else 0)
        if i == 0:
            assert watch.tracker.scroll == (0, 0)
            assert 0.02 <= watch.tracker.unexplained < 0.25
    assert watch.ground.place == (0, 0) and watch.ground.reach == (128, 0, 0, 0)
    # The same ground again pays nothing; the ninth move is past the furthest.
    walk = []
    for i in range(1, 10):
        watch.look(flat(16 * i, HELD))
        walk.append(watch.take_new_ground())
    assert walk == [0] * 8 + [1]


def test_a_hit_does_not_put_it_back() -> None:
    watch = ScreenWatch(hero_with_colours((86, 200)))
    watch.begin_go()
    for i in range(5):
        watch.look(level(100 + 16 * i, HELD))
    assert watch.take_new_ground() == 4
    # A life (or some health) lost and the screen carries on as it was: it is where
    # it was, and the next ground is new.
    watch.look(level(180, HELD), lost=1)
    watch.look(level(196, HELD))
    assert watch.ground.place == (96, 0)
    assert watch.take_new_ground() == 96 // 15 - 4 == 2
    # A life lost with a jolt of the screen that leaves it where it was on the screen
    # (a crash: the thing it ran into is gone) is no putting back either.
    crash = ScreenWatch(hero_with_colours((86, 200)))
    crash.begin_go()
    for i in range(9):
        crash.look(flat(16 * i, HELD))
    for _ in range(3):
        crash.look(flat(128, HELD))
    crash.look(flat(0, HELD), lost=1)  # as much of the screen changes as when put back
    assert 0.02 <= crash.tracker.unexplained < 0.25
    assert crash.ground.place == (128, 0)
    # And a jolt that finds it somewhere else, long after the last life was lost, is
    # not put down to that life.
    late = ScreenWatch(hero_with_colours((86, 200)))
    late.begin_go()
    for i in range(9):
        late.look(flat(16 * i, HELD), lost=1 if i == 2 else 0)
    for _ in range(3):
        late.look(flat(128, HELD))
    late.look(flat(0, START))
    assert late.tracker.scroll == (0, 0) and late.ground.place == (128, 0)


def test_a_camera_it_cannot_follow_pays_nothing() -> None:
    """Past the tracker's reach (32 pixels a look, less on a small screen) the camera's
    move cannot be measured, and nothing is counted for that look. Where that is
    common, counting the moves that can be measured would pay for going slowly: so
    once a fifth of the last moves (20 at least) could not be measured, nothing is
    counted at all, and the summary says so."""
    watch = ScreenWatch(hero_with_colours((86, 200)))
    watch.begin_go()
    for camera in (100, 116, 132, 148):
        watch.look(flat(camera, HELD))
    assert watch.ground.place == (48, 0) and watch.take_new_ground() == 3
    # One look of 100 pixels: not measured, nothing counted, and no harm done yet.
    watch.look(flat(248, HELD))
    assert watch.tracker.scroll == (0, 0) and watch.tracker.unexplained >= 0.02
    assert watch.ground.place == (48, 0) and watch.follows_camera
    watch.look(flat(264, HELD))
    assert watch.take_new_ground() == 1  # the next measured move pays as before
    assert "cannot follow" not in watch.describe()
    # Now it races, back and forth 100 pixels a look: with the one before, the 19th
    # such look makes 20 moves that could not be measured, of 24 in all.
    for i in range(19):
        assert watch.follows_camera
        watch.look(flat(248 if i % 2 else 348, HELD))
        assert watch.tracker.scroll == (0, 0)
    assert not watch.follows_camera
    assert watch.describe().endswith(
        " It cannot follow the camera here (20 of the last 24 moves could not be measured), "
        "so new ground is not counted."
    )
    # A slow stretch that can be measured is counted and not paid, then or later ...
    watch.begin_go()
    for i in range(12):
        watch.look(flat(100 + 16 * i, HELD))
    assert watch.ground.place == (176, 0) and watch.ground.steps == 11
    assert watch.take_new_ground() == 0 and not watch.follows_camera
    # ... until the moves it cannot measure are under a fifth of the last 200 again.
    for _ in range(5):
        watch.begin_go()
        for i in range(16):
            watch.look(flat(100 + 16 * i, HELD))
    assert watch.follows_camera
    watch.begin_go()
    for i in range(5):
        watch.look(flat(100 + 16 * i, HELD))
    assert watch.take_new_ground() == 4


def test_new_ground_is_taken_once_and_a_go_starts_from_nothing() -> None:
    watch = ScreenWatch(hero_with_colours((86, 200)))
    assert watch.take_new_ground() == 0
    watch.begin_go()
    for i in range(5):
        watch.look(level(100 + 16 * i, HELD))
    assert watch.take_new_ground() == 4
    assert watch.take_new_ground() == 0
    watch.look(level(180, HELD))  # a step found and not taken before the go ends
    watch.begin_go()
    assert watch.take_new_ground() == 0
    assert watch.ground.steps == 0 and watch.ground.reach == (0, 0, 0, 0)
    # The next go is paid for the same ground: each go starts over.
    for i in range(5):
        watch.look(level(100 + 16 * i, HELD))
    assert watch.take_new_ground() == 4
