"""T-976: the AI watches the screen while it plays and learns what things are."""

from __future__ import annotations

import numpy as np

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
