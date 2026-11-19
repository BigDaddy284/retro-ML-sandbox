"""T-1058: where the screen stands still, keeping the life is the way past.

A search after a life lost takes a try for the way when it stays alive and reaches
new ground. A game of one screen has no new ground once its floor is walked, so no
search there could ever succeed. Where the screen has not scrolled for a good while
the first play that keeps the life is the way.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import (
    DYING,
    GAME_OVER,
    GROUND_Y,
    LAYOUT,
    PLAYER_H,
    PLAYER_W,
    PLAYING,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    START_LIVES,
    ToyHopper,
)
from retro_ai_lab.observation.ground import STILL_LOOKS, Ground
from retro_ai_lab.training import EpisodeRunner, episode, search
from retro_ai_lab.training.episode import STAND_AFTER
from retro_ai_lab.training.search import BUDGET, CARRY_ON, TOP, Search, Way
from retro_ai_lab.training.trial import Outcome
from retro_ai_lab.training.vision import DEFAULT_LOOK_EVERY
from retro_ai_lab.training.ways import HOLDS, Change, changes
from tests.unit.training import test_going_back as going_back
from tests.unit.training import test_trial as trial
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_search import ACROSS, ALIVE, DEAD, Script, look_for
from tests.unit.training.test_search_at_a_loss import searching
from tests.unit.training.test_standing_still import NEAR, hop_about, standing
from tests.unit.training.test_trial import SPOT, Hopper

hopper = trial.hopper  # the practice game behind a connection (pytest fixture)
lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LOG = "retro_ai_lab.training.episode"
SAID = "The screen has not scrolled for a good while"


def lost_at(step: int) -> Outcome:
    return Outcome(steps=step, lost=True, ground=0)


def scripted(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[Outcome | Exception]
) -> tuple[Script, int, int]:
    """The plays scripted, and the two moves of these tests: back and on."""
    script = Script(outcomes)
    monkeypatch.setattr(search, "play", script)
    return script, hopper.move("left"), hopper.move("right")


def no_watch() -> object:
    raise AssertionError("no watch is asked for where the screen stands still")


# ------------------------------------------------------------ the rule, with scripted tries


def test_a_plain_way_that_keeps_the_life_is_the_way(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, back, on = scripted(hopper, monkeypatch, [ALIVE])
    found = look_for(hopper, [back] * 10, no_watch, carry_on=4, plain=on, still=True)
    held = (on,) * 14
    # One play, blind: alive at the end of it is all there is to ask.
    assert found == Search(Way(held, Change(0, on, 10), tries=1, ground=0, plain=on), 1)
    assert found.way is not None and found.way.lost_at is None
    assert script.calls == [(held, False)] and found.stopped_at == TOP
    # Without it the same play is looked at again, and alive on no new ground is no way.
    script, back, on = scripted(hopper, monkeypatch, [ALIVE, ALIVE] + [DEAD] * 500)
    without = look_for(hopper, [back] * 10, object, carry_on=4, plain=on, budget=6)
    assert without.way is None and script.calls[:2] == [(held, False), (held, True)]


def test_where_the_plain_way_loses_the_life_the_first_change_that_keeps_it_is_the_way(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, back, on = scripted(hopper, monkeypatch, [lost_at(6), DEAD, DEAD, ALIVE])
    found = look_for(hopper, [back] * 10, no_watch, carry_on=4, plain=on, still=True)
    base = (on,) * 6
    listed = list(changes(base, hopper.space))
    way = found.way
    assert way == Way(listed[2].apply(base, 4), listed[2], tries=4, ground=0, plain=on, lost_at=6)
    assert found.tries == 4
    assert script.calls == [
        ((on,) * 14, False),
        (listed[0].apply(base, 4), False),
        (listed[1].apply(base, 4), False),
        (listed[2].apply(base, 4), False),  # alive: the way, and not played again
    ]


def test_with_no_plain_move_the_first_change_to_what_it_did_that_keeps_the_life(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, _, on = scripted(hopper, monkeypatch, [DEAD, ALIVE])
    trace = [on] * 10
    listed = list(changes(trace, hopper.space))
    found = look_for(hopper, trace, no_watch, carry_on=4, still=True)
    assert found == Search(Way(listed[1].apply(trace, 4), listed[1], tries=2, ground=0), 2)
    assert found.way is not None and found.way.plain is None
    assert script.calls == [(listed[0].apply(trace, 4), False), (listed[1].apply(trace, 4), False)]
    # Where it stopped is the change that got past, as for any way found.
    last = len(trace) - 1
    assert found.stopped_at == (
        (0, 0),
        (HOLDS.index(listed[1].hold), last - listed[1].at),
    )


def test_the_plain_ways_changes_come_first_and_then_those_to_what_it_did(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The plain way loses the life at once and so does every change to it within its
    # half of eight tries; the first change to what it did keeps the life.
    script, back, on = scripted(hopper, monkeypatch, [lost_at(10)] + [DEAD] * 3 + [ALIVE])
    trace = [back] * 10
    found = look_for(hopper, trace, no_watch, carry_on=4, plain=on, budget=8, still=True)
    first = next(changes(trace, hopper.space))
    assert found.way == Way(first.apply(trace, 4), first, tries=5, ground=0)
    assert [looked for _, looked in script.calls] == [False] * 5
    assert script.calls[-1][0] == first.apply(trace, 4)


def test_a_search_that_keeps_no_life_ends_at_its_budget(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, _, on = scripted(hopper, monkeypatch, [DEAD] * 500)
    out = look_for(hopper, [on] * 10, no_watch, carry_on=4, budget=7, still=True)
    assert out == Search(None, 7) and not out.out_of_time
    assert len(script.calls) == 7 and not any(looked for _, looked in script.calls)
    # It says where it stopped, for the next search about there.
    assert out.stopped_at != TOP


def test_a_search_that_keeps_no_life_ends_when_its_time_is_up(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, on = scripted(hopper, monkeypatch, [DEAD] * 500)
    now = [0.0]

    def clock() -> float:
        now[0] += 1.0
        return now[0]

    out = look_for(hopper, [on] * 10, no_watch, carry_on=4, seconds=6.0, clock=clock, still=True)
    assert out.way is None and out.out_of_time and 0 < out.tries < BUDGET


def test_the_changes_are_placed_before_the_loss_was_settled(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, _, on = scripted(hopper, monkeypatch, [DEAD] * 500)
    trace = [on] * 10
    look_for(hopper, trace, no_watch, carry_on=4, live=4, budget=500, still=True)
    allowed = {change.apply(trace, 4) for change in changes(trace, hopper.space, upto=4)}
    assert {moves for moves, _ in script.calls} == allowed
    # Settled before the first step: nothing of what it did can be changed.
    script, _, on = scripted(hopper, monkeypatch, [DEAD] * 500)
    assert look_for(hopper, trace, no_watch, carry_on=4, live=0, still=True) == Search(None, 0)


def test_a_search_begins_where_the_last_one_about_there_stopped(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, _, on = scripted(hopper, monkeypatch, [DEAD] * 3)
    trace = [on] * 10
    first = look_for(hopper, trace, no_watch, carry_on=4, budget=3, still=True)
    listed = list(changes(trace, hopper.space))
    assert [moves for moves, _ in script.calls] == [c.apply(trace, 4) for c in listed[:3]]
    script, _, on = scripted(hopper, monkeypatch, [ALIVE])
    again = look_for(
        hopper, trace, no_watch, carry_on=4, budget=3, still=True, begin_at=first.stopped_at
    )
    # It begins at the place in the list the last one stopped at, not at the top
    # again: and the first play there keeps the life.
    last = len(trace) - 1
    spot = first.stopped_at[1]
    begun = next(c for c in listed if (HOLDS.index(c.hold), last - c.at) >= spot)
    assert spot != (0, 0) and begun != listed[0]
    assert again.way == Way(begun.apply(trace, 4), begun, tries=1, ground=0)


def test_a_play_that_fails_comes_out_as_it_is(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    from retro_ai_lab.core import EmulatorError

    scripted(hopper, monkeypatch, [DEAD, EmulatorError("gone")])
    with pytest.raises(EmulatorError, match="gone"):
        look_for(hopper, [hopper.move("right")] * 10, no_watch, still=True)


def test_nonsense_is_refused(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    script, _, on = scripted(hopper, monkeypatch, [ACROSS] * 10)
    trace = [on] * 10
    for bad in (1, 0, None, "yes"):
        with pytest.raises(ConfigurationError, match="still must be true or false"):
            look_for(hopper, trace, no_watch, still=bad)
    # At a standstill no life was lost: staying alive tells nothing there.
    with pytest.raises(ConfigurationError, match="still is for a search after a life lost"):
        look_for(hopper, trace, no_watch, still=True, stood=True)
    assert script.calls == []
    assert CARRY_ON == 20  # a try runs this far past where the life went, as before


# --------------------------------------------------------------- on the practice game itself


def test_on_the_practice_game_the_first_change_that_keeps_the_life_is_the_one_across(
    hopper: Hopper,
) -> None:
    """A small change is a detour, not a retreat: after it the try makes the moves the
    go made. Walking right into the gap, the first change that keeps the life is the
    jump that gets across, the one the usual search finds: it is taken a play sooner,
    with no look at the screen."""
    watch, mark, died = trial.a_go_that_fell_in(hopper)
    right = hopper.move("right")
    trace = [right] * died
    usual = look_for(hopper, trace, lambda: watch.trial(mark))
    taken = look_for(hopper, trace, no_watch, still=True)
    assert usual.way is not None and taken.way is not None
    assert taken.way.change == usual.way.change and taken.way.moves == usual.way.moves
    assert taken.tries == usual.tries - 1 and taken.way.ground == 0 < usual.way.ground
    out = hopper.play(list(taken.way.moves))
    assert not out.lost and hopper.game.state.lives == toy_hopper.START_LIVES
    hopper.adapter.load_state(SPOT)


# ------------------------------------------------------------------------------ the runner


def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """What the runner hands the search, each time it searches."""
    made: list[dict[str, Any]] = []
    real = episode.find_way

    def spy(*args: Any, **options: Any) -> Search:
        made.append(options)
        return real(*args, **options)

    monkeypatch.setattr(episode, "find_way", spy)
    return made


def stood_still(monkeypatch: pytest.MonkeyPatch, says: bool) -> None:
    monkeypatch.setattr(Ground, "still", property(lambda self: says))


def test_on_a_level_that_scrolls_the_search_is_the_one_it_was(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    made = calls(monkeypatch)
    r, _ = searching(lab)
    r.begin()
    with caplog.at_level(logging.INFO, logger=LOG):
        steps(r, DIED_AT)
    assert r.watch is not None and r.watch.ground.scrolls and not r.watch.ground.still
    assert [options["still"] for options in made] == [False]
    assert (r.ways_sought, r.ways_found) == (1, 1) and SAID not in caplog.text


def test_where_the_screen_stands_still_the_runner_says_so_to_the_search(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    made = calls(monkeypatch)
    stood_still(monkeypatch, True)
    r, _ = searching(lab, ways_per_go=2)
    r.begin()
    with caplog.at_level(logging.INFO, logger=LOG):
        steps(r, DIED_AT)
        assert [options["still"] for options in made] == [True]
        assert (r.ways_sought, r.ways_found) == (1, 1)
        # The way it took keeps the life and stays this side of the gap: walking on,
        # the life goes again and it searches again. The log says it once a go.
        for _ in range(200):
            if r.ways_sought == 2 or not r.step():
                break
    assert [options["still"] for options in made] == [True, True]
    assert caplog.text.count(SAID) == 1
    assert "a change that keeps the life is taken for the way past" in caplog.text


def test_the_next_go_says_it_again(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls(monkeypatch)
    stood_still(monkeypatch, True)
    r, _ = searching(lab)
    with caplog.at_level(logging.INFO, logger=LOG):
        r.begin()
        steps(r, DIED_AT)
        r.finish()
        lab.game.reset()
        r.begin()
        steps(r, DIED_AT)
    assert caplog.text.count(SAID) == 2


def test_at_a_standstill_it_is_never_said(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    made = calls(monkeypatch)
    stood_still(monkeypatch, True)
    r, _ = standing(lab)
    r.begin()
    for _ in range(STAND_AFTER + 40):
        if r.ways_stood:
            break
        hop_about(lab, r, NEAR, 1)
    assert r.ways_stood == 1
    # No life was lost there: staying alive tells nothing, and the search would refuse.
    assert [options["still"] for options in made if options["stood"]] == [False] * r.ways_stood


# ------------------------------------------------------------- a game of one screen


FLOOR = (64, 64, 64)
BLOCK = (240, 240, 240)
BLOCK_SIZE = 16
FALLS_FROM = 1300  # the frame the block starts down at, and every 256 frames after
FALL = 4  # pixels a frame
DYING_FOR = 12  # frames the game takes over a life


class OneScreen(ToyHopper):
    """A game of one screen in the practice hopper's memory. The player walks a floor
    as wide as the screen, 2 pixels a frame; nothing scrolls and nothing else answers
    the pad. From frame ``FALLS_FROM`` a block comes down the middle of the screen
    every 256 frames, and costs a life where it lands on the player."""

    MIDDLE = (SCREEN_WIDTH - BLOCK_SIZE) // 2

    def _block_y(self) -> int | None:
        since = self._u16(LAYOUT.clock) - FALLS_FROM
        if since < 0:
            return None
        y = (since % 256) * FALL
        return y if y < GROUND_Y else None

    def step(self, pressed: Iterable[str] = ()) -> None:
        held = frozenset(pressed)
        ram = self._ram
        ram[LAYOUT.input] = sum(toy_hopper._BUTTON_BIT[name] for name in held)
        if ram[LAYOUT.mode] != GAME_OVER:
            self._set_u16(LAYOUT.clock, (self._u16(LAYOUT.clock) + 1) & 0xFFFF)
        if ram[LAYOUT.mode] == DYING:
            ram[LAYOUT.dying] -= 1  # the block falls on meanwhile, and is gone after
            if not ram[LAYOUT.dying]:
                ram[LAYOUT.lives] -= 1
                ram[LAYOUT.mode] = PLAYING if ram[LAYOUT.lives] else GAME_OVER
        elif ram[LAYOUT.mode] == PLAYING:
            want = ("right" in held) - ("left" in held)
            x = max(0, min(SCREEN_WIDTH - PLAYER_W, ram[LAYOUT.x_low] + 2 * want))
            ram[LAYOUT.x_low] = x
            y = self._block_y()
            top = GROUND_Y - PLAYER_H
            if (
                y is not None
                and y + BLOCK_SIZE > top
                and x + PLAYER_W > self.MIDDLE
                and x < self.MIDDLE + BLOCK_SIZE
            ):
                ram[LAYOUT.mode], ram[LAYOUT.dying] = DYING, DYING_FOR
        self._set_u16(LAYOUT.frame, (self._u16(LAYOUT.frame) + 1) & 0xFFFF)
        ram[LAYOUT.timer] = (ram[LAYOUT.timer] + 1) & 0xFF
        self._write_noise()

    def render(self) -> Frame:
        ram = self._ram
        px = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        px[:] = toy_hopper.SKY
        px[GROUND_Y:, :] = FLOOR
        y = self._block_y()
        if y is not None and ram[LAYOUT.mode] == PLAYING:
            px[max(0, y) : y + BLOCK_SIZE, self.MIDDLE : self.MIDDLE + BLOCK_SIZE] = BLOCK
        x, top = ram[LAYOUT.x_low], GROUND_Y - PLAYER_H
        px[top : top + PLAYER_H, x : x + PLAYER_W] = toy_hopper.PLAYER
        px[top + 2 : top + 7, x + 2 : x + PLAYER_W - 2] = toy_hopper.PLAYER_HEAD
        return Frame(SCREEN_WIDTH, SCREEN_HEIGHT, 3, px.tobytes(), number=self._u16(LAYOUT.frame))


def under_the_block(lab: Lab, **kw: object) -> tuple[EpisodeRunner, OneScreen]:
    """A go on the game of one screen that walks its floor from end to end, comes to
    stand in the middle and stays there: under where the block comes down."""
    game = OneScreen(0)
    lab.adapter._game = game
    lab.game = game
    kw.setdefault("max_steps", 600)
    kw.setdefault("unstick_after", None)  # it stands for minutes: that is the test
    r, _ = searching(lab, **kw)
    right, left, stand = lab.move("right"), lab.move("left"), lab.move()
    lab.agent.script.extend([right] * 30 + [left] * 34 + [right] * 15 + [stand] * 600)
    r.begin()
    return r, game


LIFE_GOES = 340  # the step of such a go the life goes at


def test_the_stand_in_is_a_game_of_one_screen(lab: Lab) -> None:
    r, game = under_the_block(lab, ways_per_go=0)
    seen = set()
    for _ in range(79):
        assert r.step()
        seen.add(game.state.x)
    # It has been to both ends of the floor and stands under the block's way down.
    assert {0, SCREEN_WIDTH - PLAYER_W} <= seen
    assert OneScreen.MIDDLE - PLAYER_W < game.state.x < OneScreen.MIDDLE + BLOCK_SIZE
    steps(r, LIFE_GOES - 79 - 1)
    watch = r.watch
    assert watch is not None and not watch.ground.scrolls and watch.ground.still
    assert watch.looks >= STILL_LOOKS and DEFAULT_LOOK_EVERY == 2
    assert game.state.lives == START_LIVES
    assert r.step()
    assert game.state.lives == START_LIVES - 1  # and nothing went back for it


def test_on_a_game_of_one_screen_the_life_lost_is_gone_back_for_and_kept(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    made = calls(monkeypatch)
    r, game = under_the_block(lab)
    steps(r, LIFE_GOES - 1)
    assert made == [] and game.state.lives == START_LIVES
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()  # the life goes, and inside the same step it looks and goes back
    assert [options["still"] for options in made] == [True]
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    assert caplog.text.count(SAID) == 1 and "found a way past on try" in caplog.text
    # It plays the way: a step aside, and the block comes down beside it.
    steps(r, 60)
    # The game is past the frame the block landed at, with every life it began with.
    assert game.state.lives == START_LIVES and game.state.frame > 4 * LIFE_GOES
    x = game.state.x
    assert not OneScreen.MIDDLE - PLAYER_W < x < OneScreen.MIDDLE + BLOCK_SIZE
    assert r.ways_sought == 1


def test_asked_for_new_ground_there_no_search_could_find_a_way(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The fault this task is about, held: the floor is walked, so a step aside is on
    no new ground, and every try that keeps the life is turned down."""
    made = calls(monkeypatch)
    stood_still(monkeypatch, False)
    r, game = under_the_block(lab)
    with caplog.at_level(logging.INFO, logger=LOG):
        steps(r, LIFE_GOES)
    assert [options["still"] for options in made] == [False]
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 0, 0)
    assert game.state.lives == START_LIVES - 1
    assert "none got past" in caplog.text and SAID not in caplog.text
