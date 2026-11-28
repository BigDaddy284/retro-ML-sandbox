"""T-1073: a search that finds no way says what the changes it played came to.

In the user's log of 27 Nov, 34 searches made 17 to 99 tries each and "none got
past", and nothing more was said. Whether those changes all ran into the same loss,
lost the life a little later, or kept it and were not counted as getting anywhere
new (the screen's measure of new ground failing) are three different faults, and the
log could not tell them apart. A search for nothing now hands back a :class:`Tally`
of its changes, and :func:`tell` says it in the log.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.training import search
from retro_ai_lab.training.search import BUDGET, Known, Search, Tally, tell
from retro_ai_lab.training.trial import PAST, Outcome
from retro_ai_lab.training.ways import changes
from tests.unit.training.test_search import ACROSS, ALIVE, DEAD, NOWHERE, look_for
from tests.unit.training.test_search_known import Told, lost_at, told
from tests.unit.training.test_trial import FRAMES, LIVES, SPOT, Hopper, a_go_that_fell_in

told = told  # the scripted search of ten steps (pytest fixture)

LOG = "retro_ai_lab.training.search"


def test_changes_that_lose_the_life_no_later_or_later(told: Told) -> None:
    # Ten steps, the life went at the end of them: at step 10.
    out = told.look([lost_at(14), lost_at(10), DEAD, DEAD], budget=4)
    assert out == Search(None, 4)  # the tally is not part of what was found
    assert out.tally == Tally(sooner=3, later=1, latest=4)
    assert out.tally.changes == 4
    # The latest is the latest, not the last.
    out = told.look([lost_at(14), lost_at(11), lost_at(12)], budget=3)
    assert out.tally == Tally(later=3, latest=4)
    assert Tally(1, 2, 0, 3, 0, 4).changes == 10


def test_changes_that_keep_the_life(told: Told) -> None:
    # The first kept, looked at and nowhere new; the next lost; the last kept and the
    # tries up before it could be looked at.
    got = Outcome(steps=30, lost=False, ground=PAST - 1)
    out = told.look([ALIVE, got, DEAD, ALIVE], budget=4)
    assert [looked for _, looked in told.calls] == [False, True, False, False]
    assert out.way is None and out.tally == Tally(sooner=1, nowhere=1, ground=PAST - 1, unseen=1)
    assert out.tally.changes == 3
    # The most new ground of those looked at, not the last.
    less = Outcome(steps=30, lost=False, ground=0)
    out = told.look([ALIVE, got, ALIVE, less], budget=4)
    assert out.tally == Tally(nowhere=2, ground=PAST - 1)
    # The time up before the look, the same.
    ticks = iter([0.0, 1.0, 5.0, 5.0, 5.0, 5.0])
    out = told.look([ALIVE], seconds=5, clock=lambda: next(ticks))
    assert out.out_of_time and out.tally == Tally(unseen=1)


def test_a_change_that_ends_where_getting_nowhere_ended_is_nowhere_new(told: Told) -> None:
    # (Played blind it has no watch: what ground it says is not counted.)
    there = Outcome(steps=30, lost=False, ground=5, screen=b"the same screen")
    seen = Outcome(steps=30, lost=False, ground=1)
    # Kept and nowhere new; then another that ends on the same screen is not looked at.
    out = told.look([there, seen, there, DEAD], budget=4)
    assert [looked for _, looked in told.calls] == [False, True, False, False]
    assert out.tally == Tally(sooner=1, nowhere=2, ground=1)


def test_the_plain_way_is_counted_as_a_change(told: Told) -> None:
    right = told.right
    # Held, it loses the life two steps later than what it did; its half of the tries
    # is used up by that one play, and the rest are changes to what it did.
    out = told.look([lost_at(12), DEAD, DEAD], plain=right, budget=3)
    assert out.tally == Tally(sooner=2, later=1, latest=2)
    # Held, it keeps the life, with no try of its half left to look at it.
    out = told.look([ALIVE, DEAD], plain=right, budget=2)
    assert out.tally == Tally(sooner=1, unseen=1)
    # Looked at, it got nowhere new.
    out = told.look([ALIVE, NOWHERE, DEAD, DEAD], plain=right, budget=4)
    assert out.tally == Tally(sooner=2, nowhere=1, ground=NOWHERE.ground)
    # Its own changes lose the life against the step it lost it at, not the trace's:
    # held, it lost it at step 14, and a change to it that loses it at 13 is no later.
    out = told.look([lost_at(14), lost_at(13), DEAD, DEAD], plain=right, budget=4)
    assert out.tally == Tally(sooner=3, later=1, latest=4)


def test_second_changes_are_counted_and_the_look_at_the_first_is_not(told: Told) -> None:
    first = next(iter(changes(told.trace, told.hopper.space)))
    assert first.hold == 4
    far = lost_at(10 + first.hold + 5)  # later than waiting could have put it off
    stone = Outcome(steps=far.steps, lost=True, ground=PAST)  # looked at: a step on
    out = told.look([far, stone, DEAD, lost_at(25), ALIVE], deeper=True, budget=5)
    assert [looked for _, looked in told.calls] == [False, True, False, False, False]
    # The first change lost it 9 steps later than what it did; of the second changes
    # (against the 19 steps the first played), one no later, one 6 later, and one kept
    # with the tries up.
    assert out.tally == Tally(sooner=1, later=2, latest=9, unseen=1)


def test_the_known_change_is_counted_each_time_it_is_played(told: Told) -> None:
    first = next(iter(changes(told.trace, told.hopper.space)))
    known = Known(first.move, first.hold, 10 - first.at)
    out = told.look([DEAD, DEAD, DEAD], known=known, budget=3)
    assert told.calls[0] == told.calls[1]  # the lists come to it as well
    assert out.tally == Tally(sooner=3)
    out = told.look([ALIVE, NOWHERE, DEAD], known=Known(told.jump, 4, 3), budget=3)
    assert out.tally == Tally(sooner=1, nowhere=1, ground=NOWHERE.ground)
    out = told.look([ALIVE], known=Known(told.jump, 4, 3), budget=1)
    assert out.tally == Tally(unseen=1)


def test_a_search_that_finds_a_way_has_no_tally(told: Told) -> None:
    out = told.look([DEAD, ALIVE, ACROSS])
    assert out.way is not None and out.tally is None
    out = told.look([ALIVE, ACROSS], known=Known(told.jump, 4, 3))
    assert out.again and out.tally is None
    out = told.look([ALIVE, ACROSS], plain=told.right)
    assert out.way is not None and out.tally is None


def test_nothing_to_change_is_no_change(told: Told) -> None:
    out = told.look([], live=0)
    assert out == Search(None, 0) and out.tally == Tally()


def test_on_the_game_every_change_from_inside_the_fall_loses_the_life_no_later(
    hopper: Hopper,
) -> None:
    hopper.game.dying_frames = 200  # a long fall: forty steps before the count are in it
    watch, mark, died = a_go_that_fell_in(hopper)
    right = hopper.move("right")
    buttons = {b.name: b for b in hopper.adapter.buttons()}
    hopper.adapter.load_state(SPOT)
    hopper.adapter.step(FRAMES * (died - 40), pressed=[buttons["right"]])
    hopper.adapter.save_state("late")
    late = search.find_way(
        hopper.connection,
        "late",
        [right] * 40,
        hopper.space,
        frames_per_step=FRAMES,
        lives=[LIVES],
        watch=lambda: watch.trial(mark),
    )
    listed = len(list(changes([right] * 40, hopper.space)))
    assert late.way is None and late.tries == listed <= BUDGET
    assert late.tally == Tally(sooner=listed)


def test_on_the_game_a_short_search_at_the_edge_says_what_it_tried(hopper: Hopper) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    trace = [hopper.move("right")] * died
    out = look_for(hopper, trace, lambda: watch.trial(mark), budget=3)
    assert out.way is None and out.tally is not None
    # Every play is a change, or the look at one that kept the life.
    t = out.tally
    assert t.changes + t.nowhere == out.tries or t.changes == out.tries
    assert t.changes >= 2


# ------------------------------------------------------------ the line in the log


def said(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if rec.name == LOG]


def test_the_log_says_it(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOG)
    tell(Search(None, 9, tally=Tally(sooner=3, later=2, latest=7, nowhere=1, ground=1, unseen=2)))
    assert said(caplog) == [
        "The 8 changes it played came to this: 3 lost the life no later than it went, 2 "
        "lost it later (the latest 7 steps later), 1 kept it and got nowhere new (at most "
        "1 step of new ground, where 2 is getting past), and 2 kept it and were not "
        "looked at, the tries or the time being up."
    ]
    assert PAST == 2
    caplog.clear()
    tell(Search(None, 1, tally=Tally(later=1, latest=1, unseen=0)))
    assert said(caplog) == [
        "The 1 change it played came to this: 0 lost the life no later than it went, 1 "
        "lost it later (the latest 1 step later), 0 kept it and got nowhere new (at most "
        "0 steps of new ground, where 2 is getting past), and 0 kept it and were not "
        "looked at, the tries or the time being up."
    ]
    caplog.clear()
    tell(Search(None, 1, tally=Tally(unseen=1)))
    assert said(caplog)[0].endswith(
        "and 1 kept it and was not looked at, the tries or the time being up."
    )


def test_the_log_says_when_there_was_nothing_to_change(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOG)
    tell(Search(None, 0, tally=Tally()))
    assert said(caplog) == ["It played no change: there was nothing of what it did to change."]


def test_the_log_says_nothing_of_a_way_or_of_a_search_with_no_tally(
    told: Told, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=LOG)
    tell(Search(None, 5))
    found = told.look([DEAD, ALIVE, ACROSS])
    tell(found)
    assert found.way is not None
    tell(Search(found.way, 3, tally=Tally(sooner=1)))
    assert said(caplog) == []
    # And the search itself says nothing: the line is the caller's to ask for.
    told.look([DEAD] * 3, budget=3)
    assert said(caplog) == []
