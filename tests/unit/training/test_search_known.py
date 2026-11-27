"""T-1072: a search tries first the change that got past about here before.

The user's log of 27 Nov: in go 129 a search found the way past a place, a jump of
four steps "starting 54 steps before that would lose the life", the last step its
list could place a change at. The three goes after came to the same place with the
same moves behind them, and their searches made 77, 75 and 63 tries there for
nothing. The game had been asked anew from when on the loss was settled and said 54
steps before and not 53: the list ended a step earlier, went on two steps at a
time, and never came to the jump that had got past.

A search can now be told that change, counted from the loss it was made before, and
plays it before its lists.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import search
from retro_ai_lab.training.search import CARRY_ON, TOP, Known, Search, Way
from retro_ai_lab.training.trial import PAST, Outcome
from retro_ai_lab.training.ways import Change, changes
from tests.unit.training.test_search import ACROSS, ALIVE, DEAD, NOWHERE, Script, look_for
from tests.unit.training.test_trial import Hopper, a_go_that_fell_in

BEGIN = ((0, 2), (1, 4))


def lost_at(step: int) -> Outcome:
    return Outcome(steps=step, lost=True, ground=0)


class Told:
    """A search of ten steps (six to the right, four to the left) whose plays are
    scripted, and the moves it knows."""

    def __init__(self, hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
        self.hopper, self.monkeypatch = hopper, monkeypatch
        self.right, self.left = hopper.move("right"), hopper.move("left")
        self.jump = hopper.move("a", "right")
        self.trace = [self.right] * 6 + [self.left] * 4
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def look(self, outcomes: Sequence[Outcome], **kw: object) -> Search:
        script = Script(list(outcomes))
        self.monkeypatch.setattr(search, "play", script)
        out = look_for(self.hopper, self.trace, object, **kw)
        self.calls = script.calls
        assert out.tries == len(script.calls)
        return out

    def listed(self, base: Sequence[int], upto: int | None = None) -> list[tuple[int, ...]]:
        return [c.apply(base, CARRY_ON) for c in changes(base, self.hopper.space, upto=upto)]


@pytest.fixture
def told(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> Told:
    return Told(hopper, monkeypatch)


def test_a_known_change_to_what_it_did_is_played_before_anything_else(told: Told) -> None:
    # Three steps before the life went, which is after the last chance (six steps of
    # the ten): no list places a change there.
    known = Known(told.jump, 4, 3)
    moves = Change(7, told.jump, 4).apply(told.trace, CARRY_ON)
    assert moves not in told.listed(told.trace, 6)
    for plain in (None, told.right):
        out = told.look([ALIVE, ACROSS], known=known, live=6, plain=plain, begin_at=BEGIN)
        # Played blind and then looked at, like any change: two plays, and the way.
        assert told.calls == [(moves, False), (moves, True)]
        assert out.way == Way(moves, Change(7, told.jump, 4), 2, ACROSS.ground)
        assert out.again and out.tries == 2 and not out.out_of_time
        # The lists are left where they were.
        assert out.stopped_at == BEGIN and out.deeper_at == TOP and not out.lapped
    # A search that is told nothing, or finds its way in a list, does not say so.
    out = told.look([ACROSS, ACROSS])
    assert out.way is not None and not out.again


def test_a_known_change_to_the_plain_way_is_played_once_that_has_lost_the_life(
    told: Told,
) -> None:
    # Made to the plain way, three steps before the loss holding right comes to.
    known = Known(told.jump, 4, 3, plain=told.right)
    held = (told.right,) * (10 + CARRY_ON)
    moves = Change(9, told.jump, 4).apply((told.right,) * 12, CARRY_ON)
    # (With six of ten steps before the last chance the plain way's list ends at its
    # step 7, four before its loss at step 12.)
    assert moves not in told.listed((told.right,) * 12, 8)
    out = told.look(
        [lost_at(12), ALIVE, ACROSS], known=known, live=6, plain=told.right, begin_at=BEGIN
    )
    assert told.calls == [(held, False), (moves, False), (moves, True)]
    assert out.way == Way(
        moves, Change(9, told.jump, 4), 3, ACROSS.ground, plain=told.right, lost_at=12
    )
    assert out.again and out.stopped_at == BEGIN
    # It is counted from the loss the plain way comes to this time: two steps later,
    # two steps later.
    out = told.look([lost_at(14), ALIVE, ACROSS], known=known, live=6, plain=told.right)
    later = Change(11, told.jump, 4).apply((told.right,) * 14, CARRY_ON)
    assert told.calls[1:] == [(later, False), (later, True)]
    assert out.way is not None and (out.way.change.at, out.way.lost_at) == (11, 14)


def test_it_is_only_for_the_plain_way_it_was_made_to(told: Told) -> None:
    dead = [DEAD] * 300

    def plays(**kw: object) -> list[tuple[tuple[int, ...], bool]]:
        out = told.look([lost_at(12), *dead], live=6, **kw)
        assert out.way is None and not out.again
        return told.calls

    without = plays(plain=told.right)
    # Made to another move held: not played.
    assert plays(plain=told.right, known=Known(told.jump, 4, 3, plain=told.left)) == without
    # No plain way in this search: not played.
    assert plays(known=Known(told.jump, 4, 3, plain=told.right)) == plays()
    # A plain way that keeps the life has no loss to count from: not played.
    out = told.look(
        [ALIVE, NOWHERE, *dead], live=6, plain=told.right, known=Known(told.jump, 4, 3, told.right)
    )
    held = (told.right,) * (10 + CARRY_ON)
    assert told.calls[:2] == [(held, False), (held, True)]
    assert told.calls[2:] == [(moves, False) for moves in told.listed(told.trace, 6)]
    assert out.way is None


def test_one_that_loses_the_life_leaves_the_search_as_it_would_have_been(told: Told) -> None:
    first = next(iter(changes(told.trace, told.hopper.space, upto=6)))
    known = Known(first.move, first.hold, 10 - first.at)
    listed = told.listed(told.trace, 6)
    out = told.look([DEAD] * 300, known=known, live=6)
    # The lists come to the same change and play it again: it is not held against them.
    assert listed[0] == first.apply(told.trace, CARRY_ON)
    assert told.calls == [(listed[0], False), *((moves, False) for moves in listed)]
    assert out == Search(None, 1 + len(listed)) and not out.again
    # Where the single changes have been gone through, a change that loses the life
    # later is searched on from (T-1059). Not from the known try, which only asks
    # whether the change gets past again: from the list's own play of it.
    far = Outcome(steps=10 + first.hold + 4, lost=True, ground=0)
    out = told.look([far, far, DEAD, *[DEAD] * 300], known=known, live=6, deeper=True)
    assert told.calls[:3] == [(listed[0], False), (listed[0], False), (listed[0], True)]
    assert out.way is None


def test_one_that_keeps_the_life_and_gets_nowhere_is_not_played_again(told: Told) -> None:
    first = next(iter(changes(told.trace, told.hopper.space, upto=6)))
    known = Known(first.move, first.hold, 10 - first.at)
    listed = told.listed(told.trace, 6)
    out = told.look([ALIVE, NOWHERE, *[DEAD] * 300], known=known, live=6)
    assert told.calls == [(listed[0], False), (listed[0], True), *((m, False) for m in listed[1:])]
    assert out == Search(None, 1 + len(listed)) and not out.again


def test_it_is_placed_within_what_it_did(told: Told) -> None:
    # Ten steps before the loss is the first step: played.
    told.look([DEAD] * 300, known=Known(told.jump, 4, 10))
    assert told.calls[0] == (Change(0, told.jump, 4).apply(told.trace, CARRY_ON), False)
    assert len(told.calls) == 1 + len(told.listed(told.trace))
    # Eleven is before the save the search plays from: not played.
    told.look([DEAD] * 300, known=Known(told.jump, 4, 11))
    assert told.calls == [(moves, False) for moves in told.listed(told.trace)]
    # A hold that reaches past the loss is cut there, like any change's.
    told.look([DEAD] * 300, known=Known(told.jump, 16, 2))
    assert told.calls[0][0] == (*told.trace[:8], told.jump, told.jump, *[told.right] * CARRY_ON)


def test_the_tries_and_the_time_hold_for_it(told: Told) -> None:
    known = Known(told.jump, 4, 3)
    moves = Change(7, told.jump, 4).apply(told.trace, CARRY_ON)
    # One try: the blind play, and none left to look at it. Nothing is found.
    out = told.look([ALIVE], known=known, budget=1)
    assert told.calls == [(moves, False)] and out == Search(None, 1) and not out.again
    # Two: it is the way.
    out = told.look([ALIVE, ACROSS], known=known, budget=2)
    assert out.way is not None and out.again
    # Time up before the first play: none is made.
    ticks = iter([0.0, 5.0, 5.0, 5.0, 5.0])
    out = told.look([], known=known, seconds=5, clock=lambda: next(ticks))
    assert told.calls == [] and out.way is None and out.out_of_time
    # Time up after the blind play: it is not looked at.
    ticks = iter([0.0, 1.0, 5.0, 5.0, 5.0, 5.0])
    out = told.look([ALIVE], known=known, seconds=5, clock=lambda: next(ticks))
    assert told.calls == [(moves, False)] and out.way is None and out.out_of_time


def test_on_a_screen_that_stands_still_the_life_kept_is_the_way(told: Told) -> None:
    known = Known(told.jump, 4, 3)
    moves = Change(7, told.jump, 4).apply(told.trace, CARRY_ON)
    out = told.look([ALIVE], known=known, still=True)
    assert told.calls == [(moves, False)]
    assert out.way == Way(moves, Change(7, told.jump, 4), 1, 0) and out.again


@pytest.mark.parametrize(
    ("known", "more", "said"),
    [
        ((4, 4, 3), {}, "known is a change that got past before, or nothing."),
        (Change(3, 1, 4), {}, "known is a change that got past before, or nothing."),
        (
            Known(1, 4, 3),
            {"stood": True},
            "known is for a search after a life lost: at a standstill none was.",
        ),
        (Known(1, 0, 3), {}, "A known change's hold must be a whole number, 1 or more."),
        (Known(1, True, 3), {}, "A known change's hold must be a whole number, 1 or more."),
        (
            Known(1, 4, 0),
            {},
            "A known change's steps before the loss must be a whole number, 1 or more.",
        ),
        (
            Known(1, 4, 2.0),  # type: ignore[arg-type]
            {},
            "A known change's steps before the loss must be a whole number, 1 or more.",
        ),
    ],
)
def test_what_it_is_told_is_checked(
    told: Told, known: object, more: dict[str, object], said: str
) -> None:
    with pytest.raises(ConfigurationError) as refused:
        told.look([], known=known, **more)
    assert str(refused.value) == said
    assert told.calls == []


def test_a_move_the_controls_do_not_have_is_refused(told: Told) -> None:
    for known in (Known(999, 4, 3), Known(told.jump, 4, 3, plain=999)):
        with pytest.raises(ConfigurationError):
            told.look([], known=known)
        assert told.calls == []


def test_on_the_game_what_got_across_gets_across_again_at_once(hopper: Hopper) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    right = hopper.move("right")
    trace = [right] * died

    def trial() -> object:
        return watch.trial(mark)

    for plain in (None, right):
        found = look_for(hopper, trace, trial, plain=plain)
        way = found.way
        assert way is not None and not found.again and found.tries > 3
        base = died if plain is None else way.lost_at
        assert base is not None
        known = Known(way.change.move, way.change.hold, base - way.change.at, plain)
        again = look_for(hopper, trace, trial, plain=plain, known=known)
        assert again.way is not None and again.again
        assert again.way.moves == way.moves and again.way.change == way.change
        assert (again.way.plain, again.way.lost_at) == (way.plain, way.lost_at)
        # The blind play and the look, and the plain way before them where there is one.
        assert again.tries == again.way.tries == (2 if plain is None else 3) < found.tries
        assert again.way.ground >= PAST
        # The user's goes 130 to 133: the game says the loss was settled a little
        # earlier than the change that got past. No list places a change there, and
        # the known one is played all the same.
        early = way.change.at - 1
        told = look_for(hopper, trace, trial, plain=plain, known=known, live=early)
        assert told.way is not None and told.again and told.way.moves == way.moves
        untold = look_for(hopper, trace, trial, plain=plain, live=early)
        assert not untold.again
        assert untold.way is None or untold.way.change.at < way.change.at
        assert untold.tries > told.tries
        # The game agrees: alive, with its lives, past the gap.
        assert not hopper.play(list(told.way.moves)).lost
        assert hopper.game.state.lives == toy_hopper.START_LIVES


def test_on_the_game_it_gets_across_from_a_go_that_came_there_later(hopper: Hopper) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    right, wait = hopper.move("right"), hopper.move()
    found = look_for(hopper, [right] * died, lambda: watch.trial(mark))
    assert found.way is not None
    change = found.way.change
    known = Known(change.move, change.hold, died - change.at)
    # A go that stood for five steps first comes to the edge five steps later, and
    # loses the life five steps later. Counted from the loss the change is at the edge.
    late = [wait] * 5 + [right] * died
    assert hopper.play(late).steps == died + 5
    again = look_for(hopper, late, lambda: watch.trial(mark), known=known)
    assert again.way is not None and again.again and again.tries == 2
    assert again.way.change == Change(change.at + 5, change.move, change.hold)
    assert not hopper.play(list(again.way.moves)).lost
