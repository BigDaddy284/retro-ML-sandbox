"""T-1042: a search that found nothing says where it stopped, and the next begins there."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training import search
from retro_ai_lab.training.search import CARRY_ON, TOP, Search, Spot, Way
from retro_ai_lab.training.trial import Outcome
from retro_ai_lab.training.ways import HOLDS, Change, changes
from tests.unit.training import test_trial as trial
from tests.unit.training.test_search import ACROSS, ALIVE, DEAD, Clock, Script, look_for
from tests.unit.training.test_trial import Hopper, a_go_that_fell_in

hopper = trial.hopper  # the practice game behind a connection (pytest fixture)

Play = tuple[tuple[int, ...], bool]  # the moves of a play, and whether the screen was looked at


def noted(monkeypatch: pytest.MonkeyPatch) -> list[Play]:
    """Every play the search makes from here on, as it is made."""
    plays: list[Play] = []
    real = search.play

    def noting(*args: object, **kw: object) -> Outcome:
        plays.append((tuple(args[2]), kw.get("watch") is not None))  # type: ignore[arg-type]
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(search, "play", noting)
    return plays


def spot_of(change: Change, base: Sequence[int], upto: int | None = None) -> Spot:
    """Where ``change`` is in the list of changes to ``base``, as a search says it."""
    last = (len(base) if upto is None else min(len(base), upto)) - 1
    return HOLDS.index(change.hold), last - change.at


def scripted(monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[Outcome]) -> Script:
    script = Script(outcomes)
    monkeypatch.setattr(search, "play", script)
    return script


# ------------------------------------------------------------ on the practice game's first gap

WAY = 16  # the first change in the list that gets over the first gap of this level


def test_with_nothing_to_begin_at_it_is_the_search_it_was(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert TOP == ((0, 0), (0, 0))
    watch, mark, died = a_go_that_fell_in(hopper)
    trace = [hopper.move("right")] * died
    listed = list(changes(trace, hopper.space))
    plays = noted(monkeypatch)
    plain = look_for(hopper, trace, lambda: watch.trial(mark))
    as_it_was = list(plays)
    # The list from its top: each change before the way blind, the way blind and then
    # under the screen.
    blind = [(change.apply(trace, CARRY_ON), False) for change in listed[: WAY + 1]]
    assert as_it_was == [*blind, (listed[WAY].apply(trace, CARRY_ON), True)]
    assert plain.way is not None and plain.way.change == listed[WAY] and plain.tries == WAY + 2
    plays.clear()
    from_the_top = look_for(hopper, trace, lambda: watch.trial(mark), begin_at=TOP)
    assert from_the_top == plain and plays == as_it_was
    # It says where the change that got past is: held 4 steps, 16 steps before the
    # last step of what it did. The plain way's list was not gone into.
    assert plain.stopped_at == from_the_top.stopped_at == ((0, 0), (0, 16))
    assert spot_of(listed[WAY], trace) == (0, 16)
    # Where it stopped is no part of what it found.
    assert Search(None, 3, stopped_at=((1, 2), (2, 4))) == Search(None, 3)


def test_a_search_cut_short_says_where_and_the_next_carries_on(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    trace = [hopper.move("right")] * died
    listed = list(changes(trace, hopper.space))
    plays = noted(monkeypatch)

    def look(**kw: object) -> tuple[Search, list[Play]]:
        plays.clear()
        found = look_for(hopper, trace, lambda: watch.trial(mark), **kw)
        return found, list(plays)

    whole, _ = look()
    # Ten tries: the ten changes at the first five places, all a life lost. It stopped
    # before the first change at the sixth place, 10 steps before the last step.
    first, first_plays = look(budget=10)
    assert first == Search(None, 10) and not first.out_of_time
    assert first_plays == [(change.apply(trace, CARRY_ON), False) for change in listed[:10]]
    assert first.stopped_at == ((0, 0), (0, 10)) == ((0, 0), spot_of(listed[10], trace))
    # From the top again it is the same search, to the play.
    again, again_plays = look(budget=10)
    assert again == first and again_plays == first_plays and again.stopped_at == first.stopped_at
    # Begun where the first stopped, with no more tries than it had: none of its
    # plays again, the changes it did not get to, and the way.
    second, second_plays = look(budget=10, begin_at=first.stopped_at)
    assert second.way is not None and second.way.change == listed[WAY]
    assert second.way == Way(whole.way.moves, listed[WAY], 8, whole.way.ground)  # type: ignore[union-attr]
    assert second.tries == 8 < whole.tries == 18
    assert not {moves for moves, _ in first_plays} & {moves for moves, _ in second_plays}
    assert second_plays == [
        *[(change.apply(trace, CARRY_ON), False) for change in listed[10 : WAY + 1]],
        (listed[WAY].apply(trace, CARRY_ON), True),
    ]
    assert second.stopped_at == ((0, 0), (0, 16))
    # Cut off between the two changes at one place, the next begins that place again:
    # one play made twice, and no more.
    first, first_plays = look(budget=15)
    assert first == Search(None, 15) and first.stopped_at == ((0, 0), (0, 14))
    assert spot_of(listed[14], trace) == spot_of(listed[15], trace) == (0, 14)
    second, second_plays = look(budget=15, begin_at=first.stopped_at)
    assert second.way is not None and second.way.change == listed[WAY] and second.tries == 4
    twice = {moves for moves, _ in first_plays} & {moves for moves, _ in second_plays}
    assert twice == {listed[14].apply(trace, CARRY_ON)}


def test_a_search_begun_further_down_the_list_tries_what_is_there(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    trace = [hopper.move("right")] * died
    listed = list(changes(trace, hopper.space))
    plays = noted(monkeypatch)
    # Begun at the changes held for 8 steps, and for 16: the first of those that gets
    # over is the way, and no change held for 4 steps is played.
    for number, hold in ((1, 8), (2, 16)):
        plays.clear()
        found = look_for(hopper, trace, lambda: watch.trial(mark), begin_at=((0, 0), (number, 0)))
        assert found.way is not None and found.way.change.hold == hold
        of_that_hold = [change for change in listed if change.hold == hold]
        at = of_that_hold.index(found.way.change)
        assert [moves for moves, _ in plays[:-1]] == [
            change.apply(trace, CARRY_ON) for change in of_that_hold[: at + 1]
        ]
        assert found.stopped_at == ((0, 0), spot_of(found.way.change, trace))
        assert found.stopped_at[1][0] == number
    # A place to begin past the end of the list is the list's start.
    plays.clear()
    found = look_for(hopper, trace, lambda: watch.trial(mark), begin_at=((0, 0), (9, 9)))
    assert found.way is not None and found.way.change == listed[WAY] and found.tries == WAY + 2
    found = look_for(hopper, trace, lambda: watch.trial(mark), begin_at=((0, 0), (2, 400)))
    assert found.way is not None and found.way.change == listed[WAY]


def test_time_running_out_is_carried_like_tries(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    trace = [hopper.move("right")] * died
    listed = list(changes(trace, hopper.space))
    clock = Clock(monkeypatch)  # a second passes with every play
    first = look_for(hopper, trace, lambda: watch.trial(mark), seconds=10, clock=clock)
    assert first == Search(None, 10, out_of_time=True)
    assert first.stopped_at == ((0, 0), (0, 10))
    clock.plays.clear()
    second = look_for(
        hopper,
        trace,
        lambda: watch.trial(mark),
        seconds=10,
        clock=clock,
        begin_at=first.stopped_at,
    )
    assert second.way is not None and second.way.change == listed[WAY]
    assert second.tries == 8 and not second.out_of_time
    assert clock.plays == [False] * 7 + [True]
    # Out of time between the blind play of a change that stayed alive and the look at
    # it: that change is where it stopped, and the next search begins with it.
    clock.plays.clear()
    cut = look_for(
        hopper, trace, lambda: watch.trial(mark), seconds=7, clock=clock, begin_at=((0, 0), (0, 10))
    )
    assert cut == Search(None, 7, out_of_time=True) and cut.stopped_at == ((0, 0), (0, 16))
    assert clock.plays == [False] * 7


# ------------------------------------------------------------ the rule, with scripted plays


def test_at_the_end_of_a_list_it_comes_round_to_the_start(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    right = hopper.move("right")
    trace = [right] * 10
    listed = list(changes(trace, hopper.space))
    size = len(listed)
    assert size > 12 and len({spot_of(change, trace) for change in listed}) < size
    # Begun two changes before the list's end (the last place of the longest hold):
    # those two, then the list's first, which is the way.
    begin = spot_of(listed[-2], trace)
    assert begin == spot_of(listed[-1], trace) != spot_of(listed[-3], trace)
    script = scripted(monkeypatch, [DEAD, DEAD, ALIVE, ACROSS])
    found = look_for(hopper, trace, object, carry_on=4, begin_at=((0, 0), begin))
    assert found == Search(Way(listed[0].apply(trace, 4), listed[0], tries=4, ground=3), 4)
    assert script.calls == [
        (listed[-2].apply(trace, 4), False),
        (listed[-1].apply(trace, 4), False),
        (listed[0].apply(trace, 4), False),
        (listed[0].apply(trace, 4), True),
    ]
    assert found.stopped_at == ((0, 0), (0, 0))
    # A whole list for nothing, begun in its middle: every change once, in the list's
    # order from there and round, and it stopped where it began.
    middle = size // 2
    begin = spot_of(listed[middle], trace)
    start = next(i for i, change in enumerate(listed) if spot_of(change, trace) == begin)
    script = scripted(monkeypatch, [DEAD] * size)
    out = look_for(hopper, trace, object, carry_on=4, budget=size + 50, begin_at=((3, 1), begin))
    assert out == Search(None, size) and not out.out_of_time
    assert [moves for moves, _ in script.calls] == [
        change.apply(trace, 4) for change in listed[start:] + listed[:start]
    ]
    assert len(set(script.calls)) == size and script.outcomes == []
    assert out.stopped_at == ((3, 1), begin)  # the plain way's list was not gone into
    # And from the top, the top.
    script = scripted(monkeypatch, [DEAD] * size)
    out = look_for(hopper, trace, object, carry_on=4, budget=size + 50)
    assert out == Search(None, size) and out.stopped_at == TOP
    # With tries for all but the last change of the round, that one is where it stopped.
    script = scripted(monkeypatch, [DEAD] * size)
    out = look_for(hopper, trace, object, carry_on=4, budget=size - 1, begin_at=((0, 0), begin))
    assert out == Search(None, size - 1)
    assert out.stopped_at == ((0, 0), spot_of(listed[start - 1], trace))


def test_the_two_lists_are_carried_apart(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    back, on = hopper.move("left"), hopper.move("right")
    trace = [back] * 10
    base = (on,) * 6  # the plain way loses the life at its sixth step
    plain = list(changes(base, hopper.space))
    did = list(changes(trace, hopper.space))

    def lost_at(step: int) -> Outcome:
        return Outcome(steps=step, lost=True, ground=0)

    def played(listed: list[Change], of: Sequence[int]) -> list[Play]:
        return [(change.apply(of, 4), False) for change in listed]

    held: Play = ((on,) * 14, False)
    # Eight tries: the plain way and three of its changes (half the tries), then four
    # changes to what it did.
    script = scripted(monkeypatch, [lost_at(6)] + [DEAD] * 7)
    first = look_for(hopper, trace, object, carry_on=4, plain=on, budget=8)
    assert first == Search(None, 8)
    assert script.calls == [held, *played(plain[:3], base), *played(did[:4], trace)]
    assert first.stopped_at == (spot_of(plain[3], base), spot_of(did[4], trace))
    assert first.stopped_at == ((0, 2), (0, 4))
    # The next: the plain way once more (where it loses the life is not kept), then
    # each list from where its own stopped.
    script = scripted(monkeypatch, [lost_at(6)] + [DEAD] * 7)
    second = look_for(
        hopper, trace, object, carry_on=4, plain=on, budget=8, begin_at=first.stopped_at
    )
    assert second == Search(None, 8)
    assert script.calls == [held, *played(plain[2:5], base), *played(did[4:8], trace)]
    assert second.stopped_at == (spot_of(plain[5], base), spot_of(did[8], trace))
    # One list begun further down leaves the other at its top, either way round.
    script = scripted(monkeypatch, [lost_at(6)] + [DEAD] * 7)
    look_for(hopper, trace, object, carry_on=4, plain=on, budget=8, begin_at=((0, 0), (1, 0)))
    eights = [change for change in did if change.hold == 8]
    assert script.calls == [held, *played(plain[:3], base), *played(eights[:4], trace)]
    script = scripted(monkeypatch, [lost_at(6)] + [DEAD] * 7)
    look_for(hopper, trace, object, carry_on=4, plain=on, budget=8, begin_at=((1, 0), (0, 0)))
    # (The plain way is six steps long: two changes held for 8 steps are no other moves
    # than one held for 4, and after them the list comes round to its start.)
    eight = next(i for i, change in enumerate(plain) if change.hold == 8)
    assert [change.hold for change in plain[eight:]] == [8, 8]
    round_it = plain[eight:] + plain[:eight]
    assert script.calls == [held, *played(round_it[:3], base), *played(did[:4], trace)]
    # A way among the plain way's changes: where it is, and the other list as it was.
    script = scripted(monkeypatch, [lost_at(6), DEAD, ALIVE, ACROSS])
    found = look_for(hopper, trace, object, carry_on=4, plain=on, begin_at=((0, 2), (1, 6)))
    assert found.way is not None and found.way.change == plain[3] and found.way.plain == on
    assert found.stopped_at == (spot_of(plain[3], base), (1, 6))
    # A plain way that gets on with nothing changed went into neither list.
    script = scripted(monkeypatch, [ALIVE, ACROSS])
    found = look_for(hopper, trace, object, carry_on=4, plain=on, begin_at=((0, 2), (1, 6)))
    assert found.way is not None and found.way.change == Change(0, on, 10)
    assert found.stopped_at == ((0, 2), (1, 6))
    # Out of time (a second passes with every play): the plain way's list has half of
    # it (T-1043) and what it did the rest, and it says where each of them stopped.
    script = scripted(monkeypatch, [lost_at(6)] + [DEAD] * 7)
    out = look_for(
        hopper,
        trace,
        object,
        carry_on=4,
        plain=on,
        seconds=4,
        clock=lambda: float(len(script.calls)),
        begin_at=((0, 2), (1, 6)),
    )
    assert out == Search(None, 4, out_of_time=True)
    from_there = [change for change in did if spot_of(change, trace) >= (1, 6)]
    assert script.calls == [held, *played(plain[2:3], base), *played(from_there[:2], trace)]
    assert out.stopped_at == (spot_of(plain[3], base), spot_of(from_there[2], trace))
    assert out.stopped_at == ((0, 2), (1, 8))


def test_a_place_in_the_list_means_the_same_in_another_go(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    right = hopper.move("right")
    short, long = [right] * 10, [right] * 16  # the next go's saved moment lay 6 steps earlier
    listed = list(changes(short, hopper.space))
    script = scripted(monkeypatch, [DEAD] * 12)
    first = look_for(hopper, short, object, carry_on=4, budget=12)
    stopped = listed[12]
    assert stopped.hold == 8 and first.stopped_at == ((0, 0), spot_of(stopped, short))
    before_the_end = len(short) - stopped.at
    # In the longer trace it begins with the same hold, as many steps before the end:
    # the same change to the same stretch of the game. It is further down that list;
    # the twelfth of that list is a change held for 4 steps, somewhere else.
    script = scripted(monkeypatch, [ALIVE, ACROSS])
    second = look_for(hopper, long, object, carry_on=4, begin_at=first.stopped_at)
    assert second.way is not None
    change = second.way.change
    assert (change.hold, len(long) - change.at) == (stopped.hold, before_the_end)
    assert script.calls[0] == (change.apply(long, 4), False)
    of_the_long = list(changes(long, hopper.space))
    assert of_the_long.index(change) == 18 and of_the_long[12].hold == 4
    assert len(long) - of_the_long[12].at != before_the_end
    # With the loss settled before the end (T-1031) the steps are counted from the last
    # step a change may be placed at, in either go.
    listed = list(changes(short, hopper.space, upto=6))
    script = scripted(monkeypatch, [DEAD] * 3)
    first = look_for(hopper, short, object, carry_on=4, budget=3, live=6)
    stopped = listed[3]
    assert first.stopped_at == ((0, 0), spot_of(stopped, short, 6)) == ((0, 0), (0, 5 - stopped.at))
    script = scripted(monkeypatch, [ALIVE, ACROSS])
    second = look_for(hopper, long, object, carry_on=4, live=12, begin_at=first.stopped_at)
    assert second.way is not None
    change = second.way.change
    assert (change.hold, 12 - change.at) == (stopped.hold, 6 - stopped.at)
    # Nothing of what it did may be changed (the loss was settled before its first
    # step): the list is not gone into, and where to begin is as it was handed in.
    script = scripted(monkeypatch, [])
    out = look_for(hopper, short, object, carry_on=4, live=0, begin_at=((2, 3), (1, 4)))
    assert out == Search(None, 0) and out.stopped_at == ((2, 3), (1, 4))


def test_what_is_refused(hopper: Hopper) -> None:
    trace = [hopper.move("right")] * 10
    bad: tuple[object, ...] = (
        None,
        (0, 0),
        ((0, 0),),
        ((0, 0), (0, 0), (0, 0)),
        [(0, 0), (0, 0)],
        ((0, 0), (0,)),
        ((0, 0), [0, 0]),
        ((0, -1), (0, 0)),
        ((0, 0), (1.0, 0)),
        ((True, 0), (0, 0)),
        ((0, 0), (0, "4")),
    )
    for begin_at in bad:
        with pytest.raises(ConfigurationError, match="begin_at is two pairs"):
            look_for(hopper, trace, object, begin_at=begin_at)
    # A search at a standstill goes through its list forwards from the start.
    with pytest.raises(ConfigurationError, match="at a standstill begins at the top"):
        look_for(hopper, trace, object, stood=True, begin_at=((0, 0), (0, 2)))
    with pytest.raises(ConfigurationError, match="begin_at is two pairs"):
        look_for(hopper, trace, object, stood=True, begin_at=((0, 0), (0, -2)))
