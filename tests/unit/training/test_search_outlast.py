"""T-1019: a way a search finds must be one the AI can live on from."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.training import search
from retro_ai_lab.training.episode import NOW_SLOT
from retro_ai_lab.training.moments import LEAD
from retro_ai_lab.training.search import (
    CUT_BY,
    CUT_WITHIN,
    END_SLOT,
    OUTLAST,
    Search,
    Way,
    find_way,
)
from retro_ai_lab.training.service import BACK_SLOTS, CHECKPOINT_SLOT, PRACTISE_SLOT
from retro_ai_lab.training.trial import Outcome
from retro_ai_lab.training.ways import Change, changes
from tests.unit.training import test_trial as trial
from tests.unit.training.test_search import look_for
from tests.unit.training.test_trial import FRAMES, LIVES, SPOT, Hopper

hopper = trial.hopper  # the practice game behind a connection (pytest fixture)

Moves = tuple[int, ...]
K = 7  # the steps a move is held for in these tests


class Course:
    """Stands in for ``search.play``. A try from the saved spot stays alive unless
    its moves are among ``dead``, and under the watch it has got past once it has
    ``past`` moves. From the save of a try's end, a move held on keeps the life
    where ``keeps`` says so of the moves the game was last played to."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        hopper: Hopper,
        keeps: Callable[[Moves], Collection[int]],
        past: int = 1,
        dead: Collection[Moves] = (),
    ) -> None:
        self.keeps, self.past, self.dead = keeps, past, dead
        self.calls: list[tuple[str, Moves, bool]] = []
        self.at: Moves = ()  # the moves the game was last played to, from the spot
        self.saved: list[tuple[str, Moves]] = []
        monkeypatch.setattr(search, "play", self)
        real = hopper.adapter.save_state

        def save(slot: str) -> None:
            self.saved.append((slot, self.at))
            real(slot)

        monkeypatch.setattr(hopper.adapter, "save_state", save)

    def __call__(
        self, connection: object, slot: str, moves: Sequence[int], space: object, **kw: object
    ) -> Outcome:
        played, seen = tuple(moves), kw.get("watch") is not None
        self.calls.append((slot, played, seen))
        if slot == END_SLOT:
            # A question: one move held, never watched, from the end last saved.
            assert not seen and len(set(played)) == 1 and self.saved[-1] == (END_SLOT, self.at)
            return Outcome(len(played), played[0] not in self.keeps(self.at), 0)
        assert slot == SPOT
        self.at = played
        if played in self.dead:
            return Outcome(3, True, 0)
        return Outcome(len(played), False, 3 if seen and len(played) >= self.past else 0)

    def tries(self) -> list[tuple[int, bool]]:
        """The plays from the spot: how many moves, and whether under the watch."""
        return [(len(moves), seen) for slot, moves, seen in self.calls if slot == SPOT]

    def asked(self) -> list[list[int]]:
        """The moves held at each question, question by question."""
        out: list[list[int]] = []
        fresh = True
        for slot, moves, _ in self.calls:
            if slot != END_SLOT:
                fresh = True
                continue
            if fresh:
                out.append([])
                fresh = False
            assert moves == (moves[0],) * K
            out[-1].append(moves[0])
        return out


def order(hopper: Hopper) -> list[int]:
    """The moves in the order they are asked: "do nothing" first, then as they come."""
    nothing = hopper.move()
    return [nothing, *(m for m in range(len(hopper.space)) if m != nothing)]


def first_tries(hopper: Hopper, trace: Sequence[int], carry_on: int) -> tuple[Change, Moves]:
    change = next(iter(changes(trace, hopper.space)))
    return change, change.apply(trace, carry_on)


def test_a_way_with_room_at_its_end_is_the_way(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    moves = order(hopper)
    assert len(moves) == 15 and hopper.space.buttons_for(moves[0]) == frozenset()
    trace = [hopper.move("right")] * 6
    change, tried = first_tries(hopper, trace, 4)
    course = Course(monkeypatch, hopper, keeps=lambda at: moves)
    found = look_for(hopper, trace, object, carry_on=4, outlast=K)
    # Blind, under the watch, and one question: three tries, and the whole try is the way.
    assert found == Search(Way(tried, change, tries=3, ground=3), 3)
    assert found.way is not None and found.way.cut == 0 and found.doomed == 0
    assert course.tries() == [(10, False), (10, True)]
    # The end of the try was saved once, and each move held from there for the steps
    # asked, "do nothing" first, until half of them had kept the life.
    assert course.saved == [(END_SLOT, tried)]
    assert course.asked() == [moves[:8]]


def test_how_much_room_there_is(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    moves = order(hopper)
    trace = [hopper.move("right")] * 6
    change, tried = first_tries(hopper, trace, 4)
    assert change.at == 5 and len(tried) == 10  # the only shorter part to ask about has 6

    def run(at_the_end: Collection[int], budget: int = 5) -> tuple[Search, list[list[int]]]:
        course = Course(monkeypatch, hopper, keeps=lambda at: at_the_end if at == tried else ())
        out = look_for(hopper, trace, object, carry_on=4, outlast=K, budget=budget)
        return out, course.asked()

    whole = Search(Way(tried, change, tries=3, ground=3), 3)
    # Half the moves, rounded up, is room: eight of fifteen, wherever they come.
    found, asked = run(moves[7:])
    assert found == whole and asked == [moves]
    found, asked = run(moves[:8])
    assert found == whole and asked == [moves[:8]]
    found, asked = run([m for i, m in enumerate(moves) if i % 2 == 0])
    assert found == whole and asked == [moves]
    # Seven is a tight spot: the shorter part is asked about, and with no room there
    # either the try is the way as it is. The asking stops once eight cannot be had.
    tight = Search(Way(tried, change, tries=5, ground=3), 5)
    found, asked = run(moves[8:])
    assert found == tight and asked == [moves[:9], moves]
    found, asked = run(moves[:1])
    assert found == tight and asked == [moves[:9], moves]
    # One at the very last is still a tight spot and not nothing: all are asked.
    found, asked = run(moves[-1:])
    assert found == tight and asked == [moves, moves]
    # None: nothing keeps the life at its end or at any part's. It is turned down.
    found, asked = run(())
    assert found == Search(None, 5, doomed=1) and asked == [moves, moves]
    # "Do nothing" is asked first wherever the controls have it, the rest as they come.
    pad = ActionSpace([{"right"}, {"a"}, set(), {"a", "right"}])
    course = Course(monkeypatch, hopper, keeps=lambda at: ())
    find_way(
        hopper.connection,
        SPOT,
        [0] * 6,
        pad,
        frames_per_step=FRAMES,
        lives=[LIVES],
        watch=object,  # type: ignore[arg-type]
        carry_on=4,
        outlast=K,
        budget=3,
    )
    assert course.asked() == [[2, 0, 1, 3]]


def roomy_before(hopper: Hopper, tried: Moves, tight: int, none: int) -> Callable[..., object]:
    """Room up to ``tight`` moves of the try, a tight spot from there, nothing from
    ``none`` on: a try that got past and carried on into the next place."""
    moves = order(hopper)

    def keeps(at: Moves) -> Collection[int]:
        if at != tried[: len(at)] or len(at) < tight:
            return moves
        return moves[:1] if len(at) < none else ()

    return keeps


def test_a_way_that_runs_on_is_stopped_where_there_is_room(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = [hopper.move("right")] * 10
    change, tried = first_tries(hopper, trace, 20)
    assert len(tried) == 30 and CUT_BY == 4
    course = Course(monkeypatch, hopper, roomy_before(hopper, tried, tight=19, none=26), past=12)
    found = look_for(hopper, trace, object, carry_on=20, outlast=K)
    # Nothing at its end, nothing 4 steps back, a tight spot 8 back, room 12 back: that
    # part is looked at, has got past, and is the way.
    assert course.tries() == [
        (30, False),
        (30, True),
        (26, False),
        (22, False),
        (18, False),
        (18, True),
    ]
    assert found == Search(Way(tried[:18], change, tries=10, ground=3, cut=12), 10)
    assert found.way is not None and found.way.cut == 12 and found.doomed == 0
    # Each part was saved at its own end and asked about there.
    assert [len(at) for _, at in course.saved] == [30, 26, 22, 18]
    # A try that is only in a tight spot at its end is stopped where there is room too.
    course = Course(monkeypatch, hopper, roomy_before(hopper, tried, tight=19, none=99), past=12)
    found = look_for(hopper, trace, object, carry_on=20, outlast=K)
    assert found == Search(Way(tried[:18], change, tries=10, ground=3, cut=12), 10)
    assert [len(at) for _, at in course.saved] == [30, 26, 22, 18]


def test_with_no_roomy_part_that_got_past(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    moves = order(hopper)
    trace = [hopper.move("right")] * 10
    change, tried = first_tries(hopper, trace, 20)
    listed = [c.apply(trace, 20) for c in changes(trace, hopper.space)]
    # The part with room has not got past yet (it takes 20 moves): the looking back
    # stops there. With nothing at its end, the try's longest part that can be lived
    # on from is the way, if that got past.
    course = Course(monkeypatch, hopper, roomy_before(hopper, tried, tight=19, none=26), past=20)
    found = look_for(hopper, trace, object, carry_on=20, outlast=K)
    assert course.tries()[-3:] == [(18, False), (18, True), (22, True)]
    assert found == Search(Way(tried[:22], change, tries=11, ground=3, cut=8), 11)
    # A try that is only tight at its end is the way as it is.
    course = Course(monkeypatch, hopper, roomy_before(hopper, tried, tight=19, none=99), past=20)
    found = look_for(hopper, trace, object, carry_on=20, outlast=K)
    assert course.tries()[-2:] == [(18, False), (18, True)]
    assert found == Search(Way(tried, change, tries=10, ground=3), 10)
    # Nothing at its end, and its longest part that can be lived on from has not got
    # past either: turned down. The search goes on, and the next try, with room at
    # its end, is the way. The one turned down is not played again.
    first = roomy_before(hopper, tried, tight=19, none=26)

    def keeps(at: Moves) -> Collection[int]:
        return moves if at == listed[1] else first(at)  # type: ignore[return-value]

    course = Course(monkeypatch, hopper, keeps, past=24)
    found = look_for(hopper, trace, object, carry_on=20, outlast=K)
    second = next(c for c in changes(trace, hopper.space) if c.apply(trace, 20) == listed[1])
    assert found == Search(Way(listed[1], second, tries=14, ground=3), 14, doomed=1)
    assert course.tries() == [
        (30, False),
        (30, True),
        (26, False),
        (22, False),
        (18, False),
        (18, True),
        (22, True),
        (30, False),
        (30, True),
    ]
    assert [m for s, m, _ in course.calls if s == SPOT].count(tried) == 2  # blind and watched
    # No part that can be lived on from at all: turned down without another look.
    course = Course(monkeypatch, hopper, keeps=lambda at: (), past=1)
    found = look_for(hopper, trace, object, carry_on=20, outlast=K, budget=13)
    assert found == Search(None, 13, doomed=1)
    assert course.tries() == [(30, False), (30, True), *((n, False) for n in (26, 22, 18, 14, 10))]


def test_how_far_back_it_looks(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    right = hopper.move("right")

    def parts(trace: list[int], carry_on: int) -> list[int]:
        """How long the parts asked about are, for a try with nothing anywhere."""
        course = Course(monkeypatch, hopper, keeps=lambda at: ())
        look_for(hopper, trace, object, carry_on=carry_on, outlast=K, budget=40)
        whole, rest = course.tries()[0][0], course.tries()[2:]
        upto = next((i for i, (n, _) in enumerate(rest) if n == whole), len(rest))
        return [n for n, _ in rest[:upto]]  # up to the next try's own blind play

    # No shorter than the change the try is made of: the first try changes the last
    # step of what it did, so its parts keep that step.
    assert parts([right] * 10, 20) == [26, 22, 18, 14, 10]
    assert parts([right] * 13, 8) == [17, 13]
    assert parts([right] * 9, 0) == []  # a try that is all trace has no part to cut to
    # And no further back than 40 steps from its end, however long it carried on.
    assert CUT_WITHIN == 40
    assert parts([right] * 10, 70) == list(range(76, 39, -4))


def test_the_asking_keeps_to_the_budget_and_the_time(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = [hopper.move("right")] * 10
    change, tried = first_tries(hopper, trace, 20)
    keeps = roomy_before(hopper, tried, tight=19, none=26)
    full = Search(Way(tried[:18], change, tries=10, ground=3, cut=12), 10)
    # Ten tries are what it takes: with ten it is found, with any fewer nothing is.
    Course(monkeypatch, hopper, keeps, past=12)
    assert look_for(hopper, trace, object, carry_on=20, outlast=K, budget=10) == full
    for budget in range(1, 10):
        course = Course(monkeypatch, hopper, keeps, past=12)
        found = look_for(hopper, trace, object, carry_on=20, outlast=K, budget=budget)
        assert found == Search(None, budget), budget
        # A try whose asking was cut off was not turned down either: it is not known.
        assert found.doomed == 0
        # Nothing was played that the tries did not cover: a blind play, a look and a
        # question are one each.
        plays = len(course.tries()) + len(course.asked())
        assert plays == budget, budget
    # The time: a second passes with every play from the spot, and none starts late.
    course = Course(monkeypatch, hopper, keeps, past=12)
    ticks = [0.0]

    def clock() -> float:
        return ticks[0] + len(course.tries())

    found = look_for(hopper, trace, object, carry_on=20, outlast=K, seconds=4, clock=clock)
    assert found.way is None and found.out_of_time and found.doomed == 0
    assert course.tries() == [(30, False), (30, True), (26, False), (22, False)]


def test_the_plain_way_and_a_standstill_are_asked_too(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    moves = order(hopper)
    back, on = hopper.move("left"), hopper.move("right")
    trace = [back] * 10
    held = (on,) * 30

    def keeps(at: Moves) -> Collection[int]:
        return moves if len(at) < 19 else ()

    # The plain way gets past and runs on: it is cut like any try, down to one move
    # if need be, and is still the plain way.
    course = Course(monkeypatch, hopper, keeps, past=12)
    found = look_for(hopper, trace, object, carry_on=20, outlast=K, plain=on)
    assert course.tries() == [
        (30, False),
        (30, True),
        (26, False),
        (22, False),
        (18, False),
        (18, True),
    ]
    way = Way(held[:18], Change(0, on, 10), tries=10, ground=3, plain=on, cut=12)
    assert found == Search(way, 10)
    # At a standstill a part must also end out of the spot: the roomy part that does
    # not is no way, and with nothing at the try's end there is none.
    seen: list[int] = []

    def out_of_the_spot(outcome: Outcome) -> bool:
        seen.append(outcome.steps)
        return outcome.steps >= 20

    course = Course(monkeypatch, hopper, keeps, past=12)
    found = look_for(
        hopper,
        trace,
        object,
        carry_on=20,
        outlast=K,
        plain=on,
        stood=True,
        enough=out_of_the_spot,
        budget=10,
    )
    assert seen == [30, 18] and found == Search(None, 10, doomed=1)


def test_without_it_nothing_changes_and_nonsense_is_refused(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = [hopper.move("right")] * 6
    change, tried = first_tries(hopper, trace, 4)
    for kw in ({}, {"outlast": 0}):
        course = Course(monkeypatch, hopper, keeps=lambda at: ())
        found = look_for(hopper, trace, object, carry_on=4, **kw)
        assert found == Search(Way(tried, change, tries=2, ground=3), 2)
        assert course.tries() == [(10, False), (10, True)]
        assert course.saved == [] and course.asked() == []
    for wrong in (-1, 1.5, True, None, "30"):
        with pytest.raises(ConfigurationError, match="outlast"):
            look_for(hopper, trace, object, outlast=wrong)
    # As far as the app looks back for the cause of a life lost, and a slot of its own.
    assert OUTLAST == LEAD == 30
    assert END_SLOT not in {NOW_SLOT, CHECKPOINT_SLOT, PRACTISE_SLOT, SPOT, *BACK_SLOTS}
