"""T-1028: a story that went back is counted like one played straight through.

When the game is put back to a saved moment, the go's own counts stay (the furthest
ground reached, the best a progress number or the score has been): ground walked a
second time pays nothing until the go is past its furthest again. So the story of a
go that went back ran lower than the same moves played straight from the start, by
what those rows had paid for the steps gone back over, and a go that got further
could be held worse than the way it began with. What such rows paid for the steps
gone back over is now owed to the story, and comes back with the step at which the
row pays again: from there the story has what playing it straight would be paid.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training.route import Route

A, B, C = 1, 2, 3  # three moves
HERE = ("training", 1)  # a saved spot goes start from
KEPT = ("ground", "progress")  # rows that pay only past the go's best

Part = tuple[str, float]
G: Sequence[Part] = (("ground", 1.0),)
NOTHING: Sequence[Part] = ()


def walk(route: Route, move: int, *steps: Sequence[Part]) -> None:
    """Steps of ``move``, each paid what its parts add up to on top of the story."""
    for by in steps:
        route.did(move, route.counted + sum(amount for _, amount in by), by)


def straight(moves: Sequence[int], pays: Sequence[Sequence[Part]]) -> Route:
    """The same moves played from the start with no going back."""
    route = Route()
    route.begin(HERE)
    for move, by in zip(moves, pays, strict=True):
        walk(route, move, by)
    return route


def test_pay_gone_back_over_comes_back_when_the_row_pays_again() -> None:
    route = Route()
    route.begin(HERE)
    walk(route, A, G, G, G, G, G)  # five steps of new ground
    walk(route, A, (("life", -7.5),))  # ... and into a hole
    assert route.counted == -2.5
    # Put back three steps before the hole: the story is three steps of ground.
    route.went_back(3, KEPT)
    assert (route.story, route.counted) == ((A, A, A), 3.0)
    # The fourth and fifth step are walked again and pay nothing: the go has been there.
    walk(route, B, NOTHING, NOTHING)
    assert route.counted == 3.0
    # The sixth is new ground. Played straight, these six moves are paid six points.
    walk(route, B, G)
    assert route.counted == 6.0
    assert route.end() is True
    assert (route.moves, route.paid) == ((A, A, A, B, B, B), 6.0)
    # What came back is the row's own: it stands with the step that brought it.
    assert route.paid_by[3:] == ((), (), (("ground", 3.0),))
    same = straight((A, A, A, B, B, B), [G, G, G, G, G, G])
    assert same.counted == route.paid

    # Several rows, each back when it pays again and not before.
    route = Route()
    route.begin(HERE)
    walk(route, A, G, (("ground", 1.0), ("progress", 10.0)), G)
    route.went_back(2, KEPT)
    assert route.counted == 1.0
    walk(route, B, NOTHING, G)  # past its furthest ground, not yet past its progress
    assert route.counted == 4.0
    walk(route, B, (("progress", 10.0),))  # the next stage, and the one gone back over
    assert route.counted == 24.0
    assert route.end() is True
    assert route.paid_by == (
        (("ground", 1.0),),
        (),
        (("ground", 3.0),),
        (("progress", 20.0),),
    )


def test_a_go_that_got_further_after_going_back_beats_the_way() -> None:
    def way() -> Route:
        route = Route()
        route.begin(HERE)
        walk(route, A, G, G, G, G)
        assert route.end() and route.paid == 4.0
        return route

    # The next go plays the way, walks two steps on, loses a life, is put back four
    # steps (two of them the way's own), finds another way and gets one step past
    # where it had been: seven steps of ground where the way has four.
    route = way()
    assert route.begin(HERE) == (A, A, A, A)
    walk(route, A, G, G, G, G)
    walk(route, B, G, G)
    walk(route, B, (("life", -7.5),))
    route.went_back(5, KEPT)
    assert route.counted == 2.0
    walk(route, C, NOTHING, NOTHING, NOTHING, NOTHING, G)
    assert route.counted == 7.0
    assert route.end() is True
    assert (route.moves, route.paid) == ((A, A, C, C, C, C, C), 7.0)

    # Without being told which rows keep their count, the same go is counted as it
    # was before: three points for seven steps of ground, and the way of four stands.
    route = way()
    route.begin(HERE)
    walk(route, A, G, G, G, G)
    walk(route, B, G, G)
    walk(route, B, (("life", -7.5),))
    route.went_back(5)
    walk(route, C, NOTHING, NOTHING, NOTHING, NOTHING, G)
    assert route.counted == 3.0
    assert route.end() is False
    assert (route.moves, route.paid) == ((A, A, A, A), 4.0)

    # A go that does not get past where it had been is owed nothing it can show:
    # its story is the way's own beginning, and the way stands.
    route = way()
    route.begin(HERE)
    walk(route, A, G, G, G, G)
    walk(route, B, G, G)
    route.went_back(4, KEPT)
    walk(route, C, NOTHING, NOTHING, NOTHING)
    assert route.counted == 2.0
    assert route.end() is False
    assert (route.moves, route.paid) == ((A, A, A, A), 4.0)


def test_only_the_rows_named_are_owed() -> None:
    # A penalty, a reward that pays every time and pay that is nobody's are not owed:
    # what happens again is paid again, and a life lost and gone back over is not lost.
    route = Route()
    route.begin(HERE)
    walk(route, A, G)
    walk(route, A, (("ground", 1.0), ("coin", 5.0), ("life", -7.5)))
    route.did(A, route.counted + 2.0)  # two points with no row's name
    assert route.counted == 1.5
    route.went_back(2, KEPT)
    assert route.counted == 1.0
    walk(route, B, (("coin", 5.0),))  # the coin is there again, and taken again
    assert route.counted == 6.0
    walk(route, B, G)
    assert route.counted == 8.0  # one step of ground owed, one new
    walk(route, B, G)
    assert route.counted == 9.0  # nothing is owed twice

    # A row that never pays again never brings anything back.
    route = Route()
    route.begin(HERE)
    walk(route, A, G, (("progress", 10.0),))
    route.went_back(1, KEPT)
    walk(route, B, G, G)
    assert route.counted == 3.0

    # Going back twice: what was owed and not yet back is still owed, and a step
    # that brought something back owes all of it when it is gone back over itself.
    route = Route()
    route.begin(HERE)
    walk(route, A, G, G, G, G)
    route.went_back(2, KEPT)  # two owed
    walk(route, B, NOTHING)
    route.went_back(2, KEPT)  # back to the first step: three owed
    assert route.counted == 1.0
    walk(route, C, NOTHING, NOTHING, NOTHING, G)
    assert route.counted == 5.0
    route.went_back(1, KEPT)  # over the step that brought three back: four owed
    assert route.counted == 1.0
    walk(route, C, NOTHING, G)  # the fifth step of ground again, then a sixth
    assert route.counted == 6.0

    # Taking a point away is owed like giving one: the row decides, not the sign.
    route = Route()
    route.begin(HERE)
    walk(route, A, (("progress", -2.0),), G)
    route.went_back(2, KEPT)
    walk(route, B, (("progress", 5.0),))
    assert route.counted == 3.0

    # The whole story gone back over.
    route = Route()
    route.begin(HERE)
    walk(route, A, G, G)
    route.went_back(2, KEPT)
    assert (route.story, route.counted) == ((), 0.0)
    walk(route, B, G)
    assert route.counted == 3.0


def test_what_is_owed_ends_with_the_go_and_with_the_row() -> None:
    # A new go is owed nothing from the last.
    route = Route()
    route.begin(HERE)
    walk(route, A, G, G, G)
    route.went_back(2, KEPT)
    assert route.end() is True and route.paid == 1.0
    route.begin(HERE)
    walk(route, A, G, G)
    assert route.counted == 2.0
    route.end()

    # A row that has gone from the rewards is owed nothing either (T-1027).
    route = Route()
    route.begin(HERE)
    walk(route, A, G, (("ground", 1.0), ("progress", 10.0)), G)
    route.went_back(2, KEPT)
    assert route.gone(["progress"]) == frozenset()  # the way had none of it
    walk(route, B, (("ground", 1.0), ("progress", 10.0)))
    assert route.counted == 14.0  # the ground owed, and the progress paid this once

    # A story that is over (a life lost that stands, a step nobody can play again,
    # a go that began from no saved spot) is not gone back in.
    route = Route()
    route.begin(HERE)
    walk(route, A, G, G, G)
    route.lost(1)
    route.went_back(1, KEPT)
    walk(route, B, G)
    assert (route.story, route.counted) == ((A, A), 2.0)
    route = Route()
    route.begin(None)
    walk(route, A, G, G)
    route.went_back(1, KEPT)
    assert route.counted == 0.0

    # While the way is being played again and nothing has been gone back over, the
    # notes of what it is paid now (T-1026) are as they were; a going back ends them.
    route = Route()
    route.begin(HERE)
    walk(route, A, G, G, (("clock", 10.0),))
    assert route.end() and route.paid == 12.0
    route.begin(HERE)
    walk(route, A, G, G, NOTHING)  # the clock pays no longer
    assert route.end() is False and route.paid == 2.0 and route.shortened == 1
    route.begin(HERE)
    walk(route, A, G, G)
    route.went_back(1, KEPT)
    walk(route, A, NOTHING)
    assert route.end() is False and (route.moves, route.paid) == ((A, A), 2.0)


def test_without_rows_named_nothing_changes_and_nonsense_is_refused() -> None:
    route = Route()
    route.begin(HERE)
    walk(route, A, G, G, G)
    route.went_back(2)
    walk(route, B, NOTHING, G)
    assert route.counted == 2.0
    route.went_back(1, ())
    walk(route, B, G)
    assert route.counted == 2.0
    # Rows named that paid for nothing gone back over change nothing.
    route.went_back(1, ["score", "clock"])
    walk(route, B, G)
    assert route.counted == 2.0
    for wrong in ("ground", 7, None, [3], [("ground", 1.0)]):
        with pytest.raises(ConfigurationError):
            route.went_back(1, wrong)  # type: ignore[arg-type]
    assert (route.story, route.counted) == ((A, B, B), 2.0)  # and nothing was cut
    with pytest.raises(ConfigurationError):
        route.went_back(9, KEPT)
    assert (route.story, route.counted) == ((A, B, B), 2.0)
    # Refused as well when the story is over and nothing would be owed anyway.
    route.lost(1)
    with pytest.raises(ConfigurationError):
        route.went_back(1, "ground")  # type: ignore[arg-type]
