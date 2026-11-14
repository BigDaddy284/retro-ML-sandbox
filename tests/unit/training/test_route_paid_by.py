"""T-1027: the way's pay is kept by what paid it.

T-1026 lowers a way's figure when a go that plays it again is paid less. Pay that
went and does not show there (a go that went back is not paid twice for the same
ground, so the figures a way is remembered with run low) stayed in. Each step's pay
now comes with the reward rows it was made of, and when a row goes its pay is taken
out of the way, of the go being played and of that go's notes, exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training.route import Route

A, B, C = 1, 2, 3  # three moves
HERE = ("training", 1)  # a saved spot goes start from

Part = tuple[str, float]
Step = tuple[int | None, float, Sequence[Part]]


def played(route: Route, steps: Sequence[Step]) -> None:
    """Each step with what the story had been paid after it, and what paid this step."""
    for move, paid, by in steps:
        route.did(move, paid, by)


def remembering(steps: Sequence[Step]) -> Route:
    route = Route()
    assert route.paid_by == ()
    route.begin(HERE)
    played(route, steps)
    assert route.end() and route.moves == tuple(move for move, _, _ in steps)
    assert route.paid_by == tuple(tuple(by) for _, _, by in steps)
    return route


def figures(route: Route) -> list[float]:
    """What the way had been paid after each step: the parts added up (in these
    tests every point of a way has a row it came from)."""
    out, total = [], 0.0
    for parts in route.paid_by:
        total += sum(amount for _, amount in parts)
        out.append(total)
    assert not out or out[-1] == route.paid
    return out


G2: Sequence[Part] = (("ground", 2.0),)
CLOCK: Sequence[Part] = (("clock", 10.0),)
# Six steps paid 27: 20 by a clock, 6.5 for new ground, half a point by the score.
WAY: Sequence[Step] = [
    (A, 2.0, G2),
    (A, 4.0, G2),
    (B, 14.0, CLOCK),
    (B, 24.0, CLOCK),
    (C, 26.0, G2),
    (C, 27.0, (("ground", 0.5), ("score", 0.5))),
]
MOVES = (A, A, B, B, C, C)


def test_pay_that_is_gone_is_taken_out_of_the_way() -> None:
    route = remembering(WAY)
    assert figures(route) == [2.0, 4.0, 14.0, 24.0, 26.0, 27.0]
    # A go paid 8 for other moves is no match for the 27.
    route.begin(HERE)
    played(route, [(B, 3.0, (("ground", 3.0),)), (B, 8.0, (("ground", 5.0),))])
    assert route.end() is False and route.moves == MOVES
    # The clock's row goes: its 20 points are out, step by step.
    assert route.gone({"clock"}) == frozenset({"clock"})
    assert (route.moves, route.paid) == (MOVES, 7.0)
    assert figures(route) == [2.0, 4.0, 4.0, 4.0, 6.0, 7.0]
    assert route.paid_by[2] == route.paid_by[3] == ()
    # Gone a second time, a row that paid nothing here, and no row at all: nothing.
    for ids in ({"clock"}, {"lives"}, set(), ["clock", "lives"]):
        assert route.gone(ids) == frozenset()
        assert (route.moves, route.paid) == (MOVES, 7.0)
    # The same go again is now the better one.
    route.begin(HERE)
    played(route, [(B, 3.0, (("ground", 3.0),)), (B, 8.0, (("ground", 5.0),))])
    assert route.end() is True and (route.moves, route.paid) == ((B, B), 8.0)

    # Two rows at once, and only those that had paid here are answered with.
    route = remembering(WAY)
    assert route.gone(["clock", "score", "lives"]) == frozenset({"clock", "score"})
    assert (route.moves, route.paid) == (MOVES, 6.5)
    assert figures(route) == [2.0, 4.0, 4.0, 4.0, 6.0, 6.5]
    assert route.paid_by[5] == (("ground", 0.5),)

    # Pay that came with no row to its name stays whatever goes.
    route = remembering([(A, 3.0, (("clock", 1.0),)), (A, 5.0, ()), (B, 9.0, (("clock", 3.0),))])
    assert route.gone({"clock"}) == frozenset({"clock"})
    assert (route.moves, route.paid) == ((A, A, B), 5.0)  # 2, 2 and 1 of nobody's


def test_a_way_ends_where_what_is_left_pays_the_most() -> None:
    # The end of the way was paid by the clock alone.
    route = remembering([(A, 2.0, G2), (A, 4.0, G2), (B, 14.0, CLOCK), (B, 24.0, CLOCK)])
    assert route.gone({"clock"}) == frozenset({"clock"})
    assert (route.moves, route.paid, route.paid_by) == ((A, A), 4.0, (tuple(G2), tuple(G2)))
    assert route.begin(HERE) == (A, A) and route.playing == 2

    # The highest of what is left is inside the way: a row that took points away
    # stays, and the way ends before it. The first of two equal highs.
    route = remembering(
        [
            (A, 5.0, (("ground", 5.0),)),
            (A, 2.0, (("hurt", -3.0),)),
            (B, 5.0, (("ground", 3.0),)),
            (B, 15.0, CLOCK),
        ]
    )
    assert route.gone({"clock"}) == frozenset({"clock"})
    assert (route.moves, route.paid) == ((A,), 5.0)
    # The other way round: the row that took points away goes, and the figure rises.
    route = remembering(
        [(A, 5.0, (("ground", 5.0),)), (A, 2.0, (("hurt", -3.0),)), (B, 6.0, (("ground", 4.0),))]
    )
    assert route.gone({"hurt"}) == frozenset({"hurt"})
    assert (route.moves, route.paid) == ((A, A, B), 9.0)
    assert figures(route) == [5.0, 5.0, 9.0]

    # Nothing above zero is left: no way, and the spot the goes start from is known.
    for second in ((B, 20.0, CLOCK), (B, 19.0, (*CLOCK, ("hurt", -1.0)))):
        route = remembering([(A, 10.0, CLOCK), second])
        assert route.gone({"clock"}) == frozenset({"clock"})
        assert (route.moves, route.paid, route.paid_by) == ((), 0.0, ())
        assert route.begin(HERE) == ()
        played(route, [(C, 1.0, ()), (C, 1.0, ())])
        assert route.end() is True and route.moves == (C,)


def test_the_go_being_played_loses_that_pay_too() -> None:
    # No way yet. The clock's row goes in the middle of a go.
    route = Route()
    route.begin(HERE)
    played(route, [(A, 2.0, G2), (A, 4.0, G2), (B, 14.0, CLOCK)])
    assert route.counted == 14.0
    assert route.gone({"clock"}) == frozenset()  # there is no way it had paid for
    assert route.counted == 4.0
    played(route, [(C, 5.0, (("ground", 1.0),)), (C, 5.0, ())])
    assert route.end() is True
    assert (route.moves, route.paid) == ((A, A, B, C), 5.0)
    assert figures(route) == [2.0, 4.0, 4.0, 5.0]
    assert route.gone({"clock"}) == frozenset()  # it is in none of this any more

    # Like with like at the go's end: the way and the go lose the same row.
    route = remembering(WAY)
    route.begin(HERE)
    played(route, [(B, 12.0, (("clock", 10.0), ("ground", 2.0))), (B, 20.0, (("ground", 8.0),))])
    assert route.gone({"clock"}) == frozenset({"clock"})
    assert (route.paid, route.counted) == (7.0, 10.0)
    assert route.end() is True and (route.moves, route.paid) == ((B, B), 10.0)
    # With the go's pay left as it was, a go worth 6 would have beaten the 7.
    route = remembering(WAY)
    route.begin(HERE)
    played(route, [(B, 12.0, (("clock", 10.0), ("ground", 2.0))), (B, 16.0, (("ground", 4.0),))])
    route.gone({"clock"})
    assert route.end() is False and (route.moves, route.paid) == (MOVES, 7.0)

    # Steps gone back over take their parts with them: what goes later is taken out
    # of what is still the story.
    route = Route()
    route.begin(HERE)
    played(route, [(A, 2.0, G2), (B, 12.0, CLOCK), (B, 22.0, CLOCK)])
    route.went_back(1)
    played(route, [(C, 13.0, (("ground", 1.0),))])
    route.gone({"clock"})
    assert route.counted == 3.0
    assert route.end() is True and figures(route) == [2.0, 2.0, 3.0]
    assert route.paid_by == (tuple(G2), (), (("ground", 1.0),))
    # So do the steps a loss that stands takes off the story.
    route = Route()
    route.begin(HERE)
    played(route, [(A, 2.0, G2), (B, 12.0, CLOCK), (B, 22.0, CLOCK), (C, 23.0, ())])
    route.lost(2)
    assert route.end() is True and route.paid_by == (tuple(G2), tuple(CLOCK))

    # The notes of a go that is playing the way (T-1026) lose that pay as well: the
    # way made again from them is not cut at a figure that no longer exists.
    way: Sequence[Step] = [*WAY[:4], (C, 30.0, (("ground", 6.0),))]
    route = remembering(way)
    assert route.begin(HERE) == (A, A, B, B, C)
    played(route, way[:3])  # three steps of it, the third still paid by the clock
    assert route.gone({"clock"}) == frozenset({"clock"})
    assert (route.moves, route.paid, route.counted) == ((A, A, B, B, C), 10.0, 4.0)
    # The rest of it, with nothing from the clock, and a point less for the ground.
    played(route, [(B, 4.0, ()), (C, 9.0, (("ground", 5.0),))])
    assert route.end() is False
    assert (route.moves, route.paid, route.shortened) == ((A, A, B, B, C), 9.0, 0)
    assert figures(route) == [2.0, 4.0, 4.0, 4.0, 9.0]


def test_the_parts_follow_every_change() -> None:
    both: Sequence[Part] = (("ground", 1.0), ("clock", 1.0))
    ten: Sequence[Step] = [(A, 2.0 * n, both) for n in range(1, 11)]
    # A way cut for a life lost while it was played.
    route = remembering(ten)
    route.begin(HERE)
    played(route, ten[:7])
    assert route.failed(4) is True
    assert (route.moves, route.paid, route.paid_by) == ((A,) * 3, 6.0, (tuple(both),) * 3)
    assert route.gone({"clock"}) == frozenset({"clock"})
    assert figures(route) == [1.0, 2.0, 3.0]
    # The go carries on from two steps in, and its story becomes the way: the parts
    # are the story's, the clock's pay already out of them.
    route.went_back(5)
    played(route, [(B, 4.0, G2), (B, 6.0, G2)])
    assert route.end() is True
    assert (route.moves, route.paid) == ((A, A, B, B), 6.0)
    assert route.paid_by == ((("ground", 1.0),), (("ground", 1.0),), tuple(G2), tuple(G2))

    # A way made again from a go that played it for less (T-1026) has that go's parts.
    route = remembering(WAY)
    route.begin(HERE)
    again: Sequence[Step] = [
        (A, 2.0, G2),
        (A, 4.0, G2),
        (B, 4.0, ()),
        (B, 4.5, (("score", 0.5),)),
        (C, 4.5, ()),
        (C, 4.5, ()),
    ]
    played(route, again)
    assert route.end() is False
    assert (route.moves, route.paid, route.shortened) == ((A, A, B, B), 4.5, 2)
    assert route.paid_by == (tuple(G2), tuple(G2), (), (("score", 0.5),))
    assert route.gone({"clock"}) == frozenset()  # that go was paid nothing by it
    assert route.gone({"score"}) == frozenset({"score"})
    assert (route.moves, route.paid) == ((A, A), 4.0)

    # Forgotten, a way leaves no parts; nor does another spot's.
    route = remembering(WAY)
    route.forget()
    assert (route.moves, route.paid_by) == ((), ())
    route = remembering(WAY)
    assert route.begin(("training", 2)) == () and route.paid_by == ()
    # A go never ended leaves no parts in the next one's story.
    route = remembering(WAY)
    route.begin(HERE)
    played(route, [(B, 50.0, (("clock", 50.0),))])
    route.begin(HERE)
    played(route, [(C, 30.0, (("ground", 30.0),))])
    assert route.end() is True and route.paid_by == ((("ground", 30.0),),)


def test_without_parts_nothing_changes_and_nonsense_is_refused() -> None:
    # A step told without parts is paid by no row: nothing that goes touches it.
    route = Route()
    route.begin(HERE)
    route.did(A, 1.0)
    route.did(A, 3.0)
    assert route.end() is True and route.paid_by == ((), ())
    assert route.gone({"clock", "ground"}) == frozenset()
    assert (route.moves, route.paid) == ((A, A), 3.0)
    # A step nobody can play again, and one after the story is over, are told with
    # parts like any other and leave none.
    route.begin(HERE)
    played(route, [(A, 1.0, G2), (None, 3.0, G2), (B, 5.0, G2)])
    assert route.story == (A,) and route.counted == 1.0
    # A part is a row's name and a number.
    nan, inf = float("nan"), float("inf")
    wrong: list[object] = [
        [("clock",)],
        [("clock", 1.0, 2.0)],
        [(1, 1.0)],
        [("clock", "1")],
        [("clock", True)],
        [("clock", nan)],
        [("clock", inf)],
        ["clock"],
        5,
    ]
    for by in wrong:
        with pytest.raises(ConfigurationError):
            route.did(A, 1.0, by)  # type: ignore[arg-type]
    # The rows that went are names, and a single name is not a list of them.
    for ids in ("clock", [1], [None], 5):
        with pytest.raises(ConfigurationError):
            route.gone(ids)  # type: ignore[arg-type]
