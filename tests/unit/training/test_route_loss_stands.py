"""T-1029: a place where a life went for good before is come at from further back.

The way ends a lead before a loss no search found a way past, so the next go
arrives with the same little room and fails the same way. The route notes where
a loss stood, and each earlier one at about the same place doubles the lead.
"""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training.route import DOUBLINGS, SAME_PLACE, Route

A = 1  # a move
HERE = ("training", 1)  # a saved spot goes start from
THERE = ("training", 2)
LEAD = 10


def walked(route: Route, steps: int, start: object = HERE) -> None:
    """A go from ``start`` that has played ``steps`` steps, a point paid for each."""
    route.begin(start)  # type: ignore[arg-type]
    for count in range(1, steps + 1):
        route.did(A, count)


def stood(route: Route, lead: int) -> int:
    """The loss on the step just played stands: what the runner does about it."""
    got = route.loss_stands(lead)
    route.failed(got)
    route.lost(got)
    return got


def loss_at(route: Route, step: int, start: object = HERE) -> int:
    """A whole go that loses a life for good on step ``step``: the lead it is given."""
    walked(route, step, start)
    got = stood(route, LEAD)
    route.end()
    return got


def test_each_earlier_loss_at_the_place_doubles_the_lead() -> None:
    assert (SAME_PLACE, DOUBLINGS) == (2, 3)
    route = Route()
    # The first loss at a place is what it always was; then twice, four times, eight
    # times as far back, and no further.
    assert [loss_at(route, 100) for _ in range(6)] == [10, 20, 40, 80, 80, 80]
    # About the same place: within two leads either side, the edge included.
    route = Route()
    assert loss_at(route, 100) == 10
    assert loss_at(route, 120) == 20  # exactly two leads on
    assert loss_at(route, 141) == 10  # 21 steps past the nearest
    assert loss_at(route, 79) == 10  # 21 steps short of the nearest
    assert loss_at(route, 80) == 40  # 79 and 100 are both within twenty
    # What "two leads" is goes by the lead asked with.
    wide = Route()
    walked(wide, 100)
    assert stood(wide, 30) == 30
    wide.end()
    walked(wide, 160)
    assert stood(wide, 30) == 60  # sixty steps on is the same place at a lead of thirty
    wide.end()
    walked(wide, 221)
    assert stood(wide, 30) == 30
    # The place is how long the story is, steps gone back over not in it.
    back = Route()
    walked(back, 50)
    back.went_back(20)
    for count in range(5):
        back.did(A, 31 + count)
    assert len(back.story) == 35 and stood(back, LEAD) == 10
    back.end()
    assert loss_at(back, 56) == 10  # 21 on from step 35 (55 steps were played in all)
    assert loss_at(back, 15) == 20  # and 20 short of it


def test_nothing_is_noted_while_the_story_is_over() -> None:
    route = Route()
    walked(route, 100)
    assert stood(route, LEAD) == 10
    # The second and third lives of the same go: the story ended with the first, and
    # no longer says where the go is.
    assert route.story == (A,) * 90
    assert [stood(route, LEAD) for _ in range(3)] == [10, 10, 10]
    route.end()
    assert loss_at(route, 100) == 20  # one loss was noted there, not four
    # A step nobody can play again ended the story before the life went.
    walked(route, 100)
    route.did(None, 100)
    route.did(A, 101)
    assert stood(route, LEAD) == 10
    route.end()
    assert loss_at(route, 100) == 40  # two were noted, not three
    # A go that was not put back to a saved spot has no story at all.
    fresh = Route()
    assert loss_at(fresh, 5) == 10
    fresh.begin(None)
    fresh.did(A, 1)
    assert fresh.story == () and stood(fresh, LEAD) == 10
    fresh.end()
    assert loss_at(fresh, 26) == 10  # nothing was noted at step 0 or 1: only step 5 is
    assert loss_at(fresh, 6) == 40  # (5 and 26 are both within twenty of 6)
    # Before any go at all there is nothing to note either.
    never = Route()
    assert never.loss_stands(LEAD) == 10
    assert loss_at(never, 0) == 10


def test_the_notes_last_as_long_as_the_start() -> None:
    route = Route()
    route.begin(HERE)
    for count in range(1, 101):
        route.did(A, count, [("row", 1.0)])
    assert stood(route, LEAD) == 10
    assert route.end() and len(route.moves) == 90
    # Through the end of the go, a reward row that goes and the way being cut.
    assert route.gone(["row"]) == frozenset({"row"}) and route.moves == ()
    assert loss_at(route, 100) == 20
    walked(route, 95)
    assert route.loss_stands(LEAD) == 40
    assert route.failed(40) is True and len(route.moves) == 55
    route.lost(40)
    route.end()
    assert loss_at(route, 100) == 80
    # A way that leads nowhere any more takes its places with it.
    route.forget()
    assert loss_at(route, 100) == 10
    assert loss_at(route, 100) == 20
    # Steps from another start are other places.
    assert loss_at(route, 100, THERE) == 10
    assert loss_at(route, 100, THERE) == 20
    assert loss_at(route, 100, HERE) == 10


def test_a_lead_that_is_no_whole_number_is_refused() -> None:
    route = Route()
    walked(route, 100)
    for bad in (True, False, 1.5, -1, "10", None):
        with pytest.raises(ConfigurationError, match="The lead must be a whole number"):
            route.loss_stands(bad)  # type: ignore[arg-type]
    assert stood(route, LEAD) == 10  # a call refused noted nothing
    route.end()
    # No lead at all stays none, however often.
    walked(route, 100)
    assert stood(route, 0) == 0
