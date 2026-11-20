"""T-1064: the remembered way goes to plain data and back.

The way was in the app's memory only, and closing the app lost it. What is
remembered (the moves, the pay after each, what paid for each, and the steps a
loss stood at) is written as a dictionary a JSON file can hold, and read back
into a route that starts from the saved spot the caller says is the same one.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training.route import Route

START = ("training", 1)


def a_way() -> Route:
    """A route that remembers five moves, paid by two reward rows, with a loss that
    stood eight steps in."""
    route = Route()
    assert route.begin(START) == ()
    paid = 0.0
    for at, move in enumerate([3, 3, 1, 0, 2, 2, 2, 2]):
        gave = (("x", 1.0),) if at % 2 == 0 else (("x", 0.5), ("score", 0.25))
        paid += sum(amount for _, amount in gave)
        route.did(move, paid, gave)
    assert route.loss_stands(3) == 3
    route.lost(3)
    assert route.end()
    return route


def there_and_back(route: Route, start: object = START) -> Route:
    written = json.loads(json.dumps(route.to_dict()))
    assert written == route.to_dict()  # plain data: a JSON file holds it as it is
    return Route.from_dict(written, start)


def test_a_way_goes_to_plain_data_and_back() -> None:
    route = a_way()
    assert route.moves == (3, 3, 1, 0, 2) and route.start == START
    back = there_and_back(route)
    assert back.moves == route.moves and back.paid == route.paid == 4.5
    assert back.paid_by == route.paid_by
    assert back.to_dict() == route.to_dict()
    assert back.start == START
    # It is a way to play: a go from the same spot begins with it.
    assert back.begin(START) == (3, 3, 1, 0, 2) and back.playing == 5


def test_what_is_written() -> None:
    assert a_way().to_dict() == {
        "moves": [3, 3, 1, 0, 2],
        "paid": [1.0, 1.75, 2.75, 3.5, 4.5],
        "by": [
            [["x", 1.0]],
            [["x", 0.5], ["score", 0.25]],
            [["x", 1.0]],
            [["x", 0.5], ["score", 0.25]],
            [["x", 1.0]],
        ],
        "stood": [8],
    }
    assert Route().to_dict() == {"moves": [], "paid": [], "by": [], "stood": []}


def test_the_places_a_loss_stood_come_back_too() -> None:
    """A place that cost a life is come at from twice as far back the next time
    (T-1029): that holds for a way read back as for the one remembered."""
    for route in (a_way(), there_and_back(a_way())):
        route.begin(START)
        for move in (3, 3, 1, 0, 2, 2, 2, 2):
            route.did(move, 0.0)
        assert route.loss_stands(3) == 6


def test_a_way_read_back_loses_what_a_row_that_went_had_paid() -> None:
    back = there_and_back(a_way())
    assert back.gone(["x"]) == frozenset({"x"})
    assert back.moves == (3, 3, 1, 0) and back.paid == 0.5
    assert back.paid_by == ((), (("score", 0.25),), (), (("score", 0.25),))


def test_a_way_from_another_start_is_forgotten_as_any_other() -> None:
    back = there_and_back(a_way(), ("training", 7))
    assert back.start == ("training", 7)
    assert back.begin(("training", 8)) == () and back.moves == ()
    assert back.to_dict() == {"moves": [], "paid": [], "by": [], "stood": []}


def test_the_go_being_played_is_not_written() -> None:
    route = a_way()
    before = route.to_dict()
    route.begin(START)
    route.did(3, 1.0, (("x", 1.0),))
    route.did(0, 9.0, (("x", 8.0),))  # off the way, and paid well: not the way yet
    assert route.to_dict() == before
    back = there_and_back(route)
    assert back.story == () and back.counted == 0.0 and back.playing == 0
    assert not back.end()  # no go was begun: nothing to make a way of


def test_what_is_written_is_a_copy() -> None:
    route = a_way()
    written = route.to_dict()
    written["moves"].append(9)
    written["paid"].append(9.0)
    written["by"][0].append(["y", 1.0])
    written["stood"].clear()
    assert route.moves == (3, 3, 1, 0, 2) and route.paid_by[0] == (("x", 1.0),)
    assert route.paid == 4.5
    assert route.to_dict()["stood"] == [8]
    back = Route.from_dict(route.to_dict(), START)
    route.forget()
    assert back.moves == (3, 3, 1, 0, 2)


def test_no_way_is_a_way_with_no_moves() -> None:
    back = there_and_back(Route())
    assert back.moves == () and back.paid == 0.0 and back.begin(START) == ()


def good() -> dict[str, Any]:
    return a_way().to_dict()


def without(key: str) -> dict[str, Any]:
    data = good()
    del data[key]
    return data


def having(key: str, value: object) -> dict[str, Any]:
    data = good()
    data[key] = value
    return data


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        "way",
        without("moves"),
        without("paid"),
        without("by"),
        without("stood"),
        having("moves", (3, 3, 1, 0, 2)),
        having("moves", "33102"),
        having("paid", None),
        having("by", {}),
        having("stood", 8),
        having("moves", [3, 3, 1, 0]),  # pay for a move there is not
        having("paid", [1.0, 1.75, 2.75, 3.5, 4.5, 5.0]),
        having("by", [[], [], [], []]),
        having("moves", [3, 3, 1, 0, -1]),
        having("moves", [3, 3, 1, 0, 2.0]),
        having("moves", [3, 3, 1, 0, True]),
        having("moves", [3, 3, 1, 0, "2"]),
        having("paid", [1.0, 1.75, 2.75, 3.5, "4.5"]),
        having("paid", [1.0, 1.75, 2.75, 3.5, float("nan")]),
        having("paid", [1.0, 1.75, 2.75, 3.5, float("inf")]),
        having("paid", [1.0, 1.75, 2.75, 3.5, False]),
        having("by", [[], [], [], [], "x"]),
        having("by", [[], [], [], [], 7]),
        having("by", [[], [], [], [], None]),
        having("by", [[], [], [], [], ["x", 1.0]]),  # a pair, not a list of pairs
        having("by", [[], [], [], [], [("x", 1.0)]]),  # as the app holds it, not as written
        having("by", [[], [], [], [], [["x"]]]),
        having("by", [[], [], [], [], [["x", 1.0, 2.0]]]),
        having("by", [[], [], [], [], [[7, 1.0]]]),
        having("by", [[], [], [], [], [["x", "1.0"]]]),
        having("by", [[], [], [], [], [["x", float("nan")]]]),
        having("stood", [8, -1]),
        having("stood", [8.0]),
        having("stood", ["8"]),
        having("stood", [True]),
    ],
)
def test_what_is_not_a_way_is_refused(data: object) -> None:
    with pytest.raises(ConfigurationError):
        Route.from_dict(data, START)


def test_a_way_written_down_needs_its_start() -> None:
    with pytest.raises(ConfigurationError, match="starts from a saved spot"):
        Route.from_dict(good(), None)  # type: ignore[arg-type]
    assert Route().start is None
    route = a_way()
    route.forget()
    assert route.start is None


def test_whole_amounts_are_read_as_amounts() -> None:
    """A JSON file may hold 1 where 1.0 was written."""
    data = having("paid", [1, 2, 3, 4, 5])
    data["by"][0] = [["x", 1]]
    back = Route.from_dict(data, START)
    assert back.paid == 5.0 and isinstance(back.paid, float)
    assert back.paid_by[0] == (("x", 1.0),) and isinstance(back.paid_by[0][0][1], float)


# ------------------------------------------------------------ taking a way up (T-1065)


def another_way() -> dict[str, Any]:
    return {
        "moves": [1, 1, 4],
        "paid": [2.0, 4.0, 7.0],
        "by": [[["x", 2.0]], [["x", 2.0]], [["goal", 3.0]]],
        "stood": [20],
    }


def test_a_route_takes_up_a_way_written_down() -> None:
    """In place: a runner that holds the route plays the way taken up."""
    route = a_way()
    held = route
    route.take_up(another_way(), ("training", 2))
    assert held.moves == (1, 1, 4) and held.paid == 7.0 and held.start == ("training", 2)
    assert held.paid_by == ((("x", 2.0),), (("x", 2.0),), (("goal", 3.0),))
    assert held.to_dict() == another_way()  # nothing of the way before, nor its places
    assert held.begin(("training", 2)) == (1, 1, 4) and held.playing == 3
    # From the start the way before had, it is another spot's way and is forgotten.
    route.take_up(another_way(), ("training", 2))
    assert route.begin(START) == () and route.moves == ()


def test_a_route_with_nothing_takes_a_way_up() -> None:
    route = Route()
    route.take_up(a_way().to_dict(), START)
    assert route.moves == (3, 3, 1, 0, 2) and route.start == START
    assert route.begin(START) == (3, 3, 1, 0, 2)
    for move in (3, 3, 1, 0, 2, 2, 2, 2):
        route.did(move, 0.0)
    assert route.loss_stands(3) == 6  # the place a loss stood came with it


def test_taking_a_way_up_drops_the_go_that_was_being_played() -> None:
    route = a_way()
    route.begin(START)
    route.did(3, 1.0, (("x", 1.0),))
    route.did(0, 50.0, (("x", 49.0),))  # paid far more than either way
    assert route.story == (3, 0) and route.playing == 5
    route.take_up(another_way(), START)
    assert route.story == () and route.counted == 0.0 and route.playing == 0
    assert route.shortened == 0
    route.did(2, 60.0)  # no go has begun: nothing is added
    assert route.story == ()
    assert not route.end() and route.moves == (1, 1, 4) and route.paid == 7.0


def test_what_the_go_before_noted_of_the_old_way_does_not_remake_the_new() -> None:
    """A go that played the old way whole and was paid less would make it again from
    its own figures (T-1026): not the way taken up since."""
    route = a_way()
    route.begin(START)
    for move in route.moves:
        route.did(move, 0.5)  # all of the way, paid less than it was remembered for
    route.take_up(a_way().to_dict(), START)
    assert not route.end()
    assert route.moves == (3, 3, 1, 0, 2) and route.paid == 4.5 and route.shortened == 0


@pytest.mark.parametrize(
    "data", [None, [], without("moves"), having("paid", [1.0]), having("stood", [-1])]
)
def test_what_is_not_a_way_leaves_the_route_as_it_was(data: object) -> None:
    route = a_way()
    before = route.to_dict()
    route.begin(START)
    route.did(3, 1.0, (("x", 1.0),))
    with pytest.raises(ConfigurationError):
        route.take_up(data, ("training", 2))
    assert route.to_dict() == before and route.start == START
    assert route.story == (3,) and route.playing == 5
    route.did(3, 1.75)
    assert route.story == (3, 3)  # and the go goes on
    with pytest.raises(ConfigurationError, match="starts from a saved spot"):
        route.take_up(another_way(), None)  # type: ignore[arg-type]
    assert route.to_dict() == before
