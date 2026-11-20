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
    written["by"][0].append(["y", 1.0])
    written["stood"].clear()
    assert route.moves == (3, 3, 1, 0, 2) and route.paid_by[0] == (("x", 1.0),)
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
