"""T-1014: the bookkeeping of the furthest way played."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training.route import Route

A, B, C = 1, 2, 3  # three moves
HERE = ("training", 1)  # a saved spot goes start from


def played(route: Route, steps: list[tuple[int | None, int]]) -> None:
    for move, ground in steps:
        route.did(move, ground)


def test_the_story_is_kept_up_to_where_it_last_got_somewhere_new() -> None:
    route = Route()
    assert (route.moves, route.ground, route.story, route.playing) == ((), 0, (), 0)
    assert route.begin(HERE) == ()
    # Four steps that get somewhere, then three that stand about.
    played(route, [(A, 0), (A, 1), (B, 2), (A, 3), (C, 3), (C, 3), (A, 3)])
    assert route.story == (A, A, B, A, C, C, A)
    assert route.end() is True
    assert (route.moves, route.ground) == ((A, A, B, A), 3)
    # The next go from the same spot begins with them, and is told how many.
    assert route.begin(HERE) == (A, A, B, A) and route.playing == 4
    assert route.story == ()
    # A go that counted nothing leaves nothing.
    empty = Route()
    empty.begin(HERE)
    played(empty, [(A, 0), (B, 0)])
    assert empty.end() is False and empty.moves == ()
    # So does a go with no steps at all.
    empty.begin(HERE)
    assert empty.end() is False and empty.moves == ()


def test_only_a_story_that_got_further_replaces_the_way() -> None:
    route = Route()
    route.begin(HERE)
    played(route, [(A, 1), (A, 2), (A, 3)])
    assert route.end()
    # As far: kept as it was, even though the moves differ.
    route.begin(HERE)
    played(route, [(B, 1), (B, 2), (B, 3), (B, 3)])
    assert route.end() is False and route.moves == (A, A, A)
    # Less far: kept.
    route.begin(HERE)
    played(route, [(C, 1)])
    assert route.end() is False and (route.moves, route.ground) == ((A, A, A), 3)
    # Further: replaced, whole.
    route.begin(HERE)
    played(route, [(A, 1), (A, 2), (A, 3), (B, 3), (C, 4), (C, 4)])
    assert route.end() is True
    assert (route.moves, route.ground) == ((A, A, A, B, C), 4)
    # A second end() of the same go changes nothing.
    assert route.end() is False and route.moves == (A, A, A, B, C)


def test_going_back_and_a_loss_that_stands_shorten_the_story() -> None:
    route = Route()
    route.begin(HERE)
    played(route, [(A, 1), (A, 2), (B, 2), (B, 2), (B, 2)])
    # Put back three steps, and a way played from there.
    route.went_back(3)
    assert route.story == (A, A)
    played(route, [(C, 3), (C, 4), (A, 5), (A, 6), (A, 6)])
    assert route.story == (A, A, C, C, A, A, A)
    # A loss that stands: the story ends two steps before it, and takes no more.
    route.lost(2)
    assert route.story == (A, A, C, C, A)
    played(route, [(B, 9), (B, 10)])
    route.went_back(1)  # nothing to do: the story is over
    assert route.story == (A, A, C, C, A)
    assert route.end() and (route.moves, route.ground) == ((A, A, C, C, A), 5)
    # A lead longer than the story leaves none of it.
    short = Route()
    short.begin(HERE)
    played(short, [(A, 1), (A, 2)])
    short.lost(30)
    assert short.story == () and short.end() is False
    # Going back further than the story is a mistake of the caller's.
    wrong = Route()
    wrong.begin(HERE)
    played(wrong, [(A, 1)])
    with pytest.raises(ConfigurationError):
        wrong.went_back(2)
    wrong.went_back(1)
    assert wrong.story == ()


def test_a_life_lost_while_it_was_played_cuts_the_way() -> None:
    route = Route()
    route.begin(HERE)
    played(route, [(A, n) for n in range(1, 11)])
    assert route.end() and len(route.moves) == 10
    # The next go plays it and loses a life on its seventh step.
    route.begin(HERE)
    played(route, [(A, n) for n in range(1, 8)])
    assert route.failed(4) is True
    assert (route.moves, route.ground) == ((A, A, A), 3)  # seven steps, less four
    assert route.failed(4) is False  # already cut there
    # The go carries on (it went back and found a way): it may yet get further.
    route.went_back(5)
    played(route, [(B, 3), (B, 4), (B, 5)])
    assert route.end() and (route.moves, route.ground) == ((A, A, B, B, B), 5)
    # A life lost after the way was played out cuts nothing of it.
    route.begin(HERE)
    played(route, [(A, 1), (A, 2), (B, 3), (B, 4), (B, 5), (C, 5), (C, 5), (C, 5), (C, 5), (C, 5)])
    assert route.failed(4) is False and len(route.moves) == 5
    # One lost on its first steps leaves nothing of it.
    route.begin(HERE)
    played(route, [(A, 1), (A, 2)])
    assert route.failed(4) is True and (route.moves, route.ground) == ((), 0)


def test_a_step_that_cannot_be_played_again_ends_the_story() -> None:
    route = Route()
    route.begin(HERE)
    played(route, [(A, 1), (B, 2), (None, 3), (A, 4), (A, 5)])
    assert route.story == (A, B)
    route.lost(1)  # over already: nothing more is taken off
    assert route.story == (A, B)
    assert route.end() and (route.moves, route.ground) == ((A, B), 2)
    # A go that was not put back to a saved spot plays nothing and leaves nothing.
    assert route.begin(None) == () and route.playing == 0
    played(route, [(C, 7), (C, 8), (C, 9)])
    assert route.story == () and route.end() is False
    assert (route.moves, route.ground) == ((A, B), 2)  # and what was remembered stays
    assert route.begin(HERE) == (A, B)


def test_another_start_forgets_the_way_and_nonsense_is_refused() -> None:
    route = Route()
    route.begin(HERE)
    played(route, [(A, 1), (A, 2)])
    assert route.end()
    # The spot goes start from was saved anew: what led on from the old one is no use.
    there = ("training", 2)
    assert route.begin(there) == () and route.playing == 0 and route.moves == ()
    played(route, [(B, 1)])
    assert route.end() and route.begin(there) == (B,)
    route.forget()
    assert (route.moves, route.ground) == ((), 0) and route.begin(there) == ()
    for move, ground in ((-1, 0), (True, 0), (1.5, 0), ("a", 0), (A, -1), (A, True), (A, 0.5)):
        with pytest.raises(ConfigurationError):
            route.did(move, ground)  # type: ignore[arg-type]
    for wrong in (-1, True, 2.0):
        with pytest.raises(ConfigurationError):
            route.went_back(wrong)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            route.lost(wrong)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            route.failed(wrong)  # type: ignore[arg-type]
    assert route.story == ()
