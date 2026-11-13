"""T-1014: the bookkeeping of the best way played.

T-1015: what a go "counted" is what it was paid (the person's rewards added up),
no longer the new ground: a number that can be a fraction and can go down.
"""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training.route import Route

A, B, C = 1, 2, 3  # three moves
HERE = ("training", 1)  # a saved spot goes start from


def played(route: Route, steps: list[tuple[int | None, float]]) -> None:
    """Each step with what the story had been paid after it."""
    for move, paid in steps:
        route.did(move, paid)


def test_the_story_is_kept_up_to_where_it_last_got_somewhere_new() -> None:
    route = Route()
    assert (route.moves, route.paid, route.story, route.playing) == ((), 0, (), 0)
    assert route.begin(HERE) == ()
    # Four steps that get somewhere, then three that stand about.
    played(route, [(A, 0), (A, 1), (B, 2), (A, 3), (C, 3), (C, 3), (A, 3)])
    assert route.story == (A, A, B, A, C, C, A)
    assert route.end() is True
    assert (route.moves, route.paid) == ((A, A, B, A), 3)
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
    assert route.end() is False and (route.moves, route.paid) == ((A, A, A), 3)
    # Further: replaced, whole.
    route.begin(HERE)
    played(route, [(A, 1), (A, 2), (A, 3), (B, 3), (C, 4), (C, 4)])
    assert route.end() is True
    assert (route.moves, route.paid) == ((A, A, A, B, C), 4)
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
    assert route.end() and (route.moves, route.paid) == ((A, A, C, C, A), 5)
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
    assert (route.moves, route.paid) == ((A, A, A), 3)  # seven steps, less four
    assert route.failed(4) is False  # already cut there
    # The go carries on (it went back and found a way): it may yet get further.
    route.went_back(5)
    played(route, [(B, 3), (B, 4), (B, 5)])
    assert route.end() and (route.moves, route.paid) == ((A, A, B, B, B), 5)
    # A life lost after the way was played out cuts nothing of it.
    route.begin(HERE)
    played(route, [(A, 1), (A, 2), (B, 3), (B, 4), (B, 5), (C, 5), (C, 5), (C, 5), (C, 5), (C, 5)])
    assert route.failed(4) is False and len(route.moves) == 5
    # One lost on its first steps leaves nothing of it.
    route.begin(HERE)
    played(route, [(A, 1), (A, 2)])
    assert route.failed(4) is True and (route.moves, route.paid) == ((), 0)


def test_a_step_that_cannot_be_played_again_ends_the_story() -> None:
    route = Route()
    route.begin(HERE)
    played(route, [(A, 1), (B, 2), (None, 3), (A, 4), (A, 5)])
    assert route.story == (A, B)
    route.lost(1)  # over already: nothing more is taken off
    assert route.story == (A, B)
    assert route.end() and (route.moves, route.paid) == ((A, B), 2)
    # A go that was not put back to a saved spot plays nothing and leaves nothing.
    assert route.begin(None) == () and route.playing == 0
    played(route, [(C, 7), (C, 8), (C, 9)])
    assert route.story == () and route.end() is False
    assert (route.moves, route.paid) == ((A, B), 2)  # and what was remembered stays
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
    assert (route.moves, route.paid) == ((), 0) and route.begin(there) == ()
    # A move is a whole number, 0 or more; what was paid is any number (T-1015: pay
    # can be a fraction and can be taken away), but a number.
    nan, inf = float("nan"), float("inf")
    for move, paid in (
        (-1, 0),
        (True, 0),
        (1.5, 0),
        ("a", 0),
        (A, True),
        (A, "1"),
        (A, nan),
        (A, inf),
    ):
        with pytest.raises(ConfigurationError):
            route.did(move, paid)  # type: ignore[arg-type]
    for wrong in (-1, True, 2.0):
        with pytest.raises(ConfigurationError):
            route.went_back(wrong)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            route.lost(wrong)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            route.failed(wrong)  # type: ignore[arg-type]
    assert route.story == ()


def test_the_way_is_the_best_paid_go_up_to_where_it_was_paid_the_most() -> None:
    """T-1015: pay goes up and down, by fractions; the way ends at its highest."""
    route = Route()
    route.begin(HERE)
    assert route.counted == 0.0
    played(route, [(A, 1.5), (A, 0.5), (B, 3.0), (C, -7.0), (C, -6.5)])
    assert route.counted == -6.5
    assert route.end() and (route.moves, route.paid) == ((A, A, B), 3.0)
    # Going back takes the pay of the steps gone back over with it.
    route.begin(HERE)
    played(route, [(A, 1.5), (A, 0.5), (B, 3.0), (C, -7.0)])
    route.went_back(1)
    assert route.counted == 3.0
    route.end()
    # A go paid nothing above zero leaves no way: there is nothing the person pays for in it.
    never = Route()
    never.begin(HERE)
    played(never, [(A, -10.0), (A, -10.0), (B, -60.0), (B, -60.0)])
    assert never.end() is False and (never.moves, never.paid) == ((), 0.0)
    never.begin(HERE)
    played(never, [(A, 0.0), (B, 0.0)])
    assert never.end() is False and never.moves == ()


def test_a_go_that_played_the_whole_way_is_measured_against_this_go() -> None:
    """T-1015: a reward paid once only must not leave a way no later go can match."""
    route = Route()
    route.begin(HERE)
    played(route, [(A, 1.0), (A, 26.0), (B, 27.0)])  # 25 of it for a goal reached, once
    assert route.end() and route.paid == 27.0
    # The next go plays the way and is paid 2 for it this time, then gets 1 more.
    assert route.begin(HERE) == (A, A, B)
    played(route, [(A, 1.0), (A, 1.0), (B, 2.0), (C, 2.0), (C, 3.0), (C, 3.0)])
    assert route.end() is True
    assert (route.moves, route.paid) == ((A, A, B, C, C), 3.0)
    # Playing it again and being paid no more after it than at its end changes nothing,
    # and neither does being paid the most somewhere inside it.
    route.begin(HERE)
    played(route, [(A, 1.0), (A, 1.0), (B, 2.0), (C, 2.0), (C, 3.0), (A, 3.0), (A, 2.0)])
    assert route.end() is False and route.moves == (A, A, B, C, C)
    route.begin(HERE)
    played(route, [(A, 9.0), (A, 1.0), (B, 2.0), (C, 2.0), (C, 3.0), (A, 4.0)])
    assert route.end() is False and route.moves == (A, A, B, C, C)
    # A go that left the way (it went back and played something else) is measured
    # against what the way was paid when it was remembered.
    route.begin(HERE)
    played(route, [(A, 1.0), (A, 1.0), (B, 2.0)])
    route.went_back(2)
    played(route, [(C, 2.0), (C, 3.0)])
    assert route.end() is False and route.moves == (A, A, B, C, C)  # as much, not more
    route.begin(HERE)
    played(route, [(A, 1.0)])
    played(route, [(C, 2.0), (C, 3.5)])
    assert route.end() is True and (route.moves, route.paid) == ((A, C, C), 3.5)


def test_a_loss_close_behind_the_way_cuts_it_only_while_the_go_is_still_on_it() -> None:
    """T-1015: ``failed`` is also asked when a life goes after the way was played out."""
    route = Route()
    route.begin(HERE)
    played(route, [(A, float(n)) for n in range(1, 11)])
    assert route.end() and len(route.moves) == 10
    # The way played out, three steps of its own, and a life lost: four steps before
    # that is inside the way.
    route.begin(HERE)
    played(route, [(A, float(n)) for n in range(1, 11)] + [(B, 10.0), (B, 10.0), (B, 0.0)])
    assert route.failed(4) is True and (route.moves, route.paid) == ((A,) * 9, 9.0)
    # The story ends before the loss; once it is over nothing more is cut by it,
    # however many lives go after.
    route.lost(4)
    assert route.story == (A,) * 9
    assert route.failed(4) is False and route.failed(1) is False and len(route.moves) == 9
    # A go that had left the way by then cuts nothing of it.
    route.end()
    route.begin(HERE)
    played(route, [(A, 1.0), (A, 2.0), (C, 2.0), (C, 2.0), (C, 2.0), (C, 2.0), (C, -8.0)])
    assert route.failed(2) is False and len(route.moves) == 9
