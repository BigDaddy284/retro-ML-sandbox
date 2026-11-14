"""T-1026: the remembered way is worth what it is paid now.

A way was remembered with what its go had been paid, and only a go paid more
replaced it. When something stopped paying (a guess the AI dropped, a goal reached
once), the old figure stood and no later go could match it. A go that plays the
whole way again and is paid less for it than the way was remembered for now makes
the way again: that go's figures, up to the step at which it had been paid the most.
"""

from __future__ import annotations

from retro_ai_lab.training.route import Route

A, B, C = 1, 2, 3  # three moves
HERE = ("training", 1)  # a saved spot goes start from


def played(route: Route, steps: list[tuple[int | None, float]]) -> None:
    """Each step with what the story had been paid after it."""
    for move, paid in steps:
        route.did(move, paid)


def remembering(steps: list[tuple[int, float]]) -> Route:
    """A route that remembers ``steps`` as its way: the last is the most it was paid."""
    route = Route()
    assert route.shortened == 0
    route.begin(HERE)
    played(route, list(steps))
    assert route.end() and route.moves == tuple(move for move, _ in steps)
    assert route.shortened == 0
    return route


# Six steps, paid 27: 20 of it by something that pays no longer (a clock, say).
WITH_A_CLOCK = [(A, 2.0), (A, 4.0), (B, 14.0), (B, 24.0), (C, 26.0), (C, 27.0)]
WAY = (A, A, B, B, C, C)
WITHOUT_IT = [(A, 2.0), (A, 4.0), (B, 4.0), (B, 4.0), (C, 6.0), (C, 7.0)]


def test_a_way_paid_less_when_played_again_takes_that_gos_figures() -> None:
    route = remembering(WITH_A_CLOCK)
    assert route.paid == 27.0
    # The next go plays it, is paid 7 for it, and gets nothing more.
    assert route.begin(HERE) == WAY
    played(route, [*WITHOUT_IT, (A, 7.0), (A, 7.0)])
    assert route.end() is False  # it did no better than the way
    assert (route.moves, route.paid, route.shortened) == (WAY, 7.0, 0)
    # So a go paid 8 is the better one now. Against the 27 none ever was.
    route.begin(HERE)
    played(route, [(B, 3.0), (B, 8.0), (B, 8.0)])
    assert route.end() is True
    assert (route.moves, route.paid) == ((B, B), 8.0)

    # The same go can be the one: it plays the way, goes back into it to get past a
    # place, and from there is paid more than the way is paid now.
    route = remembering(WITH_A_CLOCK)
    route.begin(HERE)
    played(route, [*WITHOUT_IT, (A, 7.0)])
    route.went_back(3)  # back to the fourth step of the way
    assert route.counted == 4.0
    played(route, [(A, 6.0), (A, 7.5), (B, 7.5)])
    assert route.end() is True
    assert (route.moves, route.paid, route.shortened) == ((A, A, B, B, A, A), 7.5, 0)
    # A second end() of that go: the new way is not held against what the old one
    # was paid in it.
    assert route.end() is False
    assert (route.moves, route.paid, route.shortened) == ((A, A, B, B, A, A), 7.5, 0)
    # Its twin is paid no more from there than the way is now: the way stays, with
    # this go's figures.
    route = remembering(WITH_A_CLOCK)
    route.begin(HERE)
    played(route, [*WITHOUT_IT, (A, 7.0)])
    route.went_back(3)
    played(route, [(A, 6.0), (A, 7.0), (B, 7.0)])
    assert route.end() is False
    assert (route.moves, route.paid, route.shortened) == (WAY, 7.0, 0)

    # A go that ends exactly at the way's end is one that played it whole.
    route = remembering(WITH_A_CLOCK)
    route.begin(HERE)
    played(route, WITHOUT_IT)
    assert route.end() is False and (route.moves, route.paid) == (WAY, 7.0)


def test_a_way_ends_where_it_is_still_paid_for() -> None:
    # The way's last four steps were paid by the clock alone: they pay nothing now.
    route = remembering([(A, 2.0), (A, 4.0), (B, 14.0), (B, 24.0), (B, 34.0), (B, 44.0)])
    route.begin(HERE)
    played(route, [(A, 2.0), (A, 4.0), (B, 4.0), (B, 4.0), (B, 4.0), (B, 4.0), (C, 4.0), (C, 4.0)])
    assert route.end() is False
    assert (route.moves, route.paid, route.shortened) == ((A, A), 4.0, 4)
    # The next go begins with the two steps, and nothing is said of a shortening then.
    assert route.begin(HERE) == (A, A) and route.playing == 2 and route.shortened == 0
    played(route, [(A, 2.0), (A, 4.0), (C, 4.0)])
    assert route.end() is False
    assert (route.moves, route.paid, route.shortened) == ((A, A), 4.0, 0)

    # Paid the most somewhere inside it, and less than remembered at its end: it ends
    # at the first step at which it had been paid that much.
    route = remembering([(A, 2.0), (A, 9.0), (B, 10.0), (B, 11.0), (C, 12.0)])
    route.begin(HERE)
    played(route, [(A, 2.0), (A, 9.0), (B, 3.0), (B, 9.0), (C, 8.0)])
    assert route.end() is False
    assert (route.moves, route.paid, route.shortened) == ((A, A), 9.0, 3)

    # It is the way's end that is held against what it was remembered for: paid less
    # there, it ends at its highest, even where that is more than it ever was.
    route = remembering([(A, 2.0), (A, 9.0), (B, 10.0), (B, 11.0), (C, 12.0)])
    route.begin(HERE)
    played(route, [(A, 2.0), (A, 13.0), (B, 3.0), (B, 9.0), (C, 8.0)])
    assert route.end() is False
    assert (route.moves, route.paid, route.shortened) == ((A, A), 13.0, 3)

    # A go that got further than the whole of it keeps the whole of it: the steps
    # that pay nothing now may be what gets it there.
    route = remembering([(A, 2.0), (A, 4.0), (B, 14.0), (B, 24.0)])
    route.begin(HERE)
    played(route, [(A, 2.0), (A, 4.0), (B, 4.0), (B, 4.0), (C, 4.0), (C, 5.0), (C, 5.0)])
    assert route.end() is True
    assert (route.moves, route.paid) == ((A, A, B, B, C, C), 5.0)

    # Paid nothing above zero when played again: there is no way left.
    for nothing in ([0.0, 0.0, 0.0], [-1.0, -1.0, -2.0], [0.0, -1.0, 0.0]):
        route = remembering([(A, 1.0), (A, 2.0), (B, 3.0)])
        route.begin(HERE)
        played(route, list(zip((A, A, B), nothing, strict=True)))
        assert route.end() is False
        assert (route.moves, route.paid, route.shortened) == ((), 0.0, 3)
        # The spot the goes start from is still known: the next go begins with
        # nothing, and what it is paid for becomes the way.
        assert route.begin(HERE) == () and route.shortened == 0
        played(route, [(C, 1.0), (C, 1.0)])
        assert route.end() is True and (route.moves, route.paid) == ((C,), 1.0)


def test_what_is_left_alone() -> None:
    kept = (WAY, 27.0, 0)

    def state(route: Route) -> tuple[tuple[int, ...], float, int]:
        return route.moves, route.paid, route.shortened

    # Paid as much when played again.
    route = remembering(WITH_A_CLOCK)
    route.begin(HERE)
    played(route, [*WITH_A_CLOCK, (A, 27.0)])
    assert route.end() is False and state(route) == kept
    # Paid more (a go that went back is not paid twice for ground it walks again, so
    # the go that found a way was often paid less for it than one that plays it
    # straight through): the figure is never raised.
    route.begin(HERE)
    played(route, [(move, paid + 10.0) for move, paid in WITH_A_CLOCK])
    assert route.end() is False and state(route) == kept
    # Paid as much at its end, and more somewhere inside it.
    route.begin(HERE)
    played(route, [(A, 2.0), (A, 40.0), (B, 14.0), (B, 24.0), (C, 26.0), (C, 27.0)])
    assert route.end() is False and state(route) == kept
    # Played in part only: the go stopped inside it.
    route.begin(HERE)
    played(route, WITHOUT_IT[:5])
    assert route.end() is False and state(route) == kept
    # Left after two steps, gone back, and its own moves played from there: what the
    # go is paid after a going back is not what the way is paid.
    route.begin(HERE)
    played(route, [(A, 2.0), (A, 4.0), (C, 4.0)])
    route.went_back(1)
    played(route, [(B, 4.0), (B, 4.0), (C, 6.0), (C, 7.0)])
    assert route.story == WAY
    assert route.end() is False and state(route) == kept
    # Left for other moves after three steps, and as long as the way in the end.
    route.begin(HERE)
    played(route, [(A, 2.0), (A, 4.0), (B, 4.0), (C, 4.0), (C, 4.0), (C, 4.0)])
    assert route.end() is False and state(route) == kept
    # Gone back to its start while still on it, and its first moves played again.
    route.begin(HERE)
    played(route, WITHOUT_IT[:3])
    route.went_back(3)
    played(route, [(A, 1.0), (A, 1.0), (B, 1.0)])
    assert route.end() is False and state(route) == kept
    # A step nobody can play again in the middle of it.
    route.begin(HERE)
    played(route, [(A, 2.0), (None, 2.0), (A, 2.0), (B, 2.0), (B, 2.0), (C, 2.0), (C, 2.0)])
    assert route.end() is False and state(route) == kept
    # A go that did not start from a saved spot plays nothing of it.
    assert route.begin(None) == ()
    played(route, WITHOUT_IT)
    assert route.end() is False and state(route) == kept
    # A go that was never ended leaves nothing behind for the next.
    assert route.begin(HERE) == WAY
    played(route, WITHOUT_IT)
    assert route.begin(HERE) == WAY
    played(route, [(C, 1.0)])
    assert route.end() is False and state(route) == kept

    # Cut for a lost life in the same go: it ends where the loss says, and keeps the
    # figures it had. Ten steps paid 10 each; played again they are paid 1 each.
    for own in ([(B, 10.0), (B, 10.0), (B, 0.0)], []):
        route = remembering([(A, 10.0 * n) for n in range(1, 11)])
        route.begin(HERE)
        played(route, [(A, float(n)) for n in range(1, 11)] + own)
        assert route.failed(4) is True
        cut = len(route.moves)
        assert cut == 6 + len(own)
        route.lost(4)
        assert route.end() is False
        assert state(route) == ((A,) * cut, 10.0 * cut, 0)
    # The same when the go had left the way just where the cut then falls.
    route = remembering([(A, 10.0 * n) for n in range(1, 11)])
    route.begin(HERE)
    played(route, [(A, float(n)) for n in range(1, 8)] + [(B, 7.0)] * 4)
    assert route.failed(4) is True and route.moves == (A,) * 7
    route.lost(4)
    assert route.end() is False and state(route) == ((A,) * 7, 70.0, 0)

    # A second end() of the go that made the way again changes nothing more.
    route = remembering(WITH_A_CLOCK)
    route.begin(HERE)
    played(route, [(A, 2.0), (A, 4.0), (B, 4.0), (B, 4.0), (C, 4.0), (C, 4.0)])
    assert route.end() is False and state(route) == ((A, A), 4.0, 4)
    assert route.end() is False and state(route) == ((A, A), 4.0, 4)
    # A route with no way has nothing to make again.
    empty = Route()
    empty.begin(HERE)
    played(empty, [(A, 0.0), (A, -1.0)])
    assert empty.end() is False and state(empty) == ((), 0.0, 0)
