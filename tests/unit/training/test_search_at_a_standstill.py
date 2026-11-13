"""T-1018: the search for a way on from where the AI stood still, with no life lost."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import search
from retro_ai_lab.training.search import Search, Way
from retro_ai_lab.training.trial import Outcome, play
from retro_ai_lab.training.ways import changes
from tests.unit.training import test_trial as trial
from tests.unit.training.test_search import ACROSS, ALIVE, DEAD, NOWHERE, look_for
from tests.unit.training.test_search_plain_first import scripted
from tests.unit.training.test_trial import FRAMES, LIVES, SPOT, Hopper, a_go_that_fell_in

hopper = trial.hopper  # the practice game behind a connection (pytest fixture)

Moves = tuple[int, ...]


class Wall:
    """Stands in for ``trial.play`` at a wall. Every play stays alive. A blind one
    ends on the screen that standing at the wall ends on if it is one move held, and
    otherwise unless its moves are among ``moving``: then on a screen of its own, or
    on one screen for all the moves among ``alike``. One under the watch gets nowhere
    new, unless its moves are ``over``."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        moving: Collection[Moves] | None = None,
        over: Moves | None = None,
        alike: Collection[Moves] = (),
    ) -> None:
        self.moving, self.over = moving, over  # moving=None: every play ends elsewhere
        self.alike = alike
        self.calls: list[tuple[Moves, bool]] = []
        self.ends: list[bool] = []  # whether each play was asked for the screen it ends on
        monkeypatch.setattr(search, "play", self)

    def __call__(
        self, connection: object, slot: str, moves: Sequence[int], space: object, **kw: object
    ) -> Outcome:
        played, seen = tuple(moves), kw.get("watch") is not None
        self.calls.append((played, seen))
        self.ends.append(bool(kw.get("end", False)))
        if seen:
            return ACROSS if played == self.over else NOWHERE
        elsewhere = len(set(played)) > 1 and (
            self.moving is None or played in self.moving or played == self.over
        )
        if not elsewhere:
            return Outcome(30, False, 0, screen=b"at the wall")
        return Outcome(30, False, 0, screen=b"alike" if played in self.alike else bytes(played))


def test_at_a_standstill_the_changes_to_the_plain_way_are_tried(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    back, on = hopper.move("left"), hopper.move("right")
    trace = [back] * 4 + [on] * 6
    base = (on,) * 10
    listed = [
        c.apply(base, 4, whole=True) for c in changes(base, hopper.space, early=True, whole=True)
    ]
    # Holding "on" stays alive and gets nowhere new: a wall. Of the changes to that,
    # the first and third end on the same screen; the second ends elsewhere and is
    # nowhere new under the watch; the fourth is over.
    wall = Wall(monkeypatch, moving=[listed[1]], over=listed[3])
    found = look_for(hopper, trace, object, carry_on=4, plain=on, stood=True)
    way = found.way
    change = list(changes(base, hopper.space, early=True, whole=True))[3]
    assert way == Way(listed[3], change, tries=8, ground=3, plain=on)
    assert way is not None and way.lost_at is None and back not in way.moves
    assert found == Search(way, 8)
    # The plain way blind and then under the watch, as after a loss. Each change blind
    # first; only those that end on another screen than the plain way did are looked at.
    held = (on,) * 14
    assert wall.calls == [
        (held, False),
        (held, True),
        (listed[0], False),
        (listed[1], False),
        (listed[1], True),
        (listed[2], False),
        (listed[3], False),
        (listed[3], True),
    ]
    # Every blind play was asked for the screen it ends on, and no other.
    assert wall.ends == [not seen for _, seen in wall.calls]
    # A change that ends on the screen an earlier one ended on, which was looked at
    # and had got nowhere, is not looked at either.
    wall = Wall(monkeypatch, moving=listed[:3], over=listed[3], alike=[listed[0], listed[2]])
    found = look_for(hopper, trace, object, carry_on=4, plain=on, stood=True)
    assert found.way == Way(listed[3], change, tries=9, ground=3, plain=on)
    assert wall.calls[2:] == [
        (listed[0], False),
        (listed[0], True),
        (listed[1], False),
        (listed[1], True),
        (listed[2], False),  # where the first ended: nowhere
        (listed[3], False),
        (listed[3], True),
    ]
    # After a life lost the same game is searched another way: the changes are to what
    # it did, each looked at if it stays alive, and no play is asked for its screen.
    theirs = [c.apply(trace, 4) for c in changes(trace, hopper.space)]
    wall = Wall(monkeypatch, moving=[], over=theirs[1])
    found = look_for(hopper, trace, object, carry_on=4, plain=on)
    assert found.way is not None and found.way.moves == theirs[1] and found.way.plain is None
    assert wall.calls == [
        (held, False),
        (held, True),
        (theirs[0], False),
        (theirs[0], True),
        (theirs[1], False),
        (theirs[1], True),
    ]
    assert wall.ends == [False] * 6


def test_a_plain_way_that_gets_on_is_still_the_way(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, back, on = scripted(hopper, monkeypatch, [ALIVE, ACROSS])
    found = look_for(hopper, [back] * 10, object, carry_on=4, plain=on, stood=True)
    held = (on,) * 14
    way = found.way
    assert way is not None and way.change.apply([on] * 10, 4) == held
    assert found == Search(Way(held, way.change, tries=2, ground=3, plain=on), 2)
    assert script.calls == [(held, False), (held, True)]


def test_then_the_changes_to_what_it_did_and_nothing_twice(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    back, on = hopper.move("left"), hopper.move("right")
    # Nothing gets on. What it did was a wander, so its changes are another list.
    wander = [back] * 3 + [on] * 3
    base = (on,) * 6
    plain = [
        c.apply(base, 4, whole=True) for c in changes(base, hopper.space, early=True, whole=True)
    ]
    theirs = [
        c.apply(wander, 4, whole=True)
        for c in changes(wander, hopper.space, early=True, whole=True)
    ]
    rest = [m for m in theirs if m not in set(plain)]
    held = (on,) * 10
    # Every change ends where standing still does: one blind play each, none looked at.
    wall = Wall(monkeypatch, moving=[])
    out = look_for(hopper, wander, object, carry_on=4, plain=on, stood=True, budget=10_000)
    assert out.way is None and not out.out_of_time
    moves = [played for played, _ in wall.calls]
    assert moves == [held, held, *plain, *rest] and len(set(moves[1:])) == len(moves) - 1
    assert out.tries == len(moves) and [seen for _, seen in wall.calls[2:]] == [False] * (
        len(moves) - 2
    )
    # The plain way's changes have half the tries, what it did the rest.
    wall = Wall(monkeypatch, moving=[])
    out = look_for(hopper, wander, object, carry_on=4, plain=on, stood=True, budget=8)
    assert out == Search(None, 8)
    assert [played for played, _ in wall.calls] == [held, held, *plain[:2], *rest[:4]]
    # Where every change ends somewhere else and is nowhere new, each costs two plays.
    wall = Wall(monkeypatch)
    out = look_for(hopper, wander, object, carry_on=4, plain=on, stood=True, budget=8)
    assert out == Search(None, 8)
    assert wall.calls == [
        (held, False),
        (held, True),
        (plain[0], False),
        (plain[0], True),
        (rest[0], False),
        (rest[0], True),
        (rest[1], False),
        (rest[1], True),
    ]
    # A plain way that loses a life (it stood at an edge): its changes up to there, as
    # after a loss. There is no screen of standing still to tell a change by, so each
    # that stays alive is looked at.
    script, back, on = scripted(hopper, monkeypatch, [DEAD, ALIVE, NOWHERE, ALIVE, ACROSS])
    out = look_for(hopper, wander, object, carry_on=4, plain=on, stood=True)
    short = (on,) * DEAD.steps
    first = list(changes(short, hopper.space, early=True, whole=True))
    assert out.way == Way(
        first[1].apply(short, 4, whole=True),
        first[1],
        tries=5,
        ground=3,
        plain=on,
        lost_at=DEAD.steps,
    )
    assert [seen for _, seen in script.calls] == [False, False, True, False, True]
    # With no plain way to go by (nothing has been paid yet), the same.
    script, back, on = scripted(hopper, monkeypatch, [ALIVE, NOWHERE, ALIVE, ACROSS])
    out = look_for(hopper, wander, object, carry_on=4, stood=True)
    assert out.way == Way(
        theirs[1], list(changes(wander, hopper.space, early=True, whole=True))[1], tries=4, ground=3
    )


def test_stood_must_be_true_or_false(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    script, _, on = scripted(hopper, monkeypatch, [ALIVE, ACROSS])
    for wrong in (None, 1, "yes"):
        with pytest.raises(ConfigurationError, match="stood"):
            look_for(hopper, [on] * 4, object, plain=on, stood=wrong)
    assert script.calls == []


def test_the_screen_a_try_ends_on(hopper: Hopper) -> None:
    right, left, jump = hopper.move("right"), hopper.move("left"), hopper.move("a")

    def tried(moves: Sequence[int], **kw: object) -> Outcome:
        return play(  # type: ignore[call-overload]
            hopper.connection,
            SPOT,
            moves,
            hopper.space,
            frames_per_step=FRAMES,
            lives=[LIVES],
            **kw,
        )

    # Not asked for, a try has no screen; asked for, the screen after its last move.
    assert tried([right] * 6).screen is None
    walked = tried([right] * 6, end=True)
    assert isinstance(walked.screen, bytes)
    assert len(walked.screen) == toy_hopper.SCREEN_WIDTH * toy_hopper.SCREEN_HEIGHT * 3
    # The same moves end on the same screen, and so do moves that come to the same
    # (a hop on the spot, landed again); moves that end elsewhere do not.
    assert tried([right] * 6, end=True).screen == walked.screen
    stood = tried([hopper.move()] * 20, end=True)
    assert tried([jump] * 2 + [hopper.move()] * 18, end=True).screen == stood.screen
    assert tried([left] * 6, end=True).screen != walked.screen != stood.screen
    # It is no part of what the try came to: with and without it, the same outcome.
    assert walked == tried([right] * 6) == Outcome(6, False, 0)
    # A try that loses the life has none: it did not play every move.
    watch, mark, died = a_go_that_fell_in(hopper)
    fell = tried([right] * (died + 5), end=True)
    assert fell.lost and fell.steps == died and fell.screen is None
    # Under a watch the screen is taken too, and the watch is shown its own looks only:
    # an odd count of moves ends between two looks.
    lent = watch.trial(mark)
    seen = tried([right] * 7, watch=lent, end=True)
    assert seen.screen == tried([right] * 7, end=True).screen and lent.looks == 4
    # A watched try also says where the watch last had the AI; one not watched cannot.
    assert seen.spot == lent.spot and walked.spot is None and fell.spot is None
    further = watch.trial(mark)
    assert tried([right] * 15, watch=further).spot == further.spot
    assert further.spot[0] > lent.spot[0] and further.spot[1] == lent.spot[1]


def test_a_way_on_must_be_enough_where_that_is_asked(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    hop = Outcome(steps=30, lost=False, ground=3, spot=(40, 0))  # past, and still about
    away = Outcome(steps=30, lost=False, ground=3, spot=(90, 0))
    assert hop.past and away.past and hop == away  # the spot is no part of the outcome
    asked: list[Outcome] = []

    def out_of_the_spot(seen: Outcome) -> bool:
        asked.append(seen)
        return seen.spot is not None and seen.spot[0] > 64

    # The plain way that hops is not it, nor a change that hops; the first that ends
    # out of the spot is. Only tries that stayed alive on new ground are asked about.
    nowhere = Outcome(steps=30, lost=False, ground=1, spot=(200, 0))
    _, back, on = scripted(
        hopper, monkeypatch, [ALIVE, hop, ALIVE, nowhere, ALIVE, hop, ALIVE, away]
    )
    trace = [back] * 6
    found = look_for(
        hopper, trace, object, carry_on=4, plain=on, stood=True, enough=out_of_the_spot
    )
    base = (on,) * 6
    listed = list(changes(base, hopper.space, early=True, whole=True))
    assert found.way == Way(
        listed[2].apply(base, 4, whole=True), listed[2], tries=8, ground=3, plain=on
    )
    assert [seen.spot for seen in asked] == [(40, 0), (40, 0), (90, 0)]
    # Nobody asking, the plain way was it: past the place is enough.
    _, back, on = scripted(hopper, monkeypatch, [ALIVE, hop])
    found = look_for(hopper, trace, object, carry_on=4, plain=on, stood=True)
    assert found.way is not None and found.way.moves == (on,) * 10 and found.tries == 2
    # The same holds without ``stood``, and it must be something to ask.
    asked.clear()
    _, back, on = scripted(hopper, monkeypatch, [ALIVE, hop, ALIVE, away])
    found = look_for(hopper, trace, object, carry_on=4, plain=on, enough=out_of_the_spot)
    theirs = list(changes(trace, hopper.space))
    assert found.way == Way(theirs[0].apply(trace, 4), theirs[0], tries=4, ground=3)
    for wrong in (0, 4, "far", True):
        with pytest.raises(ConfigurationError, match="enough"):
            look_for(hopper, trace, object, plain=on, enough=wrong)
