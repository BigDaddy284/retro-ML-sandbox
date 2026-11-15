"""T-1043: the plain way has half of a search's time, as it has half its tries."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retro_ai_lab.training import search
from retro_ai_lab.training.search import PLAIN_SHARE, TOP, Search, Way
from retro_ai_lab.training.trial import Outcome
from retro_ai_lab.training.ways import Change, changes
from tests.unit.training import test_trial as trial
from tests.unit.training.test_search import ACROSS, ALIVE, DEAD, NOWHERE, Clock, Script, look_for
from tests.unit.training.test_search_carries_on import spot_of
from tests.unit.training.test_trial import Hopper, a_go_that_fell_in

hopper = trial.hopper  # the practice game behind a connection (pytest fixture)

Play = tuple[tuple[int, ...], bool]  # the moves of a play, and whether the screen was looked at


def lost_at(step: int) -> Outcome:
    return Outcome(steps=step, lost=True, ground=0)


class Timed:
    """The plays scripted, on a clock that moves a second with every play (or ``pace``)."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[Outcome], pace: float = 1.0
    ) -> None:
        self.script = Script(outcomes)
        self.pace = pace
        monkeypatch.setattr(search, "play", self.script)

    @property
    def calls(self) -> list[Play]:
        return self.script.calls

    def __call__(self) -> float:
        return self.pace * len(self.script.calls)


class Setting:
    """What these tests search with: what it did was "back", ten steps; the plain way
    is "on", and holding it loses the life at its sixth step."""

    def __init__(self, hopper: Hopper) -> None:
        self.hopper = hopper
        self.back, self.on = hopper.move("left"), hopper.move("right")
        self.trace = [self.back] * 10
        self.base = (self.on,) * 6
        self.plain = list(changes(self.base, hopper.space))
        self.did = list(changes(self.trace, hopper.space))
        self.held: Play = ((self.on,) * 14, False)

    def of_plain(self, some: Sequence[Change], looked: bool = False) -> list[Play]:
        return [(change.apply(self.base, 4), looked) for change in some]

    def of_did(self, some: Sequence[Change], looked: bool = False) -> list[Play]:
        return [(change.apply(self.trace, 4), looked) for change in some]

    def look(self, clock: Timed | None = None, **kw: object) -> Search:
        if clock is not None:
            kw["clock"] = clock
        return look_for(self.hopper, self.trace, object, carry_on=4, plain=self.on, **kw)


def test_with_no_time_limit_it_is_the_search_it_was(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert PLAIN_SHARE == 0.5
    s = Setting(hopper)
    assert len(s.plain) == 8 and len(s.did) > 12
    # Twenty tries: the plain way and its whole list (fewer than its ten), then
    # the changes to what it did.
    timed = Timed(monkeypatch, [lost_at(6)] + [DEAD] * 19)
    out = s.look(budget=20)
    assert out == Search(None, 20) and not out.out_of_time
    assert timed.calls == [s.held, *s.of_plain(s.plain), *s.of_did(s.did[:11])]
    # And with tries enough for neither list: half for the plain way, as ever.
    timed = Timed(monkeypatch, [lost_at(6)] + [DEAD] * 7)
    out = s.look(budget=8)
    assert out == Search(None, 8)
    assert timed.calls == [s.held, *s.of_plain(s.plain[:3]), *s.of_did(s.did[:4])]


def test_the_plain_way_has_half_the_time(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    s = Setting(hopper)
    # Eight seconds, a second a play, and tries for everything. The plain way and
    # three of its changes take four; no fourth is started. Four changes to what it
    # did have the rest, and then the search is out of time.
    timed = Timed(monkeypatch, [lost_at(6)] + [DEAD] * 30)
    out = s.look(timed, seconds=8)
    assert out == Search(None, 8, out_of_time=True)
    assert timed.calls == [s.held, *s.of_plain(s.plain[:3]), *s.of_did(s.did[:4])]
    # Where each list stopped is said, and the next search carries on in both.
    assert out.stopped_at == (spot_of(s.plain[3], s.base), spot_of(s.did[4], s.trace))
    timed = Timed(monkeypatch, [lost_at(6)] + [DEAD] * 30)
    again = s.look(timed, seconds=8, begin_at=out.stopped_at)
    assert timed.calls == [s.held, *s.of_plain(s.plain[2:5]), *s.of_did(s.did[4:8])]
    assert again.stopped_at == (spot_of(s.plain[5], s.base), spot_of(s.did[8], s.trace))
    # The half is of the time the search may take, whatever that is.
    for seconds, of_plain, of_did in ((4, 1, 2), (6, 2, 3), (12, 5, 6), (5, 2, 2), (3, 1, 1)):
        timed = Timed(monkeypatch, [lost_at(6)] + [DEAD] * 30)
        out = s.look(timed, seconds=seconds)
        assert out == Search(None, seconds, out_of_time=True)
        assert timed.calls == [s.held, *s.of_plain(s.plain[:of_plain]), *s.of_did(s.did[:of_did])]


def test_the_half_is_of_the_time_not_of_the_plays(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Setting(hopper)
    # Half a second a play, and five seconds: the plain way and four of its changes
    # are started before two and a half have passed, and five changes to what it did
    # in the rest.
    timed = Timed(monkeypatch, [lost_at(6)] + [DEAD] * 30, pace=0.5)
    out = s.look(timed, seconds=5)
    assert out == Search(None, 10, out_of_time=True)
    assert timed.calls == [s.held, *s.of_plain(s.plain[:4]), *s.of_did(s.did[:5])]
    # Two seconds a play, and eight: the plain way and one change, and two to what it did.
    timed = Timed(monkeypatch, [lost_at(6)] + [DEAD] * 30, pace=2.0)
    out = s.look(timed, seconds=8)
    assert out == Search(None, 4, out_of_time=True)
    assert timed.calls == [s.held, *s.of_plain(s.plain[:1]), *s.of_did(s.did[:2])]
    # A play that takes longer than the plain way's whole time: its list is not begun.
    timed = Timed(monkeypatch, [lost_at(6)] + [DEAD] * 30, pace=5.0)
    out = s.look(timed, seconds=8)
    assert out == Search(None, 2, out_of_time=True)
    assert timed.calls == [s.held, *s.of_did(s.did[:1])]
    assert out.stopped_at == (TOP[0], spot_of(s.did[1], s.trace))


def test_a_change_to_the_plain_way_is_not_looked_at_after_its_time(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Setting(hopper)
    # The third change to the plain way stays alive in its blind play, which ends as
    # the plain way's four seconds do: it is not looked at. It is where that list
    # stopped, so the next search at the place begins with it.
    timed = Timed(monkeypatch, [lost_at(6), DEAD, DEAD, ALIVE] + [DEAD] * 30)
    out = s.look(timed, seconds=8)
    assert out == Search(None, 8, out_of_time=True)
    assert timed.calls == [s.held, *s.of_plain(s.plain[:3]), *s.of_did(s.did[:4])]
    assert out.stopped_at[0] == spot_of(s.plain[2], s.base)
    # With a second more it is looked at, and it is the way.
    timed = Timed(monkeypatch, [lost_at(6), DEAD, DEAD, ALIVE, ACROSS])
    out = s.look(timed, seconds=10)
    way = Way(s.plain[2].apply(s.base, 4), s.plain[2], 5, 3, plain=s.on, lost_at=6)
    assert out == Search(way, 5) and not out.out_of_time
    assert timed.calls[-2:] == [*s.of_plain([s.plain[2]]), *s.of_plain([s.plain[2]], looked=True)]
    # Looked at and nowhere new, it is no way, and the look counts against the time.
    timed = Timed(monkeypatch, [lost_at(6), DEAD, ALIVE, NOWHERE] + [DEAD] * 30)
    out = s.look(timed, seconds=8)
    assert out == Search(None, 8, out_of_time=True)
    assert timed.calls == [
        s.held,
        *s.of_plain(s.plain[:2]),
        *s.of_plain([s.plain[1]], looked=True),
        *s.of_did(s.did[:4]),
    ]


def test_the_plain_ways_time_being_up_is_not_the_search_out_of_time(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Setting(hopper)
    # A way among the changes to what it did, found after the plain way's time was up.
    timed = Timed(monkeypatch, [lost_at(6), DEAD, DEAD, DEAD, DEAD, ALIVE, ACROSS])
    out = s.look(timed, seconds=8)
    way = Way(s.did[1].apply(s.trace, 4), s.did[1], 7, 3)
    assert out == Search(way, 7) and not out.out_of_time
    assert timed.calls == [
        s.held,
        *s.of_plain(s.plain[:3]),
        *s.of_did(s.did[:2]),
        *s.of_did([s.did[1]], looked=True),
    ]
    assert out.stopped_at == (spot_of(s.plain[3], s.base), spot_of(s.did[1], s.trace))
    # The loss was settled early (T-1031): only the first step of what it did may be
    # changed, six changes in all. They are through before the time is up: nothing
    # found, and not for want of time.
    few = list(changes(s.trace, hopper.space, upto=1))
    assert len(few) == 6
    timed = Timed(monkeypatch, [lost_at(10)] + [DEAD] * 30)
    out = s.look(timed, seconds=12, live=1)
    assert out == Search(None, 12) and not out.out_of_time
    base = (s.on,) * 10
    its = list(changes(base, hopper.space, upto=1))
    assert len(its) == 6
    assert timed.calls == [
        s.held,
        *[(change.apply(base, 4), False) for change in its[:5]],
        *s.of_did(few),
    ]
    # The plain way's list stopped for its time where it had got to; the other was
    # gone through whole and is where it began.
    assert out.stopped_at == (spot_of(its[5], base, 1), TOP[1])


def test_half_the_tries_is_still_its_most(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    s = Setting(hopper)
    # Time for everything, and six tries: three for the plain way.
    timed = Timed(monkeypatch, [lost_at(6)] + [DEAD] * 30)
    out = s.look(timed, seconds=100, budget=6)
    assert out == Search(None, 6) and not out.out_of_time
    assert timed.calls == [s.held, *s.of_plain(s.plain[:2]), *s.of_did(s.did[:3])]
    # Twelve tries and eight seconds: the time comes first, for both.
    timed = Timed(monkeypatch, [lost_at(6)] + [DEAD] * 30)
    out = s.look(timed, seconds=8, budget=12)
    assert out == Search(None, 8, out_of_time=True)
    assert timed.calls == [s.held, *s.of_plain(s.plain[:3]), *s.of_did(s.did[:4])]


def test_a_plain_way_that_stays_alive_is_as_it_was(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Setting(hopper)
    # Alive and on new ground: the way, in two plays.
    timed = Timed(monkeypatch, [ALIVE, ACROSS])
    out = s.look(timed, seconds=2)
    assert out.way is not None and out.way.change == Change(0, s.on, 10) and out.tries == 2
    # Alive and nowhere new: it has no list, and what it did has the time that is left.
    timed = Timed(monkeypatch, [ALIVE, NOWHERE] + [DEAD] * 30)
    out = s.look(timed, seconds=8)
    assert out == Search(None, 8, out_of_time=True)
    assert timed.calls == [s.held, ((s.on,) * 14, True), *s.of_did(s.did[:6])]


def test_at_a_standstill_the_plain_way_keeps_the_time_it_had(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = Setting(hopper)
    # It stood still and holding the plain move stands still too: the changes to the
    # plain way are tried first on purpose (T-1018), for as long as the search may take.
    timed = Timed(monkeypatch, [ALIVE, NOWHERE] + [DEAD] * 30)
    out = look_for(
        hopper, s.trace, object, carry_on=4, plain=s.on, stood=True, seconds=8, clock=timed
    )
    assert out == Search(None, 8, out_of_time=True)
    base = (s.on,) * 10
    listed = list(changes(base, hopper.space, early=True, whole=True))
    held: Play = ((s.on,) * 14, False)
    assert timed.calls == [
        held,
        ((s.on,) * 14, True),
        *[(change.apply(base, 4, whole=True), False) for change in listed[:6]],
    ]


# ------------------------------------------------------------ on the practice game's first gap


def test_what_it_did_gets_its_turn_on_the_practice_game(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    right, run = hopper.move("right"), hopper.move("b", "right")
    trace = [right] * died
    # The plain way here is running: it falls into the gap sooner, and none of the
    # first ten changes to it gets over. The way is a change to what it did (walking),
    # the seventeenth of that list.
    clock = Clock(monkeypatch)  # a second passes with every play
    found = look_for(hopper, trace, lambda: watch.trial(mark), plain=run, seconds=40, clock=clock)
    assert found.way is not None and found.way.plain is None
    assert found.way.change == list(changes(trace, hopper.space))[16]
    # Twenty seconds for the plain way and its changes, and then what it did.
    assert clock.plays[:20] == [False] * 20 and found.tries == 20 + 18
    assert not found.out_of_time
