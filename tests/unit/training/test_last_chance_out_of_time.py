"""T-1032: a go's asking for the last chance costs less, and its time running out is
not for nothing."""

from __future__ import annotations

import itertools
import logging

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import episode
from retro_ai_lab.training.chance import Chance
from retro_ai_lab.training.episode import NOW_SLOT
from retro_ai_lab.training.search import Search
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import SLOTS, Lab, steps
from tests.unit.training.test_last_chance_in_a_go import (
    GAP_ENDS,
    LOG,
    LONG,
    LONG_DIED,
    Ticking,
    going,
    in_order,
    said,
)

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

BEFORE = 87  # plays the asking of this go took before this task (T-1031)


class Asking:
    """Stands before ``last_chance`` in the runner: notes what each asking was given and
    what it said, can cut one short after so many plays, and can answer None for one."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        cut: dict[int, int] | None = None,
        nothing: tuple[int, ...] = (),
    ) -> None:
        self.real = episode.last_chance
        self.cut = cut or {}
        self.nothing = nothing
        self.calls: list[dict[str, object]] = []
        monkeypatch.setattr(episode, "last_chance", self)

    def __call__(
        self, connection: object, slot: str, trace: tuple[int, ...], space: object, **kw: object
    ) -> Chance | None:
        number = len(self.calls)
        note: dict[str, object] = {
            "slot": slot,
            "steps": len(trace),
            "shut": kw.get("shut"),
            "partial": kw.get("partial"),
        }
        self.calls.append(note)
        if number in self.nothing:
            note["said"] = None
            return None
        if number in self.cut:
            # A clock of its own that moves a second each time it is read: read once to
            # begin with and once before each play, so this many plays are made.
            reads = itertools.count(1)
            kw = {**kw, "seconds": self.cut[number] + 0.5, "clock": lambda: float(next(reads))}
        got = self.real(connection, slot, trace, space, **kw)  # type: ignore[arg-type]
        note["said"] = got
        return got


class Searching:
    """Stands before ``find_way`` in the runner: notes what each search was told."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, finds: bool = True) -> None:
        self.real = episode.find_way
        self.finds = finds
        self.calls: list[tuple[str, int, object]] = []
        monkeypatch.setattr(episode, "find_way", self)

    def __call__(
        self, connection: object, slot: str, trace: tuple[int, ...], *args: object, **kw: object
    ) -> Search:
        self.calls.append((slot, len(trace), kw["live"]))
        if not self.finds:
            return Search(None, 0)
        return self.real(connection, slot, trace, *args, **kw)  # type: ignore[arg-type]


def test_an_older_moment_is_asked_about_fewer_steps(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asking = Asking(monkeypatch)
    searching = Searching(monkeypatch)
    r, moments = going(lab, LONG, last_chance=True)
    r.begin()
    steps(r, LONG_DIED - 1)
    start, second, third, _ = moments.kept
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    # The moment 34 steps back is too late from its first step. So the one before it,
    # 20 steps older, is too late from its 20th step on at the latest, and is asked
    # about the steps before that only.
    first, then = asking.calls
    assert (first["slot"], first["steps"], first["shut"]) == (third.slot, 34, None)
    assert first["said"] == Chance(0, 15)
    assert (then["slot"], then["steps"], then["shut"]) == (second.slot, 54, 20)
    assert first["partial"] is True and then["partial"] is True
    said_then = then["said"]
    assert isinstance(said_then, Chance) and (said_then.live, said_then.sure) == (1, True)
    plays = 15 + said_then.plays
    assert plays == 73 < BEFORE  # four steps asked in full where it was five
    # The same step is found, the same moment gone back to, and the same way.
    assert searching.calls == [(start.slot, 74, 21)]
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    settled, found = said(caplog)
    assert settled.startswith(
        "Lost a life at step 74. That was settled 53 steps before: nothing pressed from "
        f"then on changes it ({plays} plays, "
    )
    assert settled.endswith(
        " to find out). The AI goes back 74 steps, not 34, and tries its changes before then."
    )
    assert found.endswith(
        "holding right instead of what it did, with right + a for 4 steps, starting 54 "
        "steps before that would lose the life."
    )
    while lab.game.state.x <= GAP_ENDS:
        assert r.step()
    assert lab.game.state.lives == toy_hopper.START_LIVES
    r.finish()

    # With three moments kept the oldest is the one asked so: it has one step before the
    # edge, as before, found in fewer plays.
    asking.calls.clear()
    searching.calls.clear()
    r, moments = going(lab, LONG, SLOTS, last_chance=True)
    r.begin()
    steps(r, LONG_DIED)
    assert [(c["slot"], c["steps"], c["shut"]) for c in asking.calls] == [
        ("back2", 34, None),
        ("back1", 54, 20),
    ]
    assert searching.calls == [("back1", 54, 1)]
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    r.finish()


def test_out_of_time_it_goes_back_to_before_what_it_knows(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The second asking is cut short after ten plays: what it did, the first step (not
    # too late), and eight of the fourteen moves at step 10. Nothing new is known, but
    # what the first moment showed stands: too late from step 20 on at the latest.
    asking = Asking(monkeypatch, cut={1: 10})
    searching = Searching(monkeypatch)
    r, moments = going(lab, LONG, last_chance=True)
    r.begin()
    steps(r, LONG_DIED - 1)
    start, second, third, _ = moments.kept
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert [c["said"] for c in asking.calls] == [Chance(0, 15), Chance(20, 10, sure=False)]
    # How many of those twenty steps are not too late nobody knows (it is one). So the
    # moment before is taken, which has twenty more for certain, and the changes are
    # tried before step 40 of what it did from there: the last 34 steps are left out.
    assert searching.calls == [(start.slot, 74, 40)]
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    assert in_order(lab.loaded) == [third.slot, second.slot, start.slot]
    settled, found = said(caplog)
    assert settled.startswith(
        "Lost a life at step 74. That was settled at least 34 steps before: nothing pressed "
        "from then on changes it (25 plays, "
    )
    assert settled.endswith(
        "; the time a search may take ran out before the AI could find the step). The AI "
        "goes back 74 steps, not 34, and tries its changes before then."
    )
    assert "Went back 74 steps and found a way past on try" in found
    while lab.game.state.x <= GAP_ENDS:
        assert r.step()
    assert lab.game.state.lives == toy_hopper.START_LIVES
    r.finish()

    # Cut short later, more is known: step 10 was found too late (16 plays), and four
    # moves at step 5 were asked when the time ran out. Too late from step 10 on.
    caplog.clear()
    asking = Asking(monkeypatch, cut={1: 20})
    searching.calls.clear()
    r, moments = going(lab, LONG, last_chance=True)
    r.begin()
    steps(r, LONG_DIED - 1)
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert [c["said"] for c in asking.calls] == [Chance(0, 15), Chance(10, 20, sure=False)]
    assert searching.calls == [(start.slot, 74, 30)]
    assert r.ways_found == 1
    assert "That was settled at least 44 steps before" in said(caplog)[0]
    r.finish()

    # No moment before it is kept: it searches from that one, before what it knows.
    caplog.clear()
    asking = Asking(monkeypatch, cut={1: 10})
    searching = Searching(monkeypatch, finds=False)
    r, moments = going(lab, LONG, SLOTS, last_chance=True)
    r.begin()
    steps(r, LONG_DIED - 1)
    oldest = moments.kept[0]
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert searching.calls == [(oldest.slot, 54, 20)]
    assert said(caplog)[0].endswith(
        "The AI goes back 54 steps, not 34, and tries its changes before then."
    )
    assert lab.loaded[-1] == NOW_SLOT and (r.ways_sought, r.ways_found) == (1, 0)
    r.finish()

    # In time, nothing is unsure and nothing says "at least".
    caplog.clear()
    asking = Asking(monkeypatch)
    searching = Searching(monkeypatch)
    r, _ = going(lab, LONG, last_chance=True)
    r.begin()
    with caplog.at_level(logging.INFO, logger=LOG):
        steps(r, LONG_DIED)
    assert not any("at least" in line or "ran out" in line for line in said(caplog))
    r.finish()


def test_out_of_time_with_nothing_known_is_the_search_it_was(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The first asking is cut short after five plays: nothing was found too late yet,
    # and no moment after it said anything. The runner's own clock (a second a reading)
    # says the time is up, with a second and a half allowed.
    asking = Asking(monkeypatch, cut={0: 5})
    searching = Searching(monkeypatch, finds=False)
    monkeypatch.setattr(episode, "time", Ticking())
    r, moments = going(lab, LONG, last_chance=True, way_seconds=1.5)
    r.begin()
    steps(r, LONG_DIED - 1)
    third = moments.kept[2]
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert [c["said"] for c in asking.calls] == [None]
    assert lab.loaded.count(third.slot) == 5
    assert searching.calls == [(third.slot, 34, None)]
    assert said(caplog)[0] == (
        "Lost a life at step 74. Finding from when on that was settled took longer than "
        "the time a search may take, so the AI looks for a way past as it did before."
    )
    assert lab.loaded[-1] == NOW_SLOT
    r.finish()


def test_a_replay_that_does_not_lose_the_life_stays_with_the_first_moment(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The second moment's asking says nothing, and not for want of time: what it did,
    # played again from there, did not lose the life on the same step. What the first
    # moment showed is no reason to search from a moment that cannot be played again.
    asking = Asking(monkeypatch, nothing=(1,))
    searching = Searching(monkeypatch, finds=False)
    r, moments = going(lab, LONG, last_chance=True, way_seconds=1000.0)
    r.begin()
    steps(r, LONG_DIED - 1)
    third = moments.kept[2]
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert [c["said"] for c in asking.calls] == [Chance(0, 15), None]
    assert searching.calls == [(third.slot, 34, 0)]
    assert said(caplog)[0].startswith(
        "Lost a life at step 74. That was settled before the earliest moment the AI can "
        "go back to, 34 steps before: nothing pressed since changes it (15 plays, "
    )
    r.finish()
