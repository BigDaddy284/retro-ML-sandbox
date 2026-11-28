"""T-1072: the runner remembers what got past a place, and its next search about there
tries that first.

The runner keeps, for each place a search found a way past, the change that did it,
counted from the loss it was made before (:class:`Known`), and hands it to the next
search about there. The log says so where that change got past again.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pytest

from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.search import TOP, Known, Search, Way
from retro_ai_lab.training.ways import Change
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_looking_ahead import THING, Near, looking
from tests.unit.training.test_search_at_a_loss import searching
from tests.unit.training.test_standing_still import FAR, hop_about, standing

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LOG = "retro_ai_lab.training.episode"
REAL = "the real search"
NOTHING = Search(None, 5)
AGAIN = " That is the change that got past about here before, tried first."


class Asked:
    """Stands in for ``find_way`` in the runner: notes what each search was told got
    past before, what it was given and what it came to, and answers from a list (the
    real search where the list says so, and once it is used up)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, answers: Sequence[object] = ()) -> None:
        self.known: list[Known | None] = []
        self.traces: list[tuple[int, ...]] = []
        self.stood: list[bool] = []
        self.out: list[Search] = []
        self.answers = list(answers)
        self.real = episode.find_way
        monkeypatch.setattr(episode, "find_way", self)

    def __call__(
        self, connection: Any, slot: str, trace: Sequence[int], *a: Any, **kw: Any
    ) -> Search:
        self.known.append(kw["known"])
        self.traces.append(tuple(trace))
        self.stood.append(kw["stood"])
        out = self.answers.pop(0) if self.answers else REAL
        if callable(out):
            out = out(tuple(trace))
        if not isinstance(out, Search):
            out = self.real(connection, slot, trace, *a, **kw)
        self.out.append(out)
        return out

    def told_from(self, number: int) -> Known:
        """What search ``number`` found, as the next search about there is told it."""
        way = self.out[number].way
        assert way is not None and way.first is None
        end = len(self.traces[number]) if way.plain is None else way.lost_at
        assert end is not None
        return Known(way.change.move, way.change.hold, end - way.change.at, way.plain)


def said(caplog: pytest.LogCaptureFixture, part: str = "") -> list[str]:
    lines = [rec.getMessage() for rec in caplog.records if rec.name == LOG]
    return [line for line in lines if part in line]


def a_go(lab: Lab, r: EpisodeRunner) -> None:
    """One go that walks right from the start for as long as the game goes on."""
    lab.game.reset()
    r.begin()
    while r.step():
        pass
    r.finish()


def to_the_gap(lab: Lab, r: EpisodeRunner) -> None:
    """A go that walks right from the start up to the step the life goes at the first
    gap, and through the search made there."""
    lab.game.reset()
    r.begin()
    steps(r, DIED_AT)


def test_the_next_search_about_a_place_is_told_what_got_past_there(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The first search finds the way over the gap. Over it a life goes somewhere
    # else, to a search for nothing; put back to the start, the go comes to the gap
    # a life down.
    asked = Asked(monkeypatch, [REAL, NOTHING, REAL, NOTHING, NOTHING, REAL])
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    with caplog.at_level(logging.INFO, logger=LOG):
        a_go(lab, r)
        assert len(asked.known) == 3 and asked.stood == [False] * 3
        told = asked.told_from(0)
        # Nothing was known at the gap the first time, nor at the other place; the
        # second search at the gap is told what the first found.
        assert asked.known == [None, None, told]
        # A jump, held for one of the times a change is held, begun before the loss.
        assert told.move == lab.move("a", "right") and told.hold in (4, 8, 16)
        assert 1 <= told.before <= len(asked.traces[0])
        # It gets past again, at once: the blind play and the look (after the plain
        # way, where there is one).
        first, again = asked.out[0], asked.out[2]
        assert again.way is not None and again.again and not first.again
        assert again.way.change == first.way.change and again.way.moves == first.way.moves  # type: ignore[union-attr]
        assert again.tries == (2 if told.plain is None else 3) < first.tries
        # It is kept from go to go, and a search for nothing leaves it as it is: the
        # next go loses three lives at the gap, and each search there is told it.
        a_go(lab, r)
        assert asked.known[3:] == [told] * 3 and asked.out[5].again
    found = said(caplog, "found a way past on try")
    assert len(found) == 3
    assert not found[0].endswith(AGAIN) and found[1].endswith(AGAIN) and found[2].endswith(AGAIN)
    # What it says of the way is what it said the first time, and that it was known.
    assert f"on try {first.tries}, after" in found[0]
    assert f"on try {again.tries}, after" in found[1]
    assert found[1].split(" s: ", 1)[1] == found[0].split(" s: ", 1)[1] + AGAIN
    assert asked.answers == []


def as_found(asked: Asked, **how: Any) -> Any:
    """An answer: the moves of the first way found, said to be another kind of way."""

    def answer(trace: tuple[int, ...]) -> Search:
        way = asked.out[0].way
        assert way is not None
        made = Way(way.moves, how.pop("change", way.change), 4, way.ground, **how)
        return Search(made, 4)

    return answer


def test_what_is_kept_is_a_single_change_and_the_newest(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asked = Asked(monkeypatch)
    r, _ = searching(lab, ways_per_go=1, give_up_after=None, max_steps=400)
    walk, jump = lab.move("right"), lab.move("a", "right")
    caplog.set_level(logging.INFO, logger=LOG)
    to_the_gap(lab, r)
    told = asked.told_from(0)
    r.finish()
    # A search for nothing leaves what is known as it is.
    asked.answers = [NOTHING]
    to_the_gap(lab, r)
    r.finish()
    # A way of two changes is not kept, nor the plain way held with no change to it:
    # the search after each is told what the first found.
    asked.answers = [as_found(asked, first=Change(3, jump, 4))]
    to_the_gap(lab, r)
    r.finish()
    asked.answers = [as_found(asked, change=Change(0, walk, DIED_AT), plain=walk)]
    to_the_gap(lab, r)
    r.finish()
    assert asked.known == [None, told, told, told]
    # A change to the plain way is kept, counted from the loss the plain way came to:
    # the newest way found about a place is the one known there.
    asked.answers = [as_found(asked, change=Change(30, jump, 8), plain=walk, lost_at=41)]
    to_the_gap(lab, r)
    r.finish()
    asked.answers = [as_found(asked, change=Change(20, jump, 16))]
    to_the_gap(lab, r)
    r.finish()
    to_the_gap(lab, r)
    r.finish()
    assert asked.known[4:] == [told, Known(jump, 8, 11, walk), Known(jump, 16, DIED_AT - 20)]
    # The log says a way was the known change where it was, and not of a way found
    # otherwise by a search that was told one.
    ways = [(known, out) for known, out in zip(asked.known, asked.out, strict=True) if out.way]
    found = said(caplog, "found a way past on try")
    assert [line.endswith(AGAIN) for line in found] == [out.again for _, out in ways]
    assert [out.again for known, out in ways if known is not None][:4] == [False] * 4


def test_no_more_places_are_kept_than_for_searches_for_nothing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert episode.MOST_SOUGHT == 32

    def goes(asked: Asked) -> list[Known | None]:
        r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
        # Over the gap; a way found at the place further on (whatever it is said to
        # be, the life goes again); at the gap again.
        asked.answers = [REAL, as_found(asked, change=Change(2, lab.move("left"), 4)), NOTHING]
        a_go(lab, r)
        return asked.known

    known = goes(Asked(monkeypatch))
    assert known[:2] == [None, None] and known[2] is not None
    monkeypatch.setattr(episode, "MOST_SOUGHT", 1)
    # Room for one place: the place further on is the one kept, and the gap forgotten.
    assert goes(Asked(monkeypatch)) == [None, None, None]
    # The one place is kept: with no way found further on, it is the gap.
    asked = Asked(monkeypatch, [REAL, NOTHING, NOTHING])
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    a_go(lab, r)
    assert asked.known == [None, None, asked.told_from(0)]


def test_a_search_at_a_standstill_is_told_nothing_and_keeps_nothing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    walk, jump = lab.move("right"), lab.move("a", "right")
    on = Search(Way((walk,) * 4, Change(0, jump, 4), 4, 9), 4)  # a way on, from standing
    asked = Asked(monkeypatch, [on, REAL])
    r, _ = standing(lab, ways_per_go=9, max_steps=5000, unstick_after=None, give_up_after=None)
    r.begin()
    # Standing about one spot it looks for a way on, and is given one. Then it walks
    # into the gap nearby: the search there is told nothing of it, and finds the way.
    hop_about(lab, r, FAR)
    assert (asked.stood, asked.known) == ([True], [None])
    while len(asked.known) < 2:
        assert r.step()
    assert (asked.stood, asked.known) == ([True, False], [None, None])
    told = asked.told_from(1)
    r.finish()
    # The next go stands about the same spot: that search is told nothing. The one
    # at the gap after it is told what got past the gap.
    asked.answers = [NOTHING, NOTHING]
    lab.game.reset()
    r.begin()
    hop_about(lab, r, FAR)
    while len(asked.known) < 4:
        assert r.step()
    assert asked.stood[2:4] == [True, False] and asked.known[2:4] == [None, told]
    r.finish()


def test_a_life_seen_going_is_told_and_kept_like_one_lost(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asked = Asked(monkeypatch)
    r, _ = looking(lab)
    near = Near(monkeypatch, r.watch)

    def a_look() -> None:
        lab.game.reset()
        r.begin()
        steps(r, 5)
        near.now = THING
        assert r.step()
        near.now = ()

    with caplog.at_level(logging.INFO, logger=LOG):
        a_look()
        assert (r.ways_ahead, r.ways_ahead_found) == (1, 1) and asked.known == [None]
        told = asked.told_from(0)
        r.finish()
        # The next go sees the life going at the same place, and tries that first.
        a_look()
        assert asked.known == [None, told] and asked.out[1].again
        assert asked.out[1].way.moves == asked.out[0].way.moves  # type: ignore[union-attr]
        r.finish()
    first, second = said(caplog, "found a way past on try")
    assert "before the life would go" in first or "before that would lose the life" in first
    assert not first.endswith(AGAIN) and second.endswith(AGAIN)
    assert TOP == ((0, 0), (0, 0))
