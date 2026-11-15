"""T-1042: the runner remembers where a search for nothing stopped, for the place."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.episode import MOST_SOUGHT, EpisodeResult
from retro_ai_lab.training.search import TOP, Search, Spot
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_going_back_rests import walk_on
from tests.unit.training.test_search_at_a_loss import searching
from tests.unit.training.test_standing_still import FAR, hop_about, standing

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

REAL = "the real search"
Begin = tuple[Spot, Spot]


def nothing(plain: Spot, did: Spot) -> Search:
    """A search that found nothing and stopped there in its two lists."""
    return Search(None, 5, stopped_at=(plain, did))


class Asked:
    """Stands in for ``find_way`` in the runner: notes where each search was to begin
    and whether it was one at a standstill, and answers from a list (the real search
    where the list says so). A search nobody expected fails the test."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, answers: list[object]) -> None:
        self.begun: list[Begin] = []
        self.stood: list[bool] = []
        self.found: list[bool] = []
        self.answers = list(answers)
        self.real = episode.find_way
        monkeypatch.setattr(episode, "find_way", self)

    def __call__(self, *args: object, **kw: object) -> Search:
        self.begun.append(kw["begin_at"])  # type: ignore[arg-type]
        self.stood.append(bool(kw["stood"]))
        out = self.answers.pop(0)
        if not isinstance(out, Search):
            out = self.real(*args, **kw)  # type: ignore[arg-type]
        self.found.append(out.way is not None)
        return out


def a_go(lab: Lab, r: EpisodeRunner) -> EpisodeResult:
    """One go that walks right from the start for as long as the game goes on."""
    lab.game.reset()
    r.begin()
    while r.step():
        pass
    return r.finish()


def carrying(caplog: pytest.LogCaptureFixture) -> list[str]:
    """How often the log said a search had been made about here before, each time it
    said a search carries on."""
    lines = [rec.getMessage() for rec in caplog.records]
    said = [line for line in lines if "carries on with the changes" in line]
    lead, rest = "The AI has looked for a way past about here ", " before and found none. "
    assert all(line.startswith(lead) and rest in line for line in said)
    assert all(
        line.endswith(
            "This search carries on with the changes the last one did not get to, instead "
            "of making the same tries again."
        )
        for line in said
    )
    return [line[len(lead) : line.index(rest)] for line in said]


# The first gap of this level, the next place a life goes at, and the one after: the
# camera has travelled 136, 329 and 553 pixels there, more than half a screen (128) apart.


def test_the_next_search_at_the_spot_begins_where_the_last_stopped(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    one, two, three = (
        nothing((0, 2), (0, 6)),
        nothing((0, 8), (1, 4)),
        nothing((2, 0), (0, 0)),
    )
    asked = Asked(monkeypatch, [one, two, three, nothing((2, 6), (0, 4)), REAL])
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        # Three lives at the first gap, three searches for nothing: each begins where
        # the one before it stopped.
        assert a_go(lab, r).ways_sought == 3
        assert asked.begun == [TOP, one.stopped_at, two.stopped_at]
        assert carrying(caplog) == ["once", "2 times"]
        # It is kept from one go to the next.
        lab.game.reset()
        r.begin()
        steps(r, DIED_AT)
        assert asked.begun[3:] == [three.stopped_at]
        assert carrying(caplog) == ["once", "2 times", "3 times"]
        # The real search, begun where it is told to, finds the way over.
        # (Put back to the start a life down, it walks into the gap again.)
        steps(r, DIED_AT)
        assert asked.begun[4:] == [((2, 6), (0, 4))] and asked.found[4:] == [True]
        assert carrying(caplog)[3:] == ["4 times"]
        assert r.ways_found == 1
    assert asked.answers == [] and asked.stood == [False] * 5
    lines = [rec.getMessage() for rec in caplog.records]
    # The lines there were are as they were, and come after the new one.
    gone = [n for n, line in enumerate(lines) if line.endswith("none got past.")]
    told = [n for n, line in enumerate(lines) if "carries on with the changes" in line]
    assert len(gone) == 4 and len(told) == 4
    assert [n < m for n, m in zip(told, gone[1:], strict=False)] == [True] * 3
    assert lines[gone[0]] == (
        "Lost a life at step 36. Went back 36 steps and made 5 tries in 0.0 s: none got past."
    )


def test_a_search_elsewhere_begins_at_the_top_and_is_remembered_apart(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gap = nothing((0, 4), (1, 6))
    further = nothing((0, 6), (1, 2))
    gap_again = nothing((0, 6), (0, 8))
    asked = Asked(
        monkeypatch,
        [nothing((0, 2), (0, 2)), gap, REAL, REAL, further, gap_again, REAL, REAL, REAL],
    )
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        # Two searches for nothing at the gap, and the third finds the way over.
        assert a_go(lab, r).ways_found == 1
        assert asked.begun == [TOP, ((0, 2), (0, 2)), gap.stopped_at]
        assert asked.found == [False, False, True]
        # The next go. The way found changed nothing of what is remembered: the search
        # at the gap begins where the last one for nothing stopped. Over the gap a
        # life goes somewhere else: that search begins at the top. Put back to the
        # start it comes to the gap a life down, and the gap is as it was remembered.
        assert a_go(lab, r).ways_found == 1
        assert asked.begun[3:] == [gap.stopped_at, TOP, gap.stopped_at]
        assert asked.found[3:] == [True, False, False]
        # The go after: the gap from where its last search for nothing stopped, the
        # place further on from where its own did, and a third place from the top.
        assert a_go(lab, r).ways_found == 3
        assert asked.begun[6:] == [gap_again.stopped_at, further.stopped_at, TOP]
        assert asked.found[6:] == [True] * 3
    assert asked.answers == []
    # The log says it of each search that carried on, and of no other.
    assert carrying(caplog) == ["once", "2 times", "2 times", "2 times", "3 times", "once"]


def test_no_more_places_are_remembered_than_it_keeps(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert MOST_SOUGHT == 32
    monkeypatch.setattr(episode, "MOST_SOUGHT", 1)
    gap = nothing((0, 4), (1, 6))
    asked = Asked(
        monkeypatch,
        [gap, gap, REAL, REAL, nothing((0, 6), (1, 2)), nothing((0, 2), (0, 2)), REAL, REAL, REAL],
    )
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    a_go(lab, r)
    assert asked.begun == [TOP, gap.stopped_at, gap.stopped_at]
    # The same goes as in the test above, with room for one place: the search for
    # nothing further on is the one remembered, and the gap is forgotten.
    a_go(lab, r)
    assert asked.begun[3:] == [gap.stopped_at, TOP, TOP]
    assert asked.found[3:] == [True, False, False]
    # Then the gap is the one remembered, and the place further on forgotten.
    a_go(lab, r)
    assert asked.begun[6:] == [((0, 2), (0, 2)), TOP, TOP]
    assert asked.found[6:] == [True] * 3 and asked.answers == []


def test_about_here_is_the_same_place_of_the_go_within_half_a_screen(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, a2, b, c, c2 = (
        nothing((0, 2), (0, 2)),
        nothing((0, 4), (0, 4)),
        nothing((1, 2), (1, 2)),
        nothing((2, 2), (2, 2)),
        nothing((2, 4), (2, 4)),
    )
    more = [nothing((1, 0), (1, 0)), nothing((1, 8), (1, 8)), nothing((0, 6), (0, 6)), a]
    asked = Asked(monkeypatch, [a, b, a2, c, c2, *more])
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    # Where each life is seen lost, as the watch would say it: the place of the go,
    # where in it, and how near counts (half the screen's longer side).
    wheres: list[tuple[int, tuple[int, int], int] | None] = [
        (0, (100, 0), 128),
        (1, (100, 0), 128),  # the same spot of another place of the go
        (0, (228, 0), 128),  # half a screen further: about there
        (0, (229, 0), 128),  # a pixel more: somewhere else
        (0, (170, 0), 128),  # near both: the nearer
        (0, (100, 129), 128),  # down counts like across
        None,  # the watch has not looked at a screen yet
        (0, (100, 0), 128),
        (1, (100, 0), 128),
    ]
    monkeypatch.setattr(r, "_lost_where", lambda: wheres.pop(0))
    for _ in range(3):
        a_go(lab, r)
    assert asked.begun == [
        TOP,
        TOP,
        a.stopped_at,
        TOP,
        c.stopped_at,
        TOP,
        TOP,
        a2.stopped_at,
        b.stopped_at,
    ]
    assert wheres == [] and asked.answers == []


def test_a_standstill_begins_at_the_top_and_remembers_nothing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    first = nothing((0, 4), (1, 6))
    asked = Asked(
        monkeypatch, [nothing((1, 1), (2, 2)), first, nothing((0, 8), (1, 8)), first, first]
    )
    r, _ = standing(lab, ways_per_go=9, max_steps=5000, unstick_after=None, give_up_after=None)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        r.begin()
        # Standing about one spot it looks for a way on: from the top.
        hop_about(lab, r, FAR)
        assert (asked.stood, asked.begun) == ([True], [TOP])
        # Then it walks into the gap twice. The first search there begins at the top
        # too: where the standstill's search stopped is nobody's to carry on from.
        walk_on(lab, r)
        assert asked.stood == [True, False, False]
        assert asked.begun[1:] == [TOP, first.stopped_at]
        r.finish()
        # And a standstill after searches for nothing where a life went is from the
        # top as well.
        lab.game.reset()
        r.begin()
        hop_about(lab, r, FAR)
        assert asked.stood[3:] == [True] and asked.begun[3:] == [TOP]
    assert carrying(caplog) == ["once"]


# ------------------------------------------------------------ with the real search, made short


def goes_until_over(lab: Lab, r: EpisodeRunner, most: int) -> int | None:
    """Goes that walk into the first gap, one search each, until one gets over: the
    number of that go, or None when none of ``most`` does."""
    for number in range(1, most + 1):
        lab.game.reset()
        r.begin()
        steps(r, DIED_AT)
        found = r.finish().ways_found
        assert found in (0, 1)
        if found:
            return number
    return None


def test_short_searches_get_over_the_gap_between_them(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    # Eight tries a search: the way over is the seventeenth change in the list.
    r, _ = searching(lab, ways_per_go=1, give_up_after=None, way_tries=8)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        assert goes_until_over(lab, r, 8) == 4
    assert carrying(caplog) == ["once", "2 times", "3 times"]
    lines = [rec.getMessage() for rec in caplog.records]
    assert sum(line.endswith("made 8 tries in 0.0 s: none got past.") for line in lines) == 3
    assert sum("found a way past on try" in line for line in lines) == 1


def test_short_searches_from_the_top_each_time_never_get_over(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same goes with nothing carried from one search to the next (as it was).
    real = episode.find_way
    tries: list[int] = []

    def from_the_top(*args: object, **kw: object) -> Search:
        kw["begin_at"] = TOP
        out = real(*args, **kw)  # type: ignore[arg-type]
        tries.append(out.tries)
        return out

    monkeypatch.setattr(episode, "find_way", from_the_top)
    r, _ = searching(lab, ways_per_go=1, give_up_after=None, way_tries=8)
    assert goes_until_over(lab, r, 8) is None
    assert tries == [8] * 8
