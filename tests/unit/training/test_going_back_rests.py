"""T-1039: where going back has worked, searches for nothing make it rest, not stop for good."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.episode import MOST_REST, NOW_SLOT, REST_GOES
from retro_ai_lab.training.search import Search, Way
from retro_ai_lab.training.ways import Change
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_search_at_a_loss import searching
from tests.unit.training.test_standing_still import FAR, Asked, hop_about, standing

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

NOTHING = Search(None, 5)
FOUND = "the real search"  # which finds the way over the first gap
# What a go that walks right until it is two lives down came to: searches made, ways
# found, and whether the runner has given up at its end. (Two lives a go, because the
# test puts the game back to its start between goes with nobody told: a life count
# that went up as often as down would look to the engine like no life count at all.)
A_WAY, NO_WAY, AT_REST = (1, 1, False), (1, 0, False), (0, 0, True)
THE_LAST = (1, 0, True)  # the search for nothing at which it gives up
FOR_GOOD = (
    "Going back found no way past 2 times in a row, so the AI stops going back in this "
    "game: here it only costs time."
)
AGAIN = (
    "The AI left going back alone for a while, after its searches kept finding nothing. "
    "From this go on it goes back after a life lost again."
)


def rests(times: str, goes: int) -> str:
    return (
        f"Going back found no way past {times} in a row. It has found ways past before in "
        f"this game, so the AI leaves going back alone for the next {goes} goes and then "
        "goes back again."
    )


def scripted(monkeypatch: pytest.MonkeyPatch, script: list[object]) -> list[object]:
    """Each search answers with the next of ``script``. One that is not expected
    (nothing left to answer with) fails the test."""
    real = episode.find_way

    def answer(*args: object, **kw: object) -> Search:
        out = script.pop(0)
        return out if isinstance(out, Search) else real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(episode, "find_way", answer)
    return script


def walk_on(lab: Lab, r: EpisodeRunner) -> None:
    """Walk right until the go is two lives down."""
    while lab.game.state.lives > toy_hopper.START_LIVES - 2:
        assert r.step()


def go(lab: Lab, r: EpisodeRunner) -> tuple[int, int, bool]:
    """One go that walks right, into the first gap, until it is two lives down."""
    lab.game.reset()
    r.begin()
    walk_on(lab, r)
    result = r.finish()
    return result.ways_sought, result.ways_found, r.gave_up


def said(caplog: pytest.LogCaptureFixture) -> list[str]:
    """What the log said about leaving going back alone, stopping it and taking it up."""
    lines = [rec.getMessage() for rec in caplog.records]
    return [line for line in lines if "going back" in line.lower() and "Lost a life" not in line]


def test_it_rests_where_going_back_has_worked(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert (REST_GOES, MOST_REST) == (4, 32)
    script = scripted(monkeypatch, [FOUND, NOTHING, NOTHING, NOTHING, NOTHING])
    r, moments = searching(lab, ways_per_go=2, give_up_after=2)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        # A way past the first gap, and then one search for nothing at the next.
        assert go(lab, r) == (2, 1, False)
        # The second search for nothing in a row. What is left of that go makes none,
        # though a go may make two.
        lab.game.reset()
        r.begin()
        steps(r, DIED_AT)
        assert r.ways_sought == 1 and r.gave_up
        saves, loads = len(lab.saved), len(lab.loaded)
        walk_on(lab, r)
        assert r.ways_sought == 1 and len(moments) == 0
        assert r.finish().ways_sought == 1
        assert said(caplog) == [rests("2 times", 4)]
        # Four goes as they were before going back existed: no search, no save, no load.
        for _ in range(4):
            assert go(lab, r) == AT_REST
            assert len(moments) == 0
        assert (len(lab.saved), len(lab.loaded)) == (saves, loads)
        assert said(caplog) == [rests("2 times", 4)]
        # The fifth goes back again, from its start.
        lab.game.reset()
        r.begin()
        assert not r.gave_up
        assert said(caplog) == [rests("2 times", 4), AGAIN]
        steps(r, DIED_AT)
        assert r.ways_sought == 1 and not r.gave_up
        assert NOW_SLOT in lab.saved[saves:] and len(lab.loaded) > loads
        walk_on(lab, r)
        assert r.finish().ways_sought == 2 and r.gave_up  # two more for nothing
    assert script == []
    assert said(caplog) == [rests("2 times", 4), AGAIN, rests("2 times", 8)]


def test_the_count_starts_again_after_a_rest(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    script = scripted(monkeypatch, [FOUND, NOTHING, NOTHING, NOTHING, NOTHING])
    r, _ = searching(lab, give_up_after=2)
    assert [go(lab, r) for _ in range(3)] == [A_WAY, NO_WAY, THE_LAST]
    assert [go(lab, r) for _ in range(4)] == [AT_REST] * 4
    # One for nothing after the rest is one in a row, not the third.
    assert go(lab, r) == NO_WAY
    assert go(lab, r) == THE_LAST
    assert script == []


def test_each_rest_for_nothing_is_twice_as_long(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The most is 32 goes in the app (the first test holds that); 16 here, which keeps
    # this test to 60 goes where it would be 140.
    monkeypatch.setattr(episode, "MOST_REST", 16)
    script = scripted(monkeypatch, [FOUND, *[NOTHING] * 10])
    r, _ = searching(lab, give_up_after=2)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        assert go(lab, r) == A_WAY
        for rest in (4, 8, 16, 16, 16):
            # The go after a rest searches: not one go more is left alone.
            assert [go(lab, r) for _ in range(2)] == [NO_WAY, THE_LAST]
            assert [go(lab, r) for _ in range(rest)] == [AT_REST] * rest
    assert script == []
    lines = [line for rest in (4, 8, 16, 16, 16) for line in (rests("2 times", rest), AGAIN)]
    assert said(caplog) == lines[:-1]  # no go began after the last rest


def test_a_way_found_makes_the_next_rest_short_again(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = scripted(
        monkeypatch, [FOUND, NOTHING, NOTHING, NOTHING, NOTHING, FOUND, NOTHING, NOTHING, NOTHING]
    )
    r, _ = searching(lab, give_up_after=2)
    assert [go(lab, r) for _ in range(3)] == [A_WAY, NO_WAY, THE_LAST]
    assert [go(lab, r) for _ in range(4)] == [AT_REST] * 4
    assert [go(lab, r) for _ in range(2)] == [NO_WAY, THE_LAST]
    assert [go(lab, r) for _ in range(8)] == [AT_REST] * 8
    # A way found: the count of searches for nothing starts again, and so do the rests.
    assert [go(lab, r) for _ in range(3)] == [A_WAY, NO_WAY, THE_LAST]
    assert [go(lab, r) for _ in range(4)] == [AT_REST] * 4
    assert go(lab, r) == NO_WAY
    assert script == []


def test_one_for_nothing_where_one_is_enough(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    script = scripted(monkeypatch, [FOUND, NOTHING, NOTHING])
    r, _ = searching(lab, give_up_after=1)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        assert [go(lab, r) for _ in range(2)] == [A_WAY, THE_LAST]
        assert [go(lab, r) for _ in range(4)] == [AT_REST] * 4
        assert go(lab, r) == THE_LAST
        assert [go(lab, r) for _ in range(8)] == [AT_REST] * 8
    assert script == []
    assert said(caplog) == [rests("1 time", 4), AGAIN, rests("1 time", 8)]


def test_where_it_never_worked_it_stops_for_good(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    script = scripted(monkeypatch, [NOTHING, NOTHING])
    r, moments = searching(lab, give_up_after=2)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        assert [go(lab, r) for _ in range(2)] == [NO_WAY, THE_LAST]
        saves, loads = len(lab.saved), len(lab.loaded)
        # More goes than the first two rests and the searches between them would take.
        assert [go(lab, r) for _ in range(16)] == [AT_REST] * 16
    assert (len(lab.saved), len(lab.loaded)) == (saves, loads) and len(moments) == 0
    assert script == []
    assert said(caplog) == [FOR_GOOD]


def test_a_way_on_at_a_standstill_does_not_count(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stand = lab.move()
    on = Search(Way((stand, stand), Change(0, stand, 2), tries=2, ground=3), 2)
    asked = Asked(monkeypatch, [on])  # and nothing found from then on
    r, _ = standing(lab, ways_per_go=9, max_steps=5000, unstick_after=None, give_up_after=2)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        r.begin()
        # Standing about one spot it looks for a way on, and finds one.
        hop_about(lab, r, FAR)
        assert [stood for *_, stood in asked.calls] == [True]
        assert (r.ways_found, r.ways_stood) == (1, 1)
        # Then it walks into the gap twice, and going back finds nothing twice.
        walk_on(lab, r)
        assert [stood for *_, stood in asked.calls] == [True, False, False]
        assert r.gave_up and not r.stood_gave_up
        r.finish()
        # No search after a life lost ever found a way: that is for good.
        assert [go(lab, r) for _ in range(12)] == [AT_REST] * 12
    assert len(asked.calls) == 3
    assert said(caplog) == [FOR_GOOD]
