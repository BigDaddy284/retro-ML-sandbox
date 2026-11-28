"""T-1074: a place that keeps costing a life is not backed off from further each time.

Where a life goes for good and no way past is found, what the go remembers of itself
ends ``LEAD`` steps before the place, and the way it began with too if it ended that
close (T-1014, T-1015). T-1029 doubled that for each earlier loss that stood about
there: 60, 120, at most 240 steps. On the practice game's hard levels (a walker
pacing before every gap) lives keep going at the same few places, the doubling soon
reached 240 and stayed there, and four times in one run it cut what the AI
remembered of the level to nothing: the next go began from the very start and met
the earlier places again. The same eight runs without the doubling finished 205
levels against 107. The route still notes the places, and the log says when a loss
stands where one stood before.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import START_LIVES
from retro_ai_lab.training.moments import LEAD
from retro_ai_lab.training.route import Route
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import Lab, steps
from tests.unit.training.test_remembered_way import planted, remembering
from tests.unit.training.test_remembered_way_ends_before_the_mistake import (
    TAKES,
    WENT,
    a_game,
    a_go,
)

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LOG = "retro_ai_lab.training.episode"
SECOND = 88  # past the first gap by a jump at step 19 and one at 47, the life goes here
CUT = (
    "A life went at step %d, close behind the end of the way the AI remembers, and it "
    "found no way past: it now remembers %d steps of that way instead of %d, to arrive "
    "there earlier."
)
SAME = (
    "A life went at step %d with no way past, about where one went for good in an earlier "
    "go. What the AI remembers of this go still ends %d steps before the place, as the "
    "first time: backing off further threw away more of the way than it helped."
)


def said(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if rec.name == LOG]


def test_the_same_place_again_and_again_is_backed_off_from_as_the_first_time(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, route, harness = remembering(lab, ways_per_go=0)  # no search: every loss stands
    walk, jump = lab.move("right"), lab.move("right", "a")
    way = [walk] * 19 + [jump] * 4 + [walk] * 24 + [jump] * 4 + [walk] * 29
    planted(route, harness, way)
    lives = START_LIVES

    def go() -> list[str]:
        """A go that plays what it remembers and walks on until the life goes."""
        lab.agent.script.clear()
        lab.agent.script.extend([walk] * 100)
        r.begin()
        steps(r, SECOND - 1)
        assert lab.game.state.lives == lives
        with caplog.at_level(logging.INFO, logger=LOG):
            caplog.clear()
            steps(r, 1)
        assert lab.game.state.lives == lives - 1
        lines = said(caplog)
        r.finish()
        return lines

    # The first time at the place: thirty steps before the loss, and the way that
    # ended eight steps before it cut to there. Nothing said of an earlier go.
    assert go() == [CUT % (SECOND, SECOND - LEAD, len(way))]
    kept = tuple(way[: SECOND - LEAD])
    assert route.moves == kept and route.story == kept
    # The second, third and fourth time: thirty steps before it still. The way, which
    # ends thirty steps before it, is not cut for the loss (doubled, it was cut to 28
    # steps the second time and to nothing the third). What a go is paid for playing it
    # may end it a step or two earlier (T-1026), and that is all.
    for _ in range(3):
        remembered = len(route.moves)
        assert go() == [SAME % (SECOND, LEAD)]
        assert r.remembered == remembered
        assert route.story == kept
        assert route.moves == kept[: len(route.moves)]
        assert len(route.moves) > SECOND - 2 * LEAD
    # The route has noted the place each time (T-1029, written down with it, T-1064).
    assert route.to_dict()["stood"] == [SECOND] * 4


def test_with_the_games_pause_known_the_lead_is_the_same_each_time(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    r, route, _, did = a_game(lab, monkeypatch, toy_hopper.DYING_FRAMES)
    went, takes = WENT[toy_hopper.DYING_FRAMES], TAKES[toy_hopper.DYING_FRAMES]
    short = went - takes - LEAD  # thirty steps before the mistake
    a_go(lab, r, did, went)
    assert r.dying_steps == takes and route.story == tuple(did[:short])
    r.finish()
    for _ in range(3):
        with caplog.at_level(logging.INFO, logger=LOG):
            caplog.clear()
            a_go(lab, r, did[len(route.moves) :], went)
        assert lab.game.state.lives == START_LIVES - 1
        assert route.story == tuple(did[:short])
        assert SAME % (went, LEAD) in said(caplog)
        r.finish()
        assert 0 < len(route.moves) <= short


def test_a_route_that_counts_earlier_losses_is_told_of_each(lab: Lab) -> None:
    # The runner asks the route as before, so the route goes on counting: a loss the
    # runner has left at a place doubles the lead for the next one there, as it did.
    r, route, _ = remembering(lab, ways_per_go=0)
    walk = lab.move("right")
    lab.agent.script.extend([walk] * 100)
    r.begin()
    while lab.game.state.lives == START_LIVES:
        assert r.step()
    r.finish()
    stood = route.to_dict()["stood"]
    assert len(stood) == 1
    copy = Route.from_dict(route.to_dict(), route.start)
    copy.begin(route.start)
    for count in range(1, stood[0] + 1):
        copy.did(walk, count)
    assert copy.loss_stands(LEAD) == 2 * LEAD
