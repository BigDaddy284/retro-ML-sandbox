"""T-1029: the runner comes at a place that cost a life for good from further back
each time, with the practice game and the real services."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training.moments import LEAD
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import Lab, steps
from tests.unit.training.test_remembered_way import planted, remembering

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

SECOND = 88  # past the first gap by a jump at step 19 and one at 47, the life goes here
FIRST = 60  # and here with the first jump only

CUT = (
    "A life went at step %d, close behind the end of the way the AI remembers, and it "
    "found no way past: it now remembers %d steps of that way instead of %d, to arrive "
    "there earlier."
)
BACK = (
    "A life went at step %d with no way past, about where one went for good in an earlier "
    "go: the AI ends what it remembers of this go %d steps before the place instead of %d, "
    "to come at it from further back."
)


def test_a_second_loss_at_the_place_backs_off_twice_as_far(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, route, harness = remembering(lab, ways_per_go=0)  # no search: every loss stands
    walk, jump = lab.move("right"), lab.move("right", "a")
    way = [walk] * 19 + [jump] * 4 + [walk] * 24 + [jump] * 4 + [walk] * 29
    assert len(way) == 80
    planted(route, harness, way)
    lives = toy_hopper.START_LIVES

    def go(lost_at: int) -> list[str]:
        """A go that plays what it remembers and walks on until the life goes."""
        lab.agent.script.clear()
        lab.agent.script.extend([walk] * 100)
        r.begin()
        steps(r, lost_at - 1)
        assert lab.game.state.lives == lives
        with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
            caplog.clear()
            steps(r, 1)
        assert lab.game.state.lives == lives - 1
        said = [rec.getMessage() for rec in caplog.records]
        r.finish()
        return said

    # The first time at the place is as it always was: thirty steps before the loss.
    assert go(SECOND) == [CUT % (SECOND, SECOND - LEAD, 80)]
    assert route.story == tuple(way[: SECOND - LEAD]) and route.moves == route.story
    # The next go arrives by that way, with thirty steps of its own, and the life goes
    # at the same step: sixty steps before it this time, for the story and for the way.
    assert go(SECOND) == [
        BACK % (SECOND, 2 * LEAD, LEAD),
        CUT % (SECOND, SECOND - 2 * LEAD, SECOND - LEAD),
    ]
    assert route.story == tuple(way[: SECOND - 2 * LEAD]) and route.moves == route.story
    # The third go's way ends before the second jump, so it walks into the gap before
    # the place: 28 steps short of it is the same place still, and 120 steps before
    # the loss is before the start.
    assert go(FIRST) == [BACK % (FIRST, 4 * LEAD, LEAD), CUT % (FIRST, 0, SECOND - 2 * LEAD)]
    assert route.story == () and route.moves == ()
