"""T-1071: what the AI remembers of a go ends before the mistake, not only before the
step the life was counted at.

A go that loses a life for good ends what it remembers of itself ``LEAD`` steps
before the loss (T-1014), "so that playing it again arrives short of the place,
alive". The loss was the step the game counted the life at. A game takes its time
over a lost life: Super Mario Bros. some 54 steps, and nothing pressed in them
changes anything. Thirty steps before the count is then 24 steps after the mistake,
and the way the next go begins with ends in the middle of the dying. In the user's
log of 27 Nov four goes in a row played the same remembered way into the same loss,
52 steps after the way's end each time, and the way was not shortened: it ended
"more than thirty steps before the loss".

Where the game has shown how long it takes over a lost life (T-1033), the story and
the way now end that much earlier as well.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import DYING, PLAYING, START_LIVES
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.moments import LEAD, Moments
from retro_ai_lab.training.route import Route
from retro_ai_lab.training.search import Search
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import Lab, steps
from tests.unit.training.test_last_chance_in_a_go import LONG, SIX
from tests.unit.training.test_remembered_way import planted, plays_from_the_start, remembering

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LOG = "retro_ai_lab.training.episode"
# Over the first gap by a jump at step 19, and on into the walker: the step the life
# is counted at, and how many steps before it nothing changes the loss any more.
WENT = {LONG: 98, toy_hopper.DYING_FRAMES: 60}
TAKES = {LONG: 50, toy_hopper.DYING_FRAMES: 12}
PAUSES = [LONG, toy_hopper.DYING_FRAMES]
SAID = "This game takes about %d steps over a lost life"
CUT = (
    "A life went at step %d, close behind the end of the way the AI remembers, and it "
    "found no way past: it now remembers %d steps of that way instead of %d, to arrive "
    "there earlier."
)


def a_game(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, frames: int, **kw: object
) -> tuple[EpisodeRunner, Route, ExperimentHarness, list[int]]:
    """A runner that remembers, on the practice game with that dying pause, whose
    searches find nothing: every loss stands. And the moves of a go that jumps the
    first gap and walks on into the walker."""
    monkeypatch.setattr(episode, "find_way", lambda *a, **k: Search(None, 0))
    lab.game.dying_frames = frames
    kw.setdefault("last_chance", True)
    kw.setdefault("moments", Moments(SIX))
    kw.setdefault("unstick_after", None)
    r, route, harness = remembering(lab, **kw)
    walk, jump = lab.move("right"), lab.move("right", "a")
    return r, route, harness, [walk] * 19 + [jump] * 4 + [walk] * 300


def a_go(lab: Lab, r: EpisodeRunner, moves: list[int], upto: int) -> None:
    """A go that plays what it remembers, then ``moves``, until step ``upto``."""
    lab.agent.script.clear()
    lab.agent.script.extend(moves)
    r.begin()
    steps(r, upto - r.steps)


def said(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if rec.name == LOG]


@pytest.mark.parametrize("frames", PAUSES, ids=["a long pause", "the game's own"])
def test_what_a_go_remembers_ends_before_the_loss_was_settled(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, frames: int
) -> None:
    r, route, harness, did = a_game(lab, monkeypatch, frames)
    went, takes = WENT[frames], TAKES[frames]
    a_go(lab, r, did, went - 1)
    assert lab.game.state.lives == START_LIVES and r.dying_steps is None
    with caplog.at_level(logging.INFO, logger=LOG):
        steps(r, 1)
    assert lab.game.state.lives == START_LIVES - 1 and r.dying_steps == takes
    # Thirty steps before the step from which nothing could change the loss.
    assert route.story == tuple(did[: went - takes - LEAD])
    assert [line for line in said(caplog) if line.startswith(SAID % takes)] != []
    r.finish()
    # The way the next go begins with is that story, up to where it was paid the most.
    way = route.moves
    assert 0 < len(way) <= len(route.story) and way == route.story[: len(way)]
    # The game agrees: played again from the start, the story ends with the player on
    # its feet, and standing still from there keeps the life for as long as you like.
    plays_from_the_start(lab, harness, [*route.story, *([lab.move()] * (takes + LEAD + 20))])
    assert (lab.game.state.mode, lab.game.state.lives) == (PLAYING, START_LIVES)
    if frames == LONG:
        # As it was, thirty steps before the count: the dying had begun 20 steps before.
        plays_from_the_start(lab, harness, did[: went - LEAD])
        assert lab.game.state.mode == DYING and went - LEAD > went - takes


def test_where_the_game_has_not_shown_how_long_it_takes_it_is_thirty_steps_as_before(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Without asking from when on a loss was settled, nothing is known of the pause.
    r, route, _, did = a_game(lab, monkeypatch, LONG, last_chance=False)
    with caplog.at_level(logging.INFO, logger=LOG):
        a_go(lab, r, did, WENT[LONG])
    assert lab.game.state.lives == START_LIVES - 1 and r.dying_steps is None
    assert route.story == tuple(did[: WENT[LONG] - LEAD])
    assert [line for line in said(caplog) if "over a lost life" in line] == []
    r.finish()


def test_a_way_that_ends_in_the_dying_is_cut_to_before_the_mistake(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The user's goes 130 to 133: the way ended two steps after the loss was settled,
    52 before the count, and was left as it was."""
    r, route, harness, did = a_game(lab, monkeypatch, LONG)
    went, takes = WENT[LONG], TAKES[LONG]
    way = did[: went - takes + 2]
    planted(route, harness, way)
    assert len(way) < went - LEAD  # more than thirty steps before the count
    with caplog.at_level(logging.INFO, logger=LOG):
        a_go(lab, r, did[len(way) :], went)
    assert lab.game.state.lives == START_LIVES - 1 and r.remembered == len(way)
    short = went - takes - LEAD
    assert route.moves == tuple(way[:short]) and route.story == route.moves
    assert CUT % (went, short, len(way)) in said(caplog)
    r.finish()
    # The next go begins with what is left, and arrives alive with thirty steps in
    # hand: standing still there it keeps its life.
    a_go(lab, r, [lab.move()] * 200, short + takes + LEAD + 20)
    assert r.remembered == short and lab.game.state.lives == START_LIVES
    assert lab.game.state.mode == PLAYING
    r.finish()


def test_a_life_that_goes_while_the_way_is_played_is_not_trusted_from_before_the_mistake(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    r, route, harness, did = a_game(lab, monkeypatch, LONG)
    went, takes = WENT[LONG], TAKES[LONG]
    a_go(lab, r, did, went)  # a first go: the game shows how long it takes
    assert r.dying_steps == takes
    r.finish()
    planted(route, harness, did[: went + 10])  # a way that walks into the walker
    with caplog.at_level(logging.INFO, logger=LOG):
        a_go(lab, r, [], went)
    assert r.remembered == went + 10 and lab.game.state.lives == START_LIVES - 1
    lines = said(caplog)
    assert (
        f"A life went at step {went} while the AI was playing the way it remembers, so it "
        f"no longer trusts that way from {takes + LEAD} steps before there."
    ) in lines
    # And as this is the second loss that stands at the place, twice the lead before
    # the mistake (T-1029), which here is before the start.
    assert CUT % (went, 0, went - takes - LEAD) in lines
    assert went - takes - 2 * LEAD < 0 and route.moves == ()
    r.finish()


def test_a_second_loss_at_the_place_backs_off_twice_as_far_before_the_mistake(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    r, route, _, did = a_game(lab, monkeypatch, toy_hopper.DYING_FRAMES)
    went, takes = WENT[toy_hopper.DYING_FRAMES], TAKES[toy_hopper.DYING_FRAMES]
    a_go(lab, r, did, went)
    assert route.story == tuple(did[: went - takes - LEAD])
    r.finish()
    # The same loss again: twice the lead (T-1029), before the mistake as well.
    with caplog.at_level(logging.INFO, logger=LOG):
        a_go(lab, r, did[len(route.moves) :], went)
    assert lab.game.state.lives == START_LIVES - 1
    assert route.story == tuple(did[: max(0, went - takes - 2 * LEAD)])
    lines = said(caplog)
    assert any(f"{2 * LEAD} steps before the place instead of {LEAD}" in line for line in lines)
    # The pause is said once in a game, not at every loss.
    assert [line for line in lines if "over a lost life, and nothing pressed" in line] == []
    r.finish()
