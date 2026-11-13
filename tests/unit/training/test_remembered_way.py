"""T-1014: a go begins by playing the furthest way the runner remembers."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.training import EpisodeRunner
from retro_ai_lab.training.episode import DEFAULT_UNSTICK_AFTER, EpisodeResult
from retro_ai_lab.training.moments import LEAD, Moments
from retro_ai_lab.training.route import Route
from retro_ai_lab.understanding import VariableKind
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, FRAMES, SLOTS, Lab, steps
from tests.unit.training.test_search_at_a_loss import GAP_ENDS

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

GREW = "The AI did better than the way it remembered: it now remembers %d steps from the start"


def remembering(
    lab: Lab, *, route: Route | None = None, **kw: object
) -> tuple[EpisodeRunner, Route, ExperimentHarness]:
    """A runner whose goes start from a checkpoint at the start of the level, keep
    moments, watch the screen, may search once, and remember the furthest way."""
    if lab.knowledge.best(VariableKind.LIVES) is None:
        lab.teach()
    lab.game.reset()
    watch = lab.watch()
    harness = ExperimentHarness(lab.connection, on_restore=lab.observation.expect_jump)
    harness.checkpoint()
    lab.agent.script.clear()
    way = Route() if route is None else route
    kw.setdefault("ways_per_go", 1)
    kw.setdefault("moments", Moments(SLOTS))
    kw.setdefault("watch", watch)
    return lab.runner(harness=harness, route=way, **kw), way, harness


def plays_from_the_start(lab: Lab, harness: ExperimentHarness, moves: Sequence[int]) -> None:
    """Play ``moves`` in the game itself from the checkpoint, off any record."""
    harness.restore()
    buttons = {b.name: b for b in lab.adapter.buttons()}
    for move in moves:
        lab.adapter.step(FRAMES, pressed=[buttons[n] for n in sorted(lab.space.buttons_for(move))])


def across(lab: Lab, r: EpisodeRunner, beyond: int = 16) -> None:
    """Step until the player stands well past the first gap."""
    while lab.game.state.x <= GAP_ENDS + beyond:
        assert r.step()


def planted(route: Route, harness: ExperimentHarness, moves: Sequence[int]) -> None:
    """Have ``route`` remember ``moves`` from the harness's checkpoint, a point paid
    with each."""
    route.begin((harness.slot, harness.taken))
    for count, move in enumerate(moves, start=1):
        route.did(move, count)
    assert route.end() and route.moves == tuple(moves)


def test_a_go_leaves_the_way_that_got_across(lab: Lab, caplog: pytest.LogCaptureFixture) -> None:
    r, route, harness = remembering(lab)
    walk, jump = lab.move("right"), lab.move("a", "right")
    assert r.route is route and harness.taken == 1
    r.begin()
    assert (r.remembered, route.moves, lab.agent.asked) == (0, (), 0)
    steps(r, DIED_AT)  # walks into the gap; goes back and finds the way
    assert (r.ways_sought, r.ways_found) == (1, 1) and route.story == ()
    across(lab, r)
    played = r.steps - DIED_AT
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        result = r.finish()
    assert result.remembered == 0 and result.ended_by == "stopped"
    # What it remembers is the way that got across: none of the 36 steps that fell in.
    way = route.moves
    assert way[:23] == (walk,) * 19 + (jump,) * 4 and set(way[23:]) == {walk}
    assert 23 < len(way) <= played and route.paid >= 2
    assert [rec.getMessage() for rec in caplog.records if "remembers" in rec.getMessage()] == [
        (GREW % len(way)) + ", and its next go begins by playing them again."
    ]
    # Played from the start in the game itself, it is across with every life.
    plays_from_the_start(lab, harness, way)
    assert lab.game.state.x > GAP_ENDS and lab.game.state.lives == toy_hopper.START_LIVES


def test_the_next_go_begins_with_it(lab: Lab) -> None:
    r, route, _ = remembering(lab)
    r.begin()
    steps(r, DIED_AT)
    across(lab, r)
    r.finish()
    way = route.moves
    # The second go: the same spot, and the way first.
    learned, asked = len(lab.agent.learned), lab.agent.asked
    r.begin()
    assert r.remembered == route.playing == len(way) > 23
    assert lab.game.state.x < 60 and lab.agent.asked == asked
    played = []
    for _ in way:
        assert r.step()
        played.append(r.last_action)
    # Its moves, in order, without asking the learner.
    assert tuple(played) == way and lab.agent.asked == asked and r.steps == len(way)
    # Across the first gap with every life, and with its search unused.
    assert lab.game.state.x > GAP_ENDS and lab.game.state.lives == toy_hopper.START_LIVES
    assert (r.ways_sought, r.times_back) == (0, 0)
    # Then it is the learner's go again.
    assert r.step() and lab.agent.asked == asked + 1
    result = r.finish()
    assert result.remembered == len(way)
    # The steps of the way were learned from like any step (the last few of a go are
    # learned when it ends, which is why this is looked at here).
    taught = [action for _, action, *_ in lab.agent.learned[learned:]]
    assert tuple(taught[: len(way)]) == way and len(taught) == len(way) + 1
    assert result.describe_remembered() == f", began with the {len(way)} steps it remembered"


def test_only_a_go_that_gets_further_replaces_it(lab: Lab) -> None:
    r, route, harness = remembering(lab)
    r.begin()
    steps(r, DIED_AT)
    across(lab, r)
    r.finish()
    first, counted = route.moves, route.paid
    # A go stopped part-way through the way, and one stopped at its end: kept as it was.
    for upto in (5, len(first)):
        r.begin()
        steps(r, upto)
        r.finish()
        assert (route.moves, route.paid) == (first, counted)
    # A go that walks on further: the way grows, and begins as it did.
    r.begin()
    steps(r, len(first))
    reached = lab.game.state.x
    steps(r, 12)
    assert lab.game.state.x > reached + 40
    r.finish()
    longer = route.moves
    assert len(first) < len(longer) <= len(first) + 12 and longer[: len(first)] == first
    assert route.paid > counted
    plays_from_the_start(lab, harness, longer)
    assert lab.game.state.x > reached + 40 and lab.game.state.lives == toy_hopper.START_LIVES


def test_a_loss_that_stands_ends_the_way_before_it(lab: Lab) -> None:
    r, route, harness = remembering(lab, ways_per_go=0)
    walk = lab.move("right")
    r.begin()
    steps(r, DIED_AT)  # into the gap, and nothing here goes back
    assert r.ways_sought == 0 and lab.game.state.lives == toy_hopper.START_LIVES - 1
    # The story stopped thirty steps before the life went, and takes no more.
    assert LEAD == 30 and route.story == (walk,) * (DIED_AT - LEAD)
    steps(r, 10)
    assert route.story == (walk,) * (DIED_AT - LEAD)
    r.finish()
    way = route.moves
    assert 0 < len(way) <= DIED_AT - LEAD and set(way) == {walk}
    # The next go arrives at the end of it alive, short of the gap.
    r.begin()
    steps(r, len(way))
    assert lab.game.state.lives == toy_hopper.START_LIVES
    assert 32 < lab.game.state.x < toy_hopper.TILE * lab.game.gaps[0][0]
    r.finish()
    plays_from_the_start(lab, harness, way)
    assert lab.game.state.lives == toy_hopper.START_LIVES


def test_a_life_lost_while_it_is_played_cuts_it(lab: Lab, caplog: pytest.LogCaptureFixture) -> None:
    r, route, harness = remembering(lab)
    walk, jump = lab.move("right"), lab.move("a", "right")
    # It remembers a way that does not hold: forty steps of walking, into the gap.
    planted(route, harness, [walk] * 40)
    r.begin()
    assert r.remembered == 40
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        steps(r, DIED_AT)
    # The life went on its 36th step: it is cut thirty steps before that, at once.
    assert route.moves == (walk,) * (DIED_AT - LEAD)
    said = [rec.getMessage() for rec in caplog.records if "no longer trusts" in rec.getMessage()]
    assert said == [
        f"A life went at step {DIED_AT} while the AI was playing the way it remembers, "
        "so it no longer trusts that way from 30 steps before there."
    ]
    # The go carries on like any other: it went back, found the way, and plays that.
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    asked = lab.agent.asked
    assert r.step() and r.last_action == walk and lab.agent.asked == asked  # the way found
    across(lab, r)
    assert lab.game.state.lives == toy_hopper.START_LIVES
    r.finish()
    # It got further than what was left of the old way: now it remembers the one that works.
    way = route.moves
    assert jump in way and len(way) > DIED_AT - LEAD
    plays_from_the_start(lab, harness, way)
    assert lab.game.state.x > GAP_ENDS and lab.game.state.lives == toy_hopper.START_LIVES


def test_nothing_is_remembered_where_it_cannot_be_played_again(lab: Lab) -> None:
    walk = lab.move("right")
    # No route: the runner is as it was.
    r, route, harness = remembering(lab)
    plain = lab.runner(harness=harness, watch=r.watch)
    assert plain.route is None
    plain.begin()
    steps(plain, 12)
    assert plain.remembered == 0 and plain.finish().remembered == 0
    for wrong in ("way", 3, [1, 2]):
        with pytest.raises(ConfigurationError, match="route"):
            lab.runner(route=wrong)
    # No screen watch: nothing says how far a go got, so nothing is kept or played.
    blind = lab.runner(harness=harness, route=route)
    blind.begin()
    steps(blind, 12)
    blind.finish()
    assert route.moves == ()
    # A go that carries on from wherever the game is, not from the checkpoint.
    r.begin(restore=False)
    steps(r, 12)
    r.finish()
    assert route.moves == () and r.remembered == 0
    # A go with everything: remembered. Then somebody else drives a step: the story
    # ends there, and what the go did afterwards is not part of it.
    r.begin()
    steps(r, 10)
    assert route.story == (walk,) * 10
    assert r.observe_step(["right"], FRAMES)
    steps(r, 6)
    assert route.story == (walk,) * 10
    r.finish()
    kept = route.moves
    assert 0 < len(kept) <= 10
    # A way planted with moves these controls do not have is dropped, not played.
    planted(route, harness, [walk, len(lab.space) + 3, walk])
    r.begin()
    assert r.remembered == 0 and route.moves == ()
    r.finish()
    # The checkpoint saved anew: what was remembered from the old one is forgotten.
    planted(route, harness, [walk] * 8)
    harness.checkpoint()
    assert harness.taken == 2
    r.begin()
    assert r.remembered == 0 and route.moves == ()
    r.finish()


def test_a_list_of_moves_is_not_ended_by_standing_still(lab: Lab) -> None:
    r, route, harness = remembering(lab)
    walk, stand = lab.move("right"), lab.move()
    # A way that stood about for longer than ends a go "stuck" before it got on.
    still = DEFAULT_UNSTICK_AFTER + 15
    planted(route, harness, [stand] * still + [walk] * 10)
    r.begin()
    steps(r, still)
    assert lab.game.state.x == 32 and lab.agent.asked == 0  # not a step of its own yet
    steps(r, 10)
    assert lab.game.state.x > 60 and r.remembered == still + 10
    r.finish()
    # With no list to play, a go that stands still that long ends "stuck" as before.
    fresh, _, _ = remembering(lab)
    lab.agent.script.extend([stand] * 200)
    fresh.begin()
    count = 0
    while fresh.step():
        count += 1
    assert fresh.finish().ended_by == "stuck" and count == DEFAULT_UNSTICK_AFTER


def test_the_steps_from_memory_do_not_count_towards_the_gos_length(lab: Lab) -> None:
    """T-1015: a way longer than a go would leave the go no steps of its own."""
    r, route, harness = remembering(lab, max_steps=12)
    walk, stand = lab.move("right"), lab.move()
    planted(route, harness, [walk] * 20)
    lab.agent.script.extend([stand] * 50)
    r.begin()
    assert (r.remembered, r.steps, r.own_steps) == (20, 0, 0)
    steps(r, 20)  # the whole way, past where a go of 12 steps used to end
    assert (r.steps, r.own_steps, lab.agent.asked) == (20, 0, 0)
    count = 0
    while r.step():
        count += 1
    assert (count, r.steps, r.own_steps, lab.agent.asked) == (11, 32, 12, 12)
    result = r.finish()
    assert (result.ended_by, result.steps, result.remembered) == ("max_steps", 32, 20)
    # A go with nothing to play from memory is as long as it always was.
    route.forget()
    r.begin()
    count = 0
    while r.step():
        count += 1
    result = r.finish()
    assert (count, result.ended_by, result.steps, result.remembered) == (11, "max_steps", 12, 0)


def test_a_life_lost_close_behind_the_way_with_no_way_past_cuts_it(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    """T-1015: a way that ends a few steps short of a loss the AI finds no way past
    would bring every go to the same end. It is cut back, so the next go arrives
    earlier, with room to do something else."""
    r, route, harness = remembering(lab, ways_per_go=0)  # no search: the loss stands
    walk = lab.move("right")
    planted(route, harness, [walk] * 30)  # six steps short of the fall
    lab.agent.script.extend([walk] * 20)
    r.begin()
    assert r.remembered == 30
    steps(r, 30)
    assert lab.agent.asked == 0 and route.moves == (walk,) * 30
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        steps(r, DIED_AT - 30)  # its own steps: into the gap, and the life goes
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1 and r.ways_sought == 0
    assert route.moves == (walk,) * (DIED_AT - LEAD)
    said = [rec.getMessage() for rec in caplog.records]
    assert said == [
        f"A life went at step {DIED_AT}, close behind the end of the way the AI remembers, "
        f"and it found no way past: it now remembers {DIED_AT - LEAD} steps of that way "
        "instead of 30, to arrive there earlier."
    ]
    r.finish()
    assert route.moves == (walk,) * (DIED_AT - LEAD)  # the go that lost the life adds nothing
    # A life lost well beyond the way's end leaves the way alone.
    far, route, harness = remembering(lab, ways_per_go=0)
    planted(route, harness, [walk] * 4)
    lab.agent.script.extend([walk] * 40)
    far.begin()
    steps(far, DIED_AT)
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1
    assert route.moves[:4] == (walk,) * 4 and len(route.moves) >= 4
    far.finish()
    # Its own story ends 30 steps before the loss, and what is kept of it is up to the
    # last of those steps that was paid for (the fifth).
    assert route.story == (walk,) * (DIED_AT - LEAD) and route.moves == (walk,) * 5


def test_the_record_counts_the_steps_from_memory() -> None:
    result = EpisodeResult(10, 40, 1.0, "max_steps", 3, remembered=187)
    assert EpisodeResult.from_dict(result.to_dict()) == result
    older = result.to_dict()
    del older["remembered"]
    assert EpisodeResult.from_dict(older).remembered == 0
    assert result.describe_remembered() == ", began with the 187 steps it remembered"
    assert EpisodeResult(1, 4, 0.0, "stuck", 1).describe_remembered() == ""
    assert EpisodeResult(1, 4, 0.0, "stuck", 1, remembered=1).describe_remembered() == (
        ", began with the step it remembered"
    )
