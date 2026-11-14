"""T-1027: when a reward row goes, the runner takes what it paid out of the way it
remembers, at once, and says so."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.rewards import GROUND_ID
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.services.rewards import RewardProfileChanged
from retro_ai_lab.training.route import Route
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import Lab, steps
from tests.unit.training.test_remembered_way import remembering

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

EPISODE = "retro_ai_lab.training.episode"


def said(caplog: pytest.LogCaptureFixture) -> list[str]:
    """What the runner's log said of rewards that are gone."""
    lines = [rec.getMessage() for rec in caplog.records if rec.name == EPISODE]
    return [line for line in lines if line.startswith("Gone from the rewards")]


def gone(lab: Lab, *rows: str) -> None:
    """The rewards service's own word that these rows are gone."""
    lab.bus.publish(RewardProfileChanged((), (), rows, "suggestions"))


def planted(route: Route, harness: ExperimentHarness, move: int, ground: int, clock: int) -> None:
    """Have ``route`` remember ``ground`` steps of ``move`` paid a point each by
    "ground", then ``clock`` steps paid ten each by "clock"."""
    route.forget()
    route.begin((harness.slot, harness.taken))
    total = 0.0
    for at in range(ground + clock):
        row, amount = ("ground", 1.0) if at < ground else ("clock", 10.0)
        total += amount
        route.did(move, total, [(row, amount)])
    assert route.end() and len(route.moves) == ground + clock
    assert route.paid == ground + 10.0 * clock


def test_what_the_practice_game_pays_comes_with_its_row(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, route, _ = remembering(lab, ways_per_go=0)
    walk = lab.move("right")
    lab.agent.script.extend([walk] * 14)
    r.begin()
    steps(r, 14)
    assert route.counted > 0
    r.finish()
    # Every point of the way has the row that paid it: here, new ground.
    assert route.paid > 0 and len(route.paid_by) == len(route.moves)
    parts = [part for step in route.paid_by for part in step]
    assert {row for row, _ in parts} == {GROUND_ID}
    assert sum(amount for _, amount in parts) == route.paid
    # That row gone, the log calls it by the name it paid under.
    paid = route.paid
    gone(lab, GROUND_ID)
    with caplog.at_level(logging.INFO, logger=EPISODE):
        r.begin()
    assert said(caplog) == [
        "Gone from the rewards: Getting somewhere new. Nothing else had paid for the way "
        f"the AI remembers ({paid:.1f} points), so it forgets that way, and its next go "
        "begins without one."
    ]
    assert r.remembered == 0


def test_the_way_is_valued_again_when_a_row_goes_and_the_log_says_so(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, route, harness = remembering(lab, ways_per_go=0)
    walk = lab.move("right")
    lab.agent.script.extend([walk] * 60)

    # Between goes, with every step of the way paid by something else as well (the
    # rewards service's word is all it takes: nothing has to be played again).
    route.begin((harness.slot, harness.taken))
    for at in range(10):
        route.did(walk, 5.0 * (at + 1), [("ground", 1.0), ("clock", 4.0)])
    assert route.end() and route.paid == 50.0
    gone(lab, "clock", "something else")
    with caplog.at_level(logging.INFO, logger=EPISODE):
        r.begin()
    assert said(caplog) == [
        "Gone from the rewards: clock. What that paid is taken out of the way the AI "
        "remembers: it is worth 10.0 points now, not 50.0."
    ]
    assert r.remembered == 10 and route.paid == 10.0
    r.finish()

    # In the middle of a go, with the end of the way paid by nothing else.
    planted(route, harness, walk, 6, 4)
    r.begin()
    assert r.remembered == 10
    steps(r, 3)
    gone(lab, "clock")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=EPISODE):
        steps(r, 1)
    assert said(caplog) == [
        "Gone from the rewards: clock. What that paid is taken out of the way the AI "
        "remembers: it is worth 6.0 points now, not 46.0, and the AI remembers 6 steps of "
        "it instead of 10, up to where it is paid the most."
    ]
    assert route.moves == (walk,) * 6
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=EPISODE):
        steps(r, 2)  # said once
    assert said(caplog) == []
    r.finish()

    # One step left.
    planted(route, harness, walk, 1, 3)
    gone(lab, "clock")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=EPISODE):
        r.begin()
    assert said(caplog) == [
        "Gone from the rewards: clock. What that paid is taken out of the way the AI "
        "remembers: it is worth 1.0 points now, not 31.0, and the AI remembers 1 step of "
        "it instead of 4, up to where it is paid the most."
    ]
    assert r.remembered == 1
    r.finish()

    # Just before a go's end, and nothing else had paid for the way.
    planted(route, harness, walk, 0, 4)
    r.begin()
    steps(r, 4)
    gone(lab, "clock")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=EPISODE):
        r.finish()
    assert said(caplog) == [
        "Gone from the rewards: clock. Nothing else had paid for the way the AI remembers "
        "(40.0 points), so it forgets that way, and its next go begins without one."
    ]
    # A row that had paid for nothing in the way: not a word.
    way = route.moves
    gone(lab, "lives")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=EPISODE):
        r.begin()
    assert said(caplog) == [] and route.moves == way
