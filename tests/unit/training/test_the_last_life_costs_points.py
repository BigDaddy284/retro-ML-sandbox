"""T-1053: through the training service, the last life costs points and pays none.

On a game that counts its lives in reserve the count leaves 0 for 255 when the last
one goes. Every go that ended by game over was paid "Gaining Lives" for that, and
neither "Losing Lives" nor a person's own "goes down" row paid for the life.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_racer
from retro_ai_lab.training.episode import EpisodeResult
from retro_ai_lab.understanding import VariableKind
from tests.unit.training import test_each_game_over_is_the_next_go as in_reserve
from tests.unit.training.test_each_game_over_is_the_next_go import RESERVE, WHY
from tests.unit.training.test_service import Lab

lab = in_reserve.lab  # the practice racer behind the training service (pytest fixture)
counted_in_reserve = in_reserve.counted_in_reserve  # ... counting its lives as his game does

LOSING, GAINING, HIS = "Losing Lives", "Gaining Lives", "Lifes goes down"
EVERY = RESERVE + 1  # the lives a game has: those in reserve and the one being played


def paid(result: EpisodeResult, row: str) -> tuple[int, float]:
    """How often the row of that name paid in the go, and how much in all."""
    found = [e for e in result.earned if e.name == row]
    assert len(found) <= 1
    return (found[0].count, found[0].amount) if found else (0, 0.0)


def test_every_life_costs_and_none_is_gained(
    lab: Lab, counted_in_reserve: None, caplog: pytest.LogCaptureFixture
) -> None:
    lab.connect()
    t = lab.training
    t.set_go_until_over(True)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.service"):
        results = [r.result for r in t.train(12)]
    over = [r for r in results if r.ended_by == "game_over"]
    assert len(over) >= 3 and {r.setbacks for r in over} == {EVERY}  # (as T-1050 left it)
    for result in results:
        # Each life lost is paid for once, the last among them, and none is gained.
        assert paid(result, LOSING) == (result.setbacks, -10.0 * result.setbacks)
        assert paid(result, GAINING) == (0, 0.0)
    said = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("Go ")]
    ended = [line for line in said if " ended by game_over after " in line]
    assert len(ended) == len(over)
    for line in ended:
        assert f"{LOSING} -{10 * EVERY}.0 ({EVERY} times), " in line
        assert f", lost a life {EVERY} times, " in line and WHY in line
    assert not any(GAINING in line for line in said)


def test_his_own_row_is_paid_for_the_last_life_too(lab: Lab, counted_in_reserve: None) -> None:
    knowledge = lab.connect()
    lives = knowledge.best(VariableKind.LIVES)
    assert lives is not None
    lab.rewards.add_own(lives.id, goes_up=False, value=-7.0, name="Lifes")
    t = lab.training
    t.set_go_until_over(True)
    results = [r.result for r in t.train(12)]
    over = [r for r in results if r.ended_by == "game_over"]
    assert len(over) >= 2 and {r.setbacks for r in over} == {EVERY}
    for result in results:
        assert paid(result, HIS) == (result.setbacks, -7.0 * result.setbacks)
        assert paid(result, LOSING) == (result.setbacks, -10.0 * result.setbacks)
        assert paid(result, GAINING) == (0, 0.0)


def test_the_practice_game_as_it_is_is_paid_as_before(lab: Lab) -> None:
    # Its own count stops at nought: nothing falls below it, and nothing is read anew.
    lab.connect()
    t = lab.training
    t.set_go_until_over(True)
    results = [r.result for r in t.train(6)]
    assert max(r.setbacks for r in results) == toy_racer.START_LIVES
    for result in results:
        assert paid(result, LOSING) == (result.setbacks, -10.0 * result.setbacks)
        assert paid(result, GAINING) == (0, 0.0)
    assert [r.reward for r in results] == AS_BEFORE


AS_BEFORE = [35.0, 0.0, 7.0, 0.0, 0.0, -18.0]  # the six goes' pay on main before this task
