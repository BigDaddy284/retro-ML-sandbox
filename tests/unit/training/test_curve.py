"""T-601: the learning curve in numbers and in words."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training import EpisodeRecord, EpisodeResult, LearningCurve, Mode
from retro_ai_lab.training.curve import IMPROVING, SLIPPING, STEADY, TOO_EARLY


def go(n: int, reward: float, steps: int = 10, objectives: int = 0) -> EpisodeRecord:
    return EpisodeRecord(
        n, Mode.GUIDED, EpisodeResult(steps, steps * 4, reward, "max_steps", 2, objectives), n * 40
    )


def curve(*rewards: float, window: int = 3, **kw: int) -> LearningCurve:
    return LearningCurve([go(i + 1, r, **kw) for i, r in enumerate(rewards)], window)


def test_points_averages_best_and_longest_match_by_hand() -> None:
    goes = [go(1, 2.0, 5), go(2, 4.0, 9, objectives=1), go(3, -3.0, 7), go(4, 9.0, 8)]
    c = LearningCurve(goes, window=2)
    assert c.attempts == 4 and c.window == 2
    assert c.rewards == (2.0, 4.0, -3.0, 9.0)
    assert [p.average for p in c.points] == [2.0, 3.0, 0.5, 3.0]
    assert [p.number for p in c.points] == [1, 2, 3, 4]
    assert c.best is not None and c.best.number == 4 and c.best.reward == 9.0
    assert c.longest is not None and c.longest.number == 2 and c.longest.steps == 9
    assert c.latest_average == 3.0 and c.earlier_average == 3.0
    assert c.successes == 1 and c.success_rate == 0.25


def test_empty_and_single_go_curves_are_safe() -> None:
    empty = LearningCurve([])
    assert empty.points == () and empty.rewards == () and empty.attempts == 0
    assert empty.best is None and empty.longest is None
    assert empty.latest_average is None and empty.earlier_average is None
    assert empty.trend == TOO_EARLY and empty.success_rate == 0.0
    assert empty.describe() == "No goes yet."
    one = curve(5.0)
    assert one.latest_average == 5.0 and one.earlier_average is None
    assert one.describe() == (
        "1 go so far; the best scored +5 points (go 1). The last 1 averaged +5.0 points."
    )
    with pytest.raises(ConfigurationError):
        LearningCurve([], window=0)


def test_trend_needs_two_windows_then_compares_them() -> None:
    assert curve(1, 2, 3, 4, 5).trend == TOO_EARLY  # fewer than 2 * 3 goes
    assert curve(1, 1, 1, 5, 5, 5).trend == IMPROVING
    assert curve(5, 5, 5, 1, 1, 1).trend == SLIPPING
    assert curve(5, 5, 5, 5, 5, 5).trend == STEADY  # no spread at all
    # A change of less than a tenth of the spread is noise.
    assert curve(0, 10, 0, 0, 10, 0.5).trend == STEADY
    assert curve(0, 10, 0, 5, 10, 0).trend == IMPROVING
    # Only the latest window and the one before count, not the whole history.
    assert curve(9, 9, 9, 1, 1, 1, 4, 4, 4).trend == IMPROVING


def test_descriptions_read_naturally() -> None:
    assert curve(1, 1, 1, 5, 5, 5, objectives=1).describe() == (
        "6 goes so far; the best scored +5 points (go 4). Reached a goal in 6 of them (100%). "
        "Getting better: the last 3 averaged +5.0 points, up from +1.0."
    )
    assert curve(5, 5, 5, 1, 1, 1).describe() == (
        "6 goes so far; the best scored +5 points (go 1). "
        "Slipping: the last 3 averaged +1.0 points, down from +5.0."
    )
    assert curve(2, 2, 2, 2, 2, 2).describe() == (
        "6 goes so far; the best scored +2 points (go 1). Holding steady at about +2.0 points a go."
    )
    assert curve(-4, -2).describe() == (
        "2 goes so far; the best scored -2 points (go 2). The last 2 averaged -3.0 points."
    )


def test_results_carry_objectives_and_old_files_load_with_none() -> None:
    r = EpisodeResult(10, 40, 1.0, "reset", 3)
    assert r.objectives == 0 and not r.succeeded
    assert EpisodeResult.from_dict(r.to_dict()) == r
    old = {"steps": 1, "frames": 4, "reward": 0.0, "ended_by": "stopped", "states_seen": 1}
    assert EpisodeResult.from_dict(old).objectives == 0
    assert EpisodeResult(1, 4, 0.0, "reset", 1, 2).succeeded
