"""T-932: exploring follows results."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.exploring import NOTICEABLE, RESULTS_WINDOW, ResultsWatch


def test_lift_rises_when_the_latest_window_is_worse_and_eases_off_after() -> None:
    watch = ResultsWatch(0.25, window=3)
    assert watch.lift == 0.0 and watch.slipping() is None
    for reward in (9.0, 9.0, 9.0, 9.0, 9.0):
        assert watch.note(reward, goes=1) == 0.0
    assert watch.slipping() is None  # six needed for two windows
    # (9, 9, 9) then (9, 9, 3): worse by 2 on a spread of 6, more than NOTICEABLE's tenth.
    assert watch.note(3.0, goes=6) == 0.25
    assert watch.rewards == (9.0, 9.0, 9.0, 9.0, 9.0, 3.0)
    assert watch.slipping() == (7.0, 9.0)
    # The last two windows together are kept, never more.
    assert watch.note(3.0, goes=7) == 0.25 and watch.rewards == (9.0, 9.0, 9.0, 9.0, 3.0, 3.0)
    assert watch.note(3.0, goes=8) == 0.25
    # Recovered: the lift falls by a window's share each go that is not worse.
    assert watch.note(9.0, goes=9) == 0.25  # (9, 3, 3) vs (3, 3, 9): the same
    assert watch.note(9.0, goes=10) == pytest.approx(0.25 - 0.25 / 3)
    assert watch.note(9.0, goes=11) == pytest.approx(0.25 - 2 * 0.25 / 3)
    assert watch.note(9.0, goes=12) == 0.0
    assert watch.note(9.0, goes=13) == 0.0  # never below zero


def test_small_drops_are_noise_and_rebound_zero_turns_it_off() -> None:
    watch = ResultsWatch(0.5, window=2)
    for reward in (10.0, 0.0, 9.0, 0.0):
        watch.note(reward, goes=1)
    assert watch.slipping() is None  # 0.5 down on a spread of 10 is within NOTICEABLE's tenth
    assert watch.lift == 0.0
    assert NOTICEABLE == 0.1 and RESULTS_WINDOW == 10
    flat = ResultsWatch(0.5, window=2)
    for reward in (1.0, 1.0, 1.0, 1.0):
        assert flat.note(reward, goes=1) == 0.0  # no spread: nothing is worse
    off = ResultsWatch(0.0, window=2)
    for reward in (10.0, 10.0, 0.0, 0.0):
        assert off.note(reward, goes=1) == 0.0
    assert off.slipping() == (0.0, 10.0)  # it still sees the drop; it just does nothing


def test_rebound_can_change_and_the_watch_round_trips() -> None:
    watch = ResultsWatch(0.25, window=2)
    for reward in (10.0, 10.0, 0.0, 0.0):
        watch.note(reward, goes=1)
    assert watch.lift == 0.25
    watch.set_rebound(0.1)
    assert watch.lift == 0.1  # a lower rebound caps the lift at once
    again = ResultsWatch(0.1, window=2)
    again.load(watch.to_dict())
    assert again.lift == 0.1 and again.rewards == watch.rewards
    again.load(None)  # a brain saved before T-932 has nothing to load
    assert again.rewards == watch.rewards
    with pytest.raises(ConfigurationError):
        ResultsWatch(0.25, window=0)
    with pytest.raises(ConfigurationError):
        again.load({"rewards": "lots"})
    with pytest.raises(ConfigurationError):
        again.load({"rewards": [], "hold": 3})  # more goes of hold than the window has


def test_the_rise_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    watch = ResultsWatch(0.3, window=2)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.learning.exploring"):
        for go, reward in enumerate((10.0, 10.0, 0.0, 0.0, 0.0), start=1):
            watch.note(reward, goes=go)
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [
        "Exploring back up to 30% after go 4: the last 2 goes averaged +0.0 points, "
        "down from +10.0."
    ]
