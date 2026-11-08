"""T-843: a burst of updates becomes one refresh."""

from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.ui.coalesce import DEFAULT_DELAY_MS, REST_DELAY_MS, Coalescer


def test_first_request_runs_at_once_and_a_burst_runs_once_more(qtbot: QtBot) -> None:
    ran: list[int] = []
    c = Coalescer(lambda: ran.append(1), delay_ms=50)
    assert c.delay_ms == 50 and not c.pending and c.runs == 0
    c.request()
    assert ran == [1] and not c.pending  # the first one shows straight away
    for _ in range(10):
        c.request()
    assert ran == [1] and c.pending  # the burst waits for the window to close
    qtbot.waitUntil(lambda: not c.pending, timeout=2000)
    assert ran == [1, 1] and c.runs == 2
    c.request()
    assert c.pending  # still inside the window of the run just done
    c.flush()
    assert ran == [1, 1, 1] and not c.pending
    c.request()
    c.cancel()
    assert not c.pending
    assert Coalescer(lambda: None).delay_ms == DEFAULT_DELAY_MS
    with pytest.raises(ConfigurationError):
        Coalescer(lambda: None, delay_ms=0)


def test_a_named_coalescer_times_its_runs() -> None:
    """T-906: a slow panel refresh is named on the stall line."""
    from retro_ai_lab.core.worklog import WorkLog

    work = WorkLog()
    named = Coalescer(lambda: None, name="rewards refresh", worklog=work)
    assert named.name == "rewards refresh"
    named.request()
    assert work.stats["rewards refresh"].count == 1
    plain = Coalescer(lambda: None, worklog=work)
    assert plain.name is None
    plain.request()
    assert set(work.stats) == {"rewards refresh"}
    with pytest.raises(ConfigurationError, match="blank"):
        Coalescer(lambda: None, name=" ")


def test_a_resting_coalescer_waits_longer_and_wakes(qtbot: QtBot) -> None:
    """T-944: while the AI practises fast a panel refreshes once every 2 s; waking it
    puts its own delay back and runs what was waiting."""
    ran: list[int] = []
    c = Coalescer(lambda: ran.append(1), delay_ms=50)
    assert REST_DELAY_MS == 2000 and not c.resting
    c.rest(True)
    assert c.resting and c.delay_ms == REST_DELAY_MS
    c.request()  # the first request of a clear window still runs at once
    c.request()
    assert ran == [1] and c.pending
    qtbot.wait(120)
    assert ran == [1]  # 50 ms would have run it; the rest holds it
    c.rest(False)  # the window it was waiting on has long passed: the wake runs it now
    assert not c.resting and c.delay_ms == 50 and not c.pending and ran == [1, 1]
    c.rest(True, delay_ms=80)
    assert c.resting and c.delay_ms == 80
    c.rest(False)
    assert c.delay_ms == 50
    with pytest.raises(ConfigurationError):
        c.rest(True, delay_ms=0)
