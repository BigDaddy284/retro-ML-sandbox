"""T-843: a burst of updates becomes one refresh."""

from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.ui.coalesce import DEFAULT_DELAY_MS, Coalescer


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
