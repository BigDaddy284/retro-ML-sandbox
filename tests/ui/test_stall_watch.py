"""T-894: GUI stalls and the app's own work are logged next to the emulator link's."""

from __future__ import annotations

import gc
import logging
import time

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QTimer
from pytestqt.qtbot import QtBot

from retro_ai_lab.core.worklog import WorkLog
from retro_ai_lab.ui.stall_watch import STALL, StallWatch

X = "\N{MULTIPLICATION SIGN}"


def test_a_busy_gui_thread_is_a_stall_and_the_window_is_logged(
    qtbot: QtBot, caplog: pytest.LogCaptureFixture
) -> None:
    # Windows the tests before left behind are deleted and collected on the next turn
    # of the event loop; that would be a real stall, so it happens before the watch starts.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()
    qtbot.wait(20)
    work = WorkLog()
    # A long window: a slow machine must not report (and clear the stats) before the
    # stall is checked below.
    watch = StallWatch(work, interval_ms=5, stall_s=0.05, report_every_s=1.0)
    assert not watch.running and watch.stalls == 0 and watch.reports == 0
    watch.start()
    assert watch.running
    qtbot.wait(60)
    assert watch.stalls == 0  # a timer that keeps up is not a stall
    QTimer.singleShot(0, lambda: time.sleep(0.12))  # the GUI thread goes away for 120 ms
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.ui.stall_watch"):
        qtbot.waitUntil(lambda: watch.stalls >= 1, timeout=2000)
        stall = work.stats[STALL]
        assert stall.count == 1 and 0.1 <= stall.max_s < 0.5
        work.note("learn", 0.02)
        qtbot.waitUntil(lambda: watch.reports >= 1, timeout=4000)
    line = next(r.message for r in caplog.records if r.message.startswith("App work"))
    assert f"GUI stall {X}1" in line and f"learn {X}1 avg 20 ms" in line
    assert work.summary() == "nothing"  # a new window
    watch.stop()
    assert not watch.running


def test_an_idle_window_writes_no_line(qtbot: QtBot, caplog: pytest.LogCaptureFixture) -> None:
    work = WorkLog()
    watch = StallWatch(work, interval_ms=5, report_every_s=0.05)
    watch.start()
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.ui.stall_watch"):
        qtbot.wait(150)
    assert watch.reports == 0 and not [r for r in caplog.records if "App work" in r.message]
    watch.stop()


def test_bad_settings_are_refused() -> None:
    with pytest.raises(ValueError):
        StallWatch(WorkLog(), interval_ms=0)
    with pytest.raises(ValueError):
        StallWatch(WorkLog(), stall_s=0)
    with pytest.raises(ValueError):
        StallWatch(WorkLog(), report_every_s=-1)
