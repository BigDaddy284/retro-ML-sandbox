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
    # Another watch (a window left by an earlier test) may write its own line here;
    # this watch's line is the one that carries the "learn" timing noted above.
    line = next(
        r.message
        for r in caplog.records
        if r.message.startswith("App work") and "learn" in r.message
    )
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
    # A report is a line (StallWatch.report writes one per count), so none was written,
    # and there was nothing for one to say. A window an earlier test left behind may
    # write its own "App work" line meanwhile (seen on a slow CI runner); that line is
    # another watch's, as the busy test above allows for too.
    assert watch.reports == 0 and not work.stats
    assert not [r for r in caplog.records if "App work" in r.message and "learn" in r.message]
    watch.stop()


def test_bad_settings_are_refused() -> None:
    with pytest.raises(ValueError):
        StallWatch(WorkLog(), interval_ms=0)
    with pytest.raises(ValueError):
        StallWatch(WorkLog(), stall_s=0)
    with pytest.raises(ValueError):
        StallWatch(WorkLog(), report_every_s=-1)


def test_the_line_leaves_out_work_under_a_millisecond(
    qtbot: QtBot, caplog: pytest.LogCaptureFixture
) -> None:
    """T-906: a 2 ms handler is named; 300 free deliveries are not."""
    from retro_ai_lab.ui.stall_watch import DEFAULT_QUIET_S

    assert DEFAULT_QUIET_S == 0.001
    work = WorkLog()
    watch = StallWatch(work, interval_ms=5, report_every_s=0.05)
    # Names of this test's own, so the line read back is this watch's: a window an
    # earlier test left behind writes "App work" lines of its own on a slow CI runner,
    # and the first such line was once that window's (T-962's CI run).
    for _ in range(300):
        work.note("on QuietTestEvent", 0.0001)
    work.note("on LoudTestEvent", 0.002)
    watch.start()
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.ui.stall_watch"):
        qtbot.waitUntil(lambda: watch.reports >= 1, timeout=2000)
    watch.stop()
    line = next(
        r.message
        for r in caplog.records
        if r.message.startswith("App work") and "LoudTestEvent" in r.message
    )
    assert f"on LoudTestEvent {X}1 avg 2 ms" in line and "QuietTestEvent" not in line
    # Only quiet work: no line at all.
    work.note("on QuietTestEvent", 0.0001)
    watch = StallWatch(work, interval_ms=5, report_every_s=0.05)
    watch.start()
    qtbot.wait(150)
    assert watch.reports == 0
    watch.stop()
    with pytest.raises(ValueError):
        StallWatch(WorkLog(), quiet_under_s=-0.1)
