"""T-009: EventBridge delivers bus events on the GUI thread."""

from __future__ import annotations

import gc
import sys
import threading
from dataclasses import dataclass

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.ui.bridge import EventBridge


@dataclass(frozen=True)
class Ping:
    n: int


def test_same_thread_events_are_delivered_immediately(qapp: QApplication) -> None:
    bus = EventBus()
    bridge = EventBridge(bus)
    seen: list[Ping] = []
    bridge.subscribe(Ping, seen.append)
    bus.publish(Ping(1))
    assert seen == [Ping(1)]  # no event loop spin needed


def test_worker_thread_events_arrive_on_gui_thread(qtbot: QtBot) -> None:
    bus = EventBus()
    bridge = EventBridge(bus)
    seen: list[tuple[Ping, bool]] = []
    gui_thread = QThread.currentThread()
    bridge.subscribe(Ping, lambda e: seen.append((e, QThread.currentThread() is gui_thread)))

    worker = threading.Thread(target=lambda: bus.publish(Ping(2)))
    worker.start()
    worker.join()
    assert seen == []  # not yet: it is queued for the GUI thread
    qtbot.waitUntil(lambda: len(seen) == 1)
    assert seen == [(Ping(2), True)]


def test_unsubscribe_and_counts(qapp: QApplication) -> None:
    bus = EventBus()
    bridge = EventBridge(bus)
    seen: list[Ping] = []
    off = bridge.subscribe(Ping, seen.append)
    assert bridge.handler_count(Ping) == 1
    off()
    assert bridge.handler_count() == 0
    bus.publish(Ping(3))
    assert seen == []
    bridge.subscribe(Ping, seen.append)
    bridge.unsubscribe(Ping, seen.append)
    bus.publish(Ping(4))
    assert seen == []


def test_detach_stops_forwarding_but_local_publish_still_works(qapp: QApplication) -> None:
    bus = EventBus()
    bridge = EventBridge(bus)
    seen: list[Ping] = []
    bridge.subscribe(Ping, seen.append)
    bridge.detach()
    bus.publish(Ping(5))
    assert seen == []
    assert bridge.publish_local(Ping(6)) == 1
    assert seen == [Ping(6)]


def test_deleted_bridge_stops_forwarding(qapp: QApplication) -> None:
    """T-202: a bridge whose Qt side is gone must not break the bus."""
    import shiboken6

    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    bridge = EventBridge(bus)
    seen: list[Ping] = []
    bridge.subscribe(Ping, seen.append)
    bus.publish(Ping(1))
    shiboken6.delete(bridge)
    assert bus.publish(Ping(2)) == 0  # handler removed, nothing raised
    assert seen == [Ping(1)]


def test_forwarding_thousands_of_events_leaks_no_references(qapp: QApplication) -> None:
    """Guards against a Qt binding that drops a reference per signal emission.

    PySide6 6.12.0 on Python 3.11 lost one reference to ``True`` for every
    ``emit``; after a few thousand events the interpreter aborted. Training sends
    thousands of events per minute, so the bridge must be reference-neutral.
    """
    bus = EventBus()
    bridge = EventBridge(bus)
    got: list[Ping] = []
    bridge.subscribe(Ping, got.append)
    gc.collect()
    before = tuple(sys.getrefcount(x) for x in (True, False, None))
    for i in range(3000):
        bus.publish(Ping(i))
    gc.collect()
    after = tuple(sys.getrefcount(x) for x in (True, False, None))
    assert len(got) == 3000
    assert after == before, f"reference counts drifted (True, False, None): {before} -> {after}"


def test_deliveries_are_timed_by_event_name(qtbot: QtBot) -> None:
    """T-906: the stall line can name the handler behind a stall."""
    from retro_ai_lab.core.worklog import WorkLog

    bus = EventBus()
    work = WorkLog()
    bridge = EventBridge(bus, worklog=work)
    bridge.subscribe(Ping, lambda _: None)
    bus.publish(Ping(1))  # the GUI thread: delivered, and timed, at once
    assert work.stats["on Ping"].count == 1

    class Worker(QThread):
        def run(self) -> None:
            bus.publish(Ping(2))

    worker = Worker()
    worker.start()
    qtbot.waitUntil(lambda: work.stats["on Ping"].count == 2)
    worker.wait()
    bridge.publish_local(Ping(3))
    assert work.stats["on Ping"].count == 3
    bus.publish(object())  # no handler: still timed (and left off the line, costing nothing)
    assert work.stats["on object"].count == 1
