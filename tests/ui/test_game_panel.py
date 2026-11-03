"""T-008: the Game panel shows live frames while connected (T-817: off the GUI thread)."""

from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtCore import QTimer
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import Capabilities, EmulatorError
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.game_panel import NO_SCREEN, NOT_CONNECTED, FramePump, GamePanel
from retro_ai_lab.ui.main_window import MainWindow


@pytest.fixture
def context() -> AppContext:
    return AppContext()


@pytest.fixture
def panel(qtbot: QtBot, context: AppContext) -> GamePanel:
    p = GamePanel(context, fps=60)
    qtbot.addWidget(p)
    p.show()
    return p


def test_starts_disconnected(panel: GamePanel) -> None:
    assert panel.status_text == NOT_CONNECTED
    assert panel.view.frame is None
    assert panel.view.placeholder == NOT_CONNECTED
    assert not panel.pump.running


def test_connect_shows_first_frame_and_starts_pump(
    qtbot: QtBot, panel: GamePanel, context: AppContext
) -> None:
    panel.clock.pause()  # paused: the pump serves the screen
    context.connection.connect("fake", seed=2)
    assert panel.pump.running
    assert panel.pump.interval_ms == 1000 // 60
    qtbot.waitUntil(lambda: panel.view.frame is not None)
    assert panel.view.frame is not None
    assert panel.view.frame.number == 0
    assert panel.status_text == "Watching Toy Racer. Frame 0."


def test_frames_update_as_the_game_advances(
    qtbot: QtBot, panel: GamePanel, context: AppContext
) -> None:
    context.connection.connect("fake")
    panel.clock.pause()  # drive time by hand so the expected frame is exact
    context.connection.call(lambda a: a.advance_frames(5))
    qtbot.waitUntil(lambda: panel.view.frame is not None and panel.view.frame.number == 5)
    assert panel.status_text.endswith("Frame 5.")
    adapter = context.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    assert panel.view.frame == adapter.game.render()
    panel.clock.play()  # and with the clock running the frames keep coming
    qtbot.waitUntil(lambda: panel.view.frame is not None and panel.view.frame.number > 20)


def test_disconnect_stops_pump_and_clears(panel: GamePanel, context: AppContext) -> None:
    context.connection.connect("fake")
    context.connection.disconnect()
    assert not panel.pump.running
    assert panel.view.frame is None
    assert panel.view.placeholder == NOT_CONNECTED
    assert panel.status_text == NOT_CONNECTED


def test_dropped_link_is_reported_in_plain_words(
    qtbot: QtBot, panel: GamePanel, context: AppContext
) -> None:
    context.connection.connect("fake", drop_after_ops=1)  # the first tick uses the one op
    panel.clock.pause()
    qtbot.waitUntil(lambda: panel.view.frame is not None)
    assert panel.pump.tick()  # this one drops the link, on the worker
    qtbot.waitUntil(lambda: not context.connection.is_connected)
    qtbot.waitUntil(lambda: not panel.pump.running)
    assert panel.view.frame is None
    assert panel.status_text.startswith("Lost the game: The emulator stopped responding.")
    assert NOT_CONNECTED in panel.status_text


def test_emulator_without_screen_capture(panel: GamePanel, context: AppContext) -> None:
    caps = Capabilities(memory_read=True, input=True, frame_advance=True)
    context.connection.connect("fake", capabilities=caps)
    assert not panel.pump.running
    assert panel.view.frame is None
    assert panel.view.placeholder == NO_SCREEN
    assert panel.status_text == f"Connected to Toy Racer. {NO_SCREEN}"
    context.connection.disconnect()
    assert panel.view.placeholder == NOT_CONNECTED


def test_pump_tick_without_connection_is_a_quiet_noop(context: AppContext) -> None:
    pump = FramePump(context.connection)
    pump.start()
    assert pump.tick() is False
    assert not pump.running
    assert pump.latest is None


def test_pump_counts_other_emulator_errors_and_keeps_running(
    qtbot: QtBot, context: AppContext
) -> None:
    pump = FramePump(context.connection, fps=10)
    context.connection.connect("fake")
    adapter = context.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)

    original = adapter._do_capture_frame
    calls = {"n": 0}

    def flaky() -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmulatorError("glitch")
        return original()

    adapter._do_capture_frame = flaky  # type: ignore[method-assign]
    pump.start()
    pump.stop()  # drive it by hand
    assert pump.tick()
    qtbot.waitUntil(lambda: pump.errors == 1)
    assert pump.latest is None
    assert pump.tick()
    qtbot.waitUntil(lambda: pump.latest is not None)
    assert pump.errors == 1 and pump.captured == 1


class _SlowCapture:
    """Makes the fake adapter's capture wait for a gate, and records its thread."""

    def __init__(self, adapter: FakeEmulatorAdapter) -> None:
        self.gate = threading.Event()
        self.threads: list[int] = []
        original = adapter._do_capture_frame

        def slow() -> object:
            self.threads.append(threading.get_ident())
            self.gate.wait(5)
            return original()

        adapter._do_capture_frame = slow  # type: ignore[method-assign]


def _slow_pump(context: AppContext, fps: int = 100) -> tuple[FramePump, _SlowCapture]:
    pump = FramePump(context.connection, fps=fps)
    context.connection.connect("fake")
    adapter = context.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    return pump, _SlowCapture(adapter)


def test_capture_runs_off_the_gui_thread_and_subscribers_on_it(
    qtbot: QtBot, context: AppContext
) -> None:
    pump, slow = _slow_pump(context)
    slow.gate.set()
    seen: list[int] = []
    pump.subscribe(lambda frame: seen.append(threading.get_ident()))
    assert pump.tick()
    assert pump.in_flight
    qtbot.waitUntil(lambda: pump.latest is not None)
    assert not pump.in_flight
    assert slow.threads and slow.threads[0] != threading.get_ident()
    assert seen == [threading.get_ident()]


def test_tick_skips_while_a_capture_is_in_flight(qtbot: QtBot, context: AppContext) -> None:
    pump, slow = _slow_pump(context)
    assert pump.tick()
    assert pump.tick() is False
    assert pump.tick() is False
    assert pump.skipped == 2 and pump.in_flight
    slow.gate.set()
    qtbot.waitUntil(lambda: pump.captured == 1)
    assert len(slow.threads) == 1  # one capture, not three
    assert pump.tick()  # free again
    qtbot.waitUntil(lambda: pump.captured == 2)


def test_gui_events_keep_flowing_while_a_capture_is_slow(qtbot: QtBot, context: AppContext) -> None:
    """The user's complaint: the window must not wait for the emulator."""
    pump, slow = _slow_pump(context)
    pump.start()
    qtbot.waitUntil(lambda: pump.in_flight)
    fired: list[float] = []
    started = time.perf_counter()
    QTimer.singleShot(0, lambda: fired.append(time.perf_counter() - started))
    qtbot.waitUntil(lambda: bool(fired))
    assert fired[0] < 0.1  # the capture is still blocked behind the gate
    assert pump.in_flight
    slow.gate.set()
    qtbot.waitUntil(lambda: pump.captured >= 1)
    pump.stop()


def test_a_frame_arriving_after_stop_is_dropped(qtbot: QtBot, context: AppContext) -> None:
    pump, slow = _slow_pump(context)
    seen: list[object] = []
    pump.subscribe(seen.append)
    assert pump.tick()
    pump.stop()
    slow.gate.set()
    qtbot.waitUntil(lambda: not pump.in_flight)
    assert pump.latest is None and seen == [] and pump.captured == 0
    assert pump.tick()  # and it still works afterwards
    qtbot.waitUntil(lambda: pump.captured == 1)
    assert len(seen) == 1


def test_pump_rejects_bad_fps(context: AppContext) -> None:
    with pytest.raises(ValueError):
        FramePump(context.connection, fps=0)


def test_main_window_hosts_the_game_panel(qtbot: QtBot, context: AppContext) -> None:
    w = MainWindow(context)
    qtbot.addWidget(w)
    w.show()
    assert w.context is context
    assert w.game_panel.parent() is w.panel("game").content
    context.connection.connect("fake")
    qtbot.waitUntil(lambda: w.game_panel.view.frame is not None)
    assert w.game_panel.clock.running and not w.game_panel.pump.running  # T-823
    w.close()


def test_closing_window_stops_pump_and_unsubscribes(qtbot: QtBot, context: AppContext) -> None:
    w = MainWindow(context)
    qtbot.addWidget(w)
    w.show()
    context.connection.connect("fake")
    before = context.bridge.handler_count()
    w.close()
    w.deleteLater()
    qtbot.waitUntil(lambda: context.bridge.handler_count() < before)
    context.connection.disconnect()  # no handler errors after teardown


def test_pump_feeds_the_observation_service(qtbot: QtBot, context: AppContext) -> None:
    """T-203: every captured frame reaches the screen tracker."""
    panel = GamePanel(context, fps=100)
    qtbot.addWidget(panel)
    context.connection.connect("fake")  # the panel shows a first frame on connect
    panel.clock.pause()
    qtbot.waitUntil(lambda: context.observation.frames_seen >= 1)
    panel.pump.stop()  # from here on, frames only when asked
    qtbot.waitUntil(lambda: not panel.pump.in_flight)
    seen = context.observation.frames_seen
    panel.clock.tick(2)
    assert panel.pump.tick()
    qtbot.waitUntil(lambda: context.observation.frames_seen == seen + 1)
    assert context.observation.screen.transitions == seen
    assert context.observation.screen.observed == seen + 1


# ------------------------------------------- T-823 the clock carries the picture


def test_clock_carries_the_screen_while_playing_and_the_pump_rests(
    qtbot: QtBot, panel: GamePanel, context: AppContext
) -> None:
    context.connection.connect("fake", seed=2)
    assert panel.clock.running and not panel.pump.running
    qtbot.waitUntil(lambda: panel.clock.frames_captured >= 3)
    assert panel.pump.captured == 0  # every picture came with a step
    assert panel.view.frame is not None and panel.status_text.startswith("Watching Toy Racer.")
    qtbot.waitUntil(lambda: context.observation.frames_seen >= 3)  # and reached the tracker
    panel.clock.pause()  # paused: the pump takes over
    assert panel.pump.running
    qtbot.waitUntil(lambda: panel.pump.captured >= 1)
    panel.clock.play()
    assert not panel.pump.running
    context.connection.disconnect()
    assert not panel.clock.running and not panel.pump.running


def test_screenshot_rate_follows_the_clock_to_pump_ratio(qtbot: QtBot, context: AppContext) -> None:
    panel = GamePanel(context, fps=25, clock_fps=100)
    qtbot.addWidget(panel)
    assert panel.clock.screenshot_every == 4 and panel.clock.interval_ms == 10
    panel.clock.pause()
    context.connection.connect("fake")
    panel.pump.stop()
    qtbot.waitUntil(lambda: not panel.pump.in_flight)
    for _ in range(8):
        assert panel.clock.pulse()
        qtbot.waitUntil(lambda: not panel.clock.in_flight)
    assert panel.clock.frames == 8 and panel.clock.frames_captured == 2
    assert panel.view.frame is not None and panel.view.frame.number == 5  # pulses 1 and 5


def test_pad_buttons_ride_along_on_the_clock_step(
    qtbot: QtBot, panel: GamePanel, context: AppContext
) -> None:
    context.connection.connect("fake")
    panel.clock.pause()
    adapter = context.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    qtbot.waitUntil(lambda: not panel.clock.in_flight)
    panel.pad.press("up")
    qtbot.waitUntil(lambda: not panel.pad.sending)
    adapter.set_input([])  # as if the pad's own send had been lost
    assert panel.clock.tick()
    assert adapter.held == panel.pad.held and len(adapter.held) == 1


def test_emulator_without_a_screen_gets_plain_steps(
    qtbot: QtBot, panel: GamePanel, context: AppContext
) -> None:
    context.connection.connect(
        "fake", capabilities=Capabilities(frame_advance=True, memory_read=True, input=True)
    )
    assert panel.clock.running and not panel.pump.running
    qtbot.waitUntil(lambda: panel.clock.frames >= 3)
    assert panel.clock.frames_captured == 0 and panel.clock.errors == 0
    assert panel.view.placeholder == NO_SCREEN
    panel.clock.pause()
    assert not panel.pump.running  # nothing to show, so the pump stays off
