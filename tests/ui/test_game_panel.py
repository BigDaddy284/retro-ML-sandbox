"""T-008: the Game panel shows live frames while connected."""

from __future__ import annotations

import pytest
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


def test_connect_shows_first_frame_and_starts_pump(panel: GamePanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    assert panel.pump.running
    assert panel.pump.interval_ms == 1000 // 60
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


def test_dropped_link_is_reported_in_plain_words(panel: GamePanel, context: AppContext) -> None:
    context.connection.connect("fake", drop_after_ops=1)  # the first tick uses the one op
    assert panel.view.frame is not None
    panel.pump.tick()  # this one drops the link
    assert not panel.pump.running
    assert not context.connection.is_connected
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
    assert pump.tick() is None
    assert not pump.running
    assert pump.latest is None


def test_pump_counts_other_emulator_errors_and_keeps_running(context: AppContext) -> None:
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
    assert pump.tick() is None
    assert pump.errors == 1
    assert pump.running
    assert pump.tick() is not None
    assert pump.errors == 1


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
    assert w.game_panel.view.frame is not None
    assert w.game_panel.pump.running
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
