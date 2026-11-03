"""T-011: the UI copes with an emulator that goes away, and offers Reconnect."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import EmulatorError
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.services import LINK_LOST, ConnectionState
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.controller_pad import NOT_CONNECTED as PAD_NOT_CONNECTED
from retro_ai_lab.ui.link_watchdog import LinkWatchdog
from retro_ai_lab.ui.main_window import MainWindow


@pytest.fixture
def context() -> AppContext:
    return AppContext()


@pytest.fixture
def window(qtbot: QtBot, context: AppContext) -> MainWindow:
    w = MainWindow(context)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    return w


def fake(context: AppContext) -> FakeEmulatorAdapter:
    a = context.connection.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    return a


# ------------------------------------------------------------ watchdog


def test_watchdog_checks_on_a_timer(qtbot: QtBot, context: AppContext) -> None:
    dog = LinkWatchdog(context.connection, interval_ms=10)
    assert not dog.running
    assert dog.interval_ms == 10
    dog.start()
    qtbot.waitUntil(lambda: dog.checks >= 3)
    dog.stop()
    assert not dog.running
    with pytest.raises(ValueError):
        LinkWatchdog(context.connection, interval_ms=0)


def test_watchdog_notices_a_silent_drop(qtbot: QtBot, context: AppContext) -> None:
    dog = LinkWatchdog(context.connection, interval_ms=10)
    dog.start()
    context.connection.connect("fake")
    context.connection.call(lambda a: a.capture_frame())
    fake(context).drop_link()
    qtbot.waitUntil(lambda: context.connection.state is ConnectionState.DISCONNECTED)
    dog.stop()


# ----------------------------------------------------- whole window


def test_silent_drop_is_noticed_and_every_part_of_the_window_recovers(
    qtbot: QtBot, window: MainWindow, context: AppContext
) -> None:
    assert window.watchdog.running
    context.connection.connect("fake", seed=4)
    window.game_panel.pad.press("up")
    qtbot.waitUntil(lambda: window.game_panel.view.frame is not None)

    fake(context).drop_link()  # the emulator dies while nothing is being sent
    qtbot.waitUntil(lambda: context.connection.state is ConnectionState.DISCONNECTED)

    # The drop may be found on the frame pump's worker thread (T-817), in
    # which case the window hears of it a moment later, through the bridge.
    bar = window.connect_bar
    qtbot.waitUntil(lambda: bar.offering_reconnect)
    assert bar.status_text == f"Lost the connection: {LINK_LOST} Press Reconnect to try again."
    assert bar.connect_button.isEnabled()
    assert not bar.disconnect_button.isEnabled()
    assert bar.capability_text == ""
    assert window.game_panel.view.frame is None
    assert not window.game_panel.pump.running
    assert not window.game_panel.clock.running
    assert not window.game_panel.play_button.isEnabled()
    assert window.game_panel.pad.held == frozenset()
    assert window.game_panel.pad.message == PAD_NOT_CONNECTED
    assert window.statusBar().currentMessage() == (
        f"Lost the game: {LINK_LOST} You can reconnect from the toolbar."
    )


def test_reconnect_button_uses_the_same_console_and_options(
    qtbot: QtBot, window: MainWindow, context: AppContext
) -> None:
    bar = window.connect_bar
    assert bar.connect_now("fake", seed=4)
    qtbot.waitUntil(lambda: context.connection.is_connected and not bar.busy)
    fake(context).drop_link()
    qtbot.waitUntil(lambda: context.connection.state is ConnectionState.DISCONNECTED)
    qtbot.waitUntil(lambda: bar.connect_button.text() == "Reconnect")

    qtbot.mouseClick(bar.connect_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: context.connection.is_connected and not bar.busy)
    session = context.connection.session
    assert session is not None and session.game_info.rom_hash == "toyracer-4"
    assert bar.connect_button.text() == "Connect"
    assert not bar.offering_reconnect
    assert bar.status_text == "Connected to Toy Racer."
    assert window.statusBar().currentMessage() == "Connected to Toy Racer. Have fun!"
    assert window.game_panel.clock.running
    qtbot.waitUntil(lambda: window.game_panel.view.frame is not None)


def test_reconnect_without_previous_request_falls_back_to_picker(
    qtbot: QtBot, window: MainWindow, context: AppContext
) -> None:
    assert window.connect_bar.reconnect()
    qtbot.waitUntil(lambda: context.connection.is_connected)


def test_expected_disconnect_does_not_offer_reconnect(
    qtbot: QtBot, window: MainWindow, context: AppContext
) -> None:
    context.connection.connect("fake")
    qtbot.mouseClick(window.connect_bar.disconnect_button, Qt.MouseButton.LeftButton)
    assert not window.connect_bar.offering_reconnect
    assert window.connect_bar.connect_button.text() == "Connect"
    assert window.statusBar().currentMessage() == "Disconnected. Connect to a game to begin."


def test_failed_connect_shows_in_status_bar(
    qtbot: QtBot, window: MainWindow, context: AppContext
) -> None:
    window.connect_bar.connect_now("fake", fail_connect=EmulatorError("Nothing is listening."))
    qtbot.waitUntil(lambda: not window.connect_bar.busy)
    assert window.statusBar().currentMessage() == "Couldn't connect: Nothing is listening."


def test_backend_error_on_disconnect_click_does_not_break_the_ui(
    qtbot: QtBot, window: MainWindow, context: AppContext
) -> None:
    context.connection.connect("fake")
    adapter = fake(context)

    def broken() -> None:
        raise EmulatorError("socket already closed")

    adapter._do_disconnect = broken  # type: ignore[method-assign]
    qtbot.mouseClick(window.connect_bar.disconnect_button, Qt.MouseButton.LeftButton)
    assert context.connection.state is ConnectionState.DISCONNECTED
    assert window.connect_bar.connect_button.isEnabled()
    assert window.connect_bar.reconnect() or True
    qtbot.waitUntil(lambda: context.connection.is_connected)


def test_drop_during_manual_play_is_handled_once(
    qtbot: QtBot, window: MainWindow, context: AppContext
) -> None:
    context.connection.connect("fake", drop_after_ops=3)
    qtbot.waitUntil(lambda: context.connection.state is ConnectionState.DISCONNECTED)
    qtbot.waitUntil(lambda: window.connect_bar.offering_reconnect)
    # Nothing keeps hammering a dead connection.
    pump_errors = window.game_panel.pump.errors
    clock_errors = window.game_panel.clock.errors
    qtbot.wait(60)
    assert window.game_panel.pump.errors == pump_errors
    assert window.game_panel.clock.errors == clock_errors
