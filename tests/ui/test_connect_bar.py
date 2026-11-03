"""T-009: connect / disconnect UI with capability display."""

from __future__ import annotations

import time
from collections.abc import Sequence

import pytest
from PySide6.QtCore import Qt, QTimer
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import Button, Capabilities, EmulatorError, GameInfo
from retro_ai_lab.emulator.base import EmulatorAdapter
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.registry import AdapterRegistry
from retro_ai_lab.services import ConnectionState
from retro_ai_lab.ui.connect_bar import (
    CAPABILITY_LABELS,
    STATUS_CONNECTING,
    STATUS_IDLE,
    ConnectBar,
    describe_capabilities,
)
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.main_window import MainWindow


class SlowAdapter(EmulatorAdapter):
    """Blocks for a while in connect, to prove the GUI thread stays free."""

    name = "slow"
    display_name = "Slow console"

    def __init__(self, delay_s: float = 0.3) -> None:
        super().__init__()
        self.delay_s = delay_s

    def _do_connect(self, timeout_s: float) -> None:
        time.sleep(self.delay_s)

    def _do_disconnect(self) -> None: ...

    def _do_capabilities(self) -> Capabilities:
        return Capabilities(input=True)

    def _do_game_info(self) -> GameInfo:
        return GameInfo("SLOW", "Slow Game")

    def _do_buttons(self) -> Sequence[Button]:
        return [Button("a", "A")]


@pytest.fixture
def context() -> AppContext:
    registry = AdapterRegistry()
    registry.register(FakeEmulatorAdapter)
    registry.register(SlowAdapter)
    return AppContext(registry=registry)


@pytest.fixture
def bar(qtbot: QtBot, context: AppContext) -> ConnectBar:
    b = ConnectBar(context)
    qtbot.addWidget(b)
    b.show()
    return b


# ------------------------------------------------------- describe_capabilities


def test_describe_all_and_none() -> None:
    assert describe_capabilities(Capabilities.all()) == (
        "Can: see the screen, look inside the game, press buttons, "
        "step one frame at a time, change speed, save and rewind."
    )
    assert describe_capabilities(Capabilities()) == (
        "Can't: see the screen, look inside the game, press buttons, "
        "step one frame at a time, change speed, save and rewind."
    )


def test_describe_mixed() -> None:
    text = describe_capabilities(Capabilities(screen_capture=True, input=True))
    assert text == (
        "Can: see the screen, press buttons. "
        "Can't: look inside the game, step one frame at a time, change speed, save and rewind."
    )
    for label in CAPABILITY_LABELS.values():
        assert "_" not in label and label.islower()


# ---------------------------------------------------------------- initial


def test_initial_state(bar: ConnectBar, context: AppContext) -> None:
    assert bar.status_text == STATUS_IDLE
    assert bar.capability_text == ""
    assert bar.selected_adapter == "fake"
    assert bar.connect_button.isEnabled()
    assert not bar.disconnect_button.isEnabled()
    assert not bar.busy
    bar.select_adapter("slow")
    assert bar.selected_adapter == "slow"
    with pytest.raises(ValueError):
        bar.select_adapter("nope")


# ------------------------------------------------------- connect by click


def test_click_connect_then_disconnect(qtbot: QtBot, bar: ConnectBar, context: AppContext) -> None:
    qtbot.mouseClick(bar.connect_button, Qt.MouseButton.LeftButton)
    assert bar.busy
    assert not bar.connect_button.isEnabled()
    qtbot.waitUntil(lambda: context.connection.is_connected and not bar.busy)
    assert bar.status_text == "Connected to Toy Racer."
    assert bar.capability_text == describe_capabilities(Capabilities.all())
    assert not bar.connect_button.isEnabled()
    assert bar.disconnect_button.isEnabled()
    assert bar.last_error is None

    qtbot.mouseClick(bar.disconnect_button, Qt.MouseButton.LeftButton)
    assert context.connection.state is ConnectionState.DISCONNECTED
    assert bar.status_text == STATUS_IDLE
    assert bar.capability_text == ""
    assert bar.connect_button.isEnabled()
    assert not bar.disconnect_button.isEnabled()


def test_connect_reports_failure_in_plain_words(
    qtbot: QtBot, bar: ConnectBar, context: AppContext
) -> None:
    assert bar.connect_now("fake", fail_connect=EmulatorError("Nothing is listening."))
    qtbot.waitUntil(lambda: not bar.busy)
    assert context.connection.state is ConnectionState.DISCONNECTED
    assert bar.status_text == "Couldn't connect: Nothing is listening."
    assert bar.last_error == "Nothing is listening."
    assert bar.connect_button.isEnabled()
    # And we can try again straight away.
    assert bar.connect_now("fake")
    qtbot.waitUntil(lambda: context.connection.is_connected)


def test_gui_stays_responsive_while_connecting(
    qtbot: QtBot, bar: ConnectBar, context: AppContext
) -> None:
    ticks: list[int] = []
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    assert bar.connect_now("slow", delay_s=0.4)
    assert bar.status_text == STATUS_CONNECTING
    assert not bar.connect_now("slow")  # already busy
    qtbot.waitUntil(lambda: context.connection.is_connected and not bar.busy, timeout=5000)
    timer.stop()
    assert len(ticks) >= 5  # the event loop kept turning during the 0.4 s connect
    assert bar.status_text == "Connected to Slow Game."
    assert bar.capability_text.startswith("Can: press buttons.")


def test_connect_rejected_when_already_connected(
    qtbot: QtBot, bar: ConnectBar, context: AppContext
) -> None:
    context.connection.connect("fake")
    assert not bar.connect_now("fake")
    assert not bar.connect_button.isEnabled()


def test_no_adapters_available(qtbot: QtBot) -> None:
    ctx = AppContext(registry=AdapterRegistry())
    b = ConnectBar(ctx)
    qtbot.addWidget(b)
    assert b.selected_adapter is None
    assert not b.connect_now()
    assert "no console" in b.status_text


def test_unexpected_loss_is_shown(qtbot: QtBot, bar: ConnectBar, context: AppContext) -> None:
    context.connection.connect("fake", drop_after_ops=0)
    with pytest.raises(Exception, match="stopped responding"):
        context.connection.call(lambda a: a.capture_frame())
    assert bar.status_text == (
        "Lost the connection: The emulator stopped responding. Press Reconnect to try again."
    )
    assert bar.offering_reconnect
    assert bar.capability_text == ""
    assert bar.connect_button.isEnabled()


# ------------------------------------------------------------ main window


def test_main_window_connect_bar_drives_game_panel(qtbot: QtBot, context: AppContext) -> None:
    w = MainWindow(context)
    qtbot.addWidget(w)
    w.show()
    assert w.connect_bar.parent() is not None
    qtbot.mouseClick(w.connect_bar.connect_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: w.game_panel.view.frame is not None)
    assert w.game_panel.pump.running
    qtbot.mouseClick(w.connect_bar.disconnect_button, Qt.MouseButton.LeftButton)
    assert w.game_panel.view.frame is None
    assert not w.game_panel.pump.running
    w.close()


def test_bizhawk_is_offered_and_a_missing_emulahawk_is_explained(qtbot: QtBot) -> None:
    import re
    import socket

    from retro_ai_lab.emulator.bizhawk import SCRIPT_PATH
    from retro_ai_lab.emulator.registry import default_registry

    ctx = AppContext(registry=default_registry())
    b = ConnectBar(ctx)
    qtbot.addWidget(b)
    assert [b._picker.itemText(i) for i in range(b._picker.count())] == [
        "Practice console (Toy Racer)",
        "BizHawk (EmuHawk)",
        "mGBA",
    ]
    b.select_adapter("bizhawk")
    with socket.socket() as probe:  # a port nobody is listening on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert b.connect_now(port=port, timeout_s=1.0)
    qtbot.waitUntil(lambda: not b.busy, timeout=5000)
    assert b.status_text.startswith("Couldn't connect: Couldn't find BizHawk on port")
    assert re.search(re.escape(str(SCRIPT_PATH)), b.status_text)
    assert b.last_error is not None
    assert ctx.connection.state is ConnectionState.DISCONNECTED
