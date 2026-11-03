"""T-010: on-screen pad and keyboard send inputs to the emulator."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import Capabilities
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.controller_pad import KEY_HINT, NO_INPUT, NOT_CONNECTED, ControllerPad

BY_NAME = {b.name: b for b in BUTTONS}


@pytest.fixture
def context() -> AppContext:
    return AppContext()


@pytest.fixture
def pad(qtbot: QtBot, context: AppContext) -> ControllerPad:
    p = ControllerPad(context)
    qtbot.addWidget(p)
    p.show()
    return p


def adapter_of(context: AppContext) -> FakeEmulatorAdapter:
    a = context.connection.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    return a


def test_starts_empty(pad: ControllerPad) -> None:
    assert pad.button_names == ()
    assert pad.held == frozenset()
    assert pad.message == NOT_CONNECTED
    assert not pad.press("a")
    assert not pad.key_down(int(Qt.Key.Key_Up))


def test_builds_buttons_from_the_emulator(pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake")
    assert pad.button_names == tuple(b.name for b in BUTTONS)
    assert pad.message == KEY_HINT
    for b in BUTTONS:
        assert pad.widget_for(b.name).text() == b.label
    context.connection.disconnect()
    assert pad.button_names == ()
    assert pad.message == NOT_CONNECTED


def test_press_and_release_send_held_set(pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake")
    seen: list[frozenset[object]] = []
    pad.input_changed.connect(seen.append)
    assert pad.press("up")
    assert adapter_of(context).held == frozenset({BY_NAME["up"]})
    assert pad.press("a")
    assert adapter_of(context).held == frozenset({BY_NAME["up"], BY_NAME["a"]})
    assert not pad.press("up")  # already held
    assert pad.release("up")
    assert adapter_of(context).held == frozenset({BY_NAME["a"]})
    assert not pad.release("up")
    pad.release_all()
    assert adapter_of(context).held == frozenset()
    pad.release_all()  # nothing held: no extra signal
    assert seen == [
        frozenset({BY_NAME["up"]}),
        frozenset({BY_NAME["up"], BY_NAME["a"]}),
        frozenset({BY_NAME["a"]}),
        frozenset(),
    ]


def test_unknown_button_is_ignored(pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake")
    assert not pad.press("turbo")
    assert adapter_of(context).held == frozenset()


def test_on_screen_buttons_by_mouse(qtbot: QtBot, pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake")
    w = pad.widget_for("b")
    qtbot.mousePress(w, Qt.MouseButton.LeftButton)
    assert adapter_of(context).held == frozenset({BY_NAME["b"]})
    assert w.isDown()
    qtbot.mouseRelease(w, Qt.MouseButton.LeftButton)
    assert adapter_of(context).held == frozenset()


def test_keyboard_mapping(qtbot: QtBot, pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake")
    pad.setFocus()
    qtbot.keyPress(pad, Qt.Key.Key_Up)
    qtbot.keyPress(pad, Qt.Key.Key_Z)
    assert adapter_of(context).held == frozenset({BY_NAME["up"], BY_NAME["a"]})
    assert pad.widget_for("up").isDown()
    qtbot.keyRelease(pad, Qt.Key.Key_Up)
    assert adapter_of(context).held == frozenset({BY_NAME["a"]})
    qtbot.keyPress(pad, Qt.Key.Key_Return)
    assert BY_NAME["start"] in adapter_of(context).held
    qtbot.keyPress(pad, Qt.Key.Key_F5)  # unmapped: ignored
    assert BY_NAME["start"] in adapter_of(context).held
    assert len(adapter_of(context).held) == 2


def test_auto_repeat_is_ignored(pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake")
    assert pad.key_down(int(Qt.Key.Key_Up))
    assert not pad.key_down(int(Qt.Key.Key_Up), auto_repeat=True)
    assert not pad.key_up(int(Qt.Key.Key_Up), auto_repeat=True)
    assert BY_NAME["up"] in adapter_of(context).held
    assert pad.key_up(int(Qt.Key.Key_Up))
    assert adapter_of(context).held == frozenset()


def test_custom_keymap(qtbot: QtBot, context: AppContext) -> None:
    pad = ControllerPad(context, keymap={int(Qt.Key.Key_J): "a"})
    qtbot.addWidget(pad)
    context.connection.connect("fake")
    assert pad.key_down(int(Qt.Key.Key_J))
    assert not pad.key_down(int(Qt.Key.Key_Up))
    assert adapter_of(context).held == frozenset({BY_NAME["a"]})


def test_emulator_without_input_disables_pad(pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake", capabilities=Capabilities(screen_capture=True))
    assert pad.button_names == tuple(b.name for b in BUTTONS)
    assert pad.message == NO_INPUT
    assert not pad.press("a")
    assert pad.held == frozenset()


def test_disconnect_releases_everything(pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake")
    pad.press("up")
    seen: list[frozenset[object]] = []
    pad.input_changed.connect(seen.append)
    context.connection.disconnect()
    assert pad.held == frozenset()
    assert seen == [frozenset()]


def test_dropped_link_while_sending_does_not_crash(pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake", drop_after_ops=0)
    assert pad.press("up")  # the send drops the link
    assert not context.connection.is_connected
    assert pad.message == NOT_CONNECTED
    assert pad.held == frozenset()
    assert pad.send_errors == 0  # a drop is not counted as a glitch


def test_transient_send_error_is_counted(pad: ControllerPad, context: AppContext) -> None:
    from retro_ai_lab.core import EmulatorError

    context.connection.connect("fake")
    adapter = adapter_of(context)
    calls = {"n": 0}
    original = adapter._do_set_input

    def flaky(pressed: frozenset[object]) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmulatorError("glitch")
        original(pressed)  # type: ignore[arg-type]

    adapter._do_set_input = flaky  # type: ignore[method-assign]
    pad.press("up")
    assert pad.send_errors == 1
    pad.release("up")
    assert adapter.held == frozenset()
    assert pad.send_errors == 1
