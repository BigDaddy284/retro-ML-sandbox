"""T-010: on-screen pad and keyboard send inputs to the emulator."""

from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtCore import Qt, QTimer
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


def _delivered(qtbot: QtBot, pad: ControllerPad, context: AppContext) -> frozenset[object]:
    """Wait for the pad's held set to reach the emulator (it goes by a worker)."""
    qtbot.waitUntil(lambda: not pad.sending and adapter_of(context).held == pad.held)
    return adapter_of(context).held


def test_press_and_release_send_held_set(
    qtbot: QtBot, pad: ControllerPad, context: AppContext
) -> None:
    context.connection.connect("fake")
    seen: list[frozenset[object]] = []
    pad.input_changed.connect(seen.append)
    assert pad.press("up")
    assert _delivered(qtbot, pad, context) == frozenset({BY_NAME["up"]})
    assert pad.press("a")
    assert _delivered(qtbot, pad, context) == frozenset({BY_NAME["up"], BY_NAME["a"]})
    assert not pad.press("up")  # already held
    assert pad.release("up")
    assert _delivered(qtbot, pad, context) == frozenset({BY_NAME["a"]})
    assert not pad.release("up")
    pad.release_all()
    assert _delivered(qtbot, pad, context) == frozenset()
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
    assert _delivered(qtbot, pad, context) == frozenset({BY_NAME["b"]})
    assert w.isDown()
    qtbot.mouseRelease(w, Qt.MouseButton.LeftButton)
    assert _delivered(qtbot, pad, context) == frozenset()


def test_keyboard_mapping(qtbot: QtBot, pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake")
    pad.setFocus()
    qtbot.keyPress(pad, Qt.Key.Key_Up)
    qtbot.keyPress(pad, Qt.Key.Key_Z)
    assert _delivered(qtbot, pad, context) == frozenset({BY_NAME["up"], BY_NAME["a"]})
    assert pad.widget_for("up").isDown()
    qtbot.keyRelease(pad, Qt.Key.Key_Up)
    assert _delivered(qtbot, pad, context) == frozenset({BY_NAME["a"]})
    qtbot.keyPress(pad, Qt.Key.Key_Return)
    assert BY_NAME["start"] in _delivered(qtbot, pad, context)
    qtbot.keyPress(pad, Qt.Key.Key_F5)  # unmapped: ignored
    assert BY_NAME["start"] in _delivered(qtbot, pad, context)
    assert len(adapter_of(context).held) == 2


def test_auto_repeat_is_ignored(qtbot: QtBot, pad: ControllerPad, context: AppContext) -> None:
    context.connection.connect("fake")
    assert pad.key_down(int(Qt.Key.Key_Up))
    assert not pad.key_down(int(Qt.Key.Key_Up), auto_repeat=True)
    assert not pad.key_up(int(Qt.Key.Key_Up), auto_repeat=True)
    assert BY_NAME["up"] in _delivered(qtbot, pad, context)
    assert pad.key_up(int(Qt.Key.Key_Up))
    assert _delivered(qtbot, pad, context) == frozenset()


def test_custom_keymap(qtbot: QtBot, context: AppContext) -> None:
    pad = ControllerPad(context, keymap={int(Qt.Key.Key_J): "a"})
    qtbot.addWidget(pad)
    context.connection.connect("fake")
    assert pad.key_down(int(Qt.Key.Key_J))
    assert not pad.key_down(int(Qt.Key.Key_Up))
    assert _delivered(qtbot, pad, context) == frozenset({BY_NAME["a"]})


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


def test_dropped_link_while_sending_does_not_crash(
    qtbot: QtBot, pad: ControllerPad, context: AppContext
) -> None:
    context.connection.connect("fake", drop_after_ops=0)
    assert pad.press("up")  # the send drops the link
    qtbot.waitUntil(lambda: pad.message == NOT_CONNECTED)
    assert not context.connection.is_connected
    assert pad.held == frozenset()
    assert pad.send_errors == 0  # a drop is not counted as a glitch


def test_transient_send_error_is_counted(
    qtbot: QtBot, pad: ControllerPad, context: AppContext
) -> None:
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
    qtbot.waitUntil(lambda: pad.send_errors == 1)
    pad.release("up")
    assert _delivered(qtbot, pad, context) == frozenset()
    assert pad.send_errors == 1


def test_pad_input_is_recorded_as_human(pad: ControllerPad, context: AppContext) -> None:
    """T-204: every pad change lands in the input log tagged with the frame."""
    context.connection.connect("fake")
    pad.press("up")
    context.observation.advance(3)
    pad.press("a")
    pad.release("up")
    log = context.observation.inputs
    assert [(e.frame, set(e.buttons), e.source) for e in log] == [
        (0, {"up"}, "human"),
        (3, {"a"}, "human"),
    ]


# ------------------------------------------------------ T-821 off the GUI thread


class _SlowInput:
    """Makes the fake adapter's set_input wait for a gate, and records each call."""

    def __init__(self, adapter: FakeEmulatorAdapter) -> None:
        self.gate = threading.Event()
        self.calls: list[tuple[int, frozenset[object]]] = []
        original = adapter._do_set_input

        def slow(pressed: frozenset[object]) -> None:
            self.calls.append((threading.get_ident(), pressed))
            self.gate.wait(5)
            original(pressed)  # type: ignore[arg-type]

        adapter._do_set_input = slow  # type: ignore[method-assign]


def test_press_returns_at_once_and_sends_on_a_worker(
    qtbot: QtBot, pad: ControllerPad, context: AppContext
) -> None:
    context.connection.connect("fake")
    slow = _SlowInput(adapter_of(context))
    started = time.perf_counter()
    assert pad.press("up")
    assert time.perf_counter() - started < 0.5  # did not wait for the emulator
    assert pad.held == frozenset({BY_NAME["up"]}) and pad.sending
    fired: list[bool] = []
    QTimer.singleShot(0, lambda: fired.append(True))
    qtbot.waitUntil(lambda: bool(fired))  # the window keeps going meanwhile
    slow.gate.set()
    qtbot.waitUntil(lambda: not pad.sending)
    assert len(slow.calls) == 1 and slow.calls[0][0] != threading.get_ident()
    assert adapter_of(context).held == frozenset({BY_NAME["up"]})


def test_changes_during_a_send_are_merged_into_one_more_send(
    qtbot: QtBot, pad: ControllerPad, context: AppContext
) -> None:
    context.connection.connect("fake")
    slow = _SlowInput(adapter_of(context))
    assert pad.press("up")
    assert pad.press("a")  # while "up" is on its way
    assert pad.release("up")
    assert pad.sends == 1 and pad.held == frozenset({BY_NAME["a"]})
    slow.gate.set()
    qtbot.waitUntil(lambda: pad.sends == 2 and not pad.sending)
    assert [pressed for _, pressed in slow.calls] == [
        frozenset({BY_NAME["up"]}),
        frozenset({BY_NAME["a"]}),
    ]
    assert adapter_of(context).held == frozenset({BY_NAME["a"]})


def test_a_merged_send_still_goes_out_after_a_glitch(
    qtbot: QtBot, pad: ControllerPad, context: AppContext
) -> None:
    from retro_ai_lab.core import EmulatorError

    context.connection.connect("fake")
    adapter = adapter_of(context)
    gate = threading.Event()
    original = adapter._do_set_input
    calls: list[frozenset[object]] = []

    def flaky(pressed: frozenset[object]) -> None:
        calls.append(pressed)
        if len(calls) == 1:
            gate.wait(5)
            raise EmulatorError("glitch")
        original(pressed)  # type: ignore[arg-type]

    adapter._do_set_input = flaky  # type: ignore[method-assign]
    assert pad.press("up")
    assert pad.press("a")  # merged while the first send is failing
    gate.set()
    qtbot.waitUntil(lambda: pad.send_errors == 1 and not pad.sending)
    assert calls == [frozenset({BY_NAME["up"]}), frozenset({BY_NAME["up"], BY_NAME["a"]})]
    assert adapter.held == frozenset({BY_NAME["up"], BY_NAME["a"]})
