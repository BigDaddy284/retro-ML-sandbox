"""On-screen controller and keyboard mapping for manual play (DESIGN.md §8).

The pad is built from whatever buttons the connected emulator reports, so it
works for any console. Held buttons are sent to the adapter whenever the set
changes; the pad also emits :attr:`input_changed` so later tasks (input
recording, demonstrations) can observe the child's play.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from retro_ai_lab.core import Button, EmulatorDisconnectedError
from retro_ai_lab.services import EmulatorConnected, EmulatorDisconnected
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.worker import report

# Keyboard keys for the button names most consoles use. Unlisted buttons are
# only reachable from the on-screen pad.
DEFAULT_KEYMAP: dict[int, str] = {
    int(Qt.Key.Key_Up): "up",
    int(Qt.Key.Key_Down): "down",
    int(Qt.Key.Key_Left): "left",
    int(Qt.Key.Key_Right): "right",
    int(Qt.Key.Key_Z): "a",
    int(Qt.Key.Key_X): "b",
    int(Qt.Key.Key_A): "x",
    int(Qt.Key.Key_S): "y",
    int(Qt.Key.Key_Q): "l",
    int(Qt.Key.Key_W): "r",
    int(Qt.Key.Key_Return): "start",
    int(Qt.Key.Key_Enter): "start",
    int(Qt.Key.Key_Space): "select",
}

NO_INPUT = "This emulator can't take button presses from here."
NOT_CONNECTED = "Connect to a game to play it yourself."
KEY_HINT = "Keys: arrows, Z = A, X = B, Enter = Start."
AI_HAS_CONTROLS = "The AI has the controls while it plays."

# Rough layout positions (row, col) for well-known buttons; others are appended.
_LAYOUT: dict[str, tuple[int, int]] = {
    "up": (0, 1),
    "left": (1, 0),
    "down": (2, 1),
    "right": (1, 2),
    "b": (1, 4),
    "a": (1, 5),
    "y": (0, 4),
    "x": (0, 5),
    "l": (0, 0),
    "r": (0, 6),
    "select": (3, 3),
    "start": (3, 4),
}


class _SendSignals(QObject):
    done = Signal(object)  # None, or the exception


class _SendTask(QRunnable):
    """One ``set_input`` call, off the GUI thread (T-821)."""

    def __init__(self, context: AppContext, held: frozenset[Button]) -> None:
        super().__init__()
        self.signals = _SendSignals()
        self._context = context
        self._held = held

    def run(self) -> None:
        held = self._held
        try:
            self._context.connection.call(lambda a: a.set_input(held))
        except Exception as exc:
            report(self.signals.done, exc)
        else:
            report(self.signals.done, None)


class ControllerPad(QWidget):
    """On-screen buttons and keyboard mapping for the connected game.

    The held set is the pad's own state and changes at once; the request to
    the emulator goes out on a worker thread, because a real emulator takes a
    frame or more to answer and the window must not wait for it. Changes made
    while a request is in flight are merged: the final held set goes out once
    the worker is free.
    """

    input_changed = Signal(object)  # frozenset[Button]

    def __init__(
        self,
        context: AppContext,
        parent: QWidget | None = None,
        keymap: dict[int, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("controller_pad")
        self._context = context
        self._keymap = dict(DEFAULT_KEYMAP if keymap is None else keymap)
        self._buttons: dict[str, Button] = {}
        self._widgets: dict[str, QPushButton] = {}
        self._held: set[str] = set()
        self._enabled = False
        self._locked = False
        self._send_errors = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._sending = False
        self._pending: frozenset[Button] | None = None
        self._sends = 0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._grid_host = QWidget(self)
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._message = QLabel(NOT_CONNECTED, self)
        self._message.setObjectName("pad_message")
        self._message.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._grid_host)
        layout.addWidget(self._message)

        unsubscribe = (
            context.bridge.subscribe(EmulatorConnected, self._on_connected),
            context.bridge.subscribe(EmulatorDisconnected, self._on_disconnected),
        )
        self.destroyed.connect(lambda *_: [off() for off in unsubscribe])

    # --------------------------------------------------------------- API

    @property
    def held(self) -> frozenset[Button]:
        return frozenset(self._buttons[name] for name in self._held)

    @property
    def button_names(self) -> tuple[str, ...]:
        return tuple(self._buttons)

    @property
    def message(self) -> str:
        return self._message.text()

    @property
    def send_errors(self) -> int:
        return self._send_errors

    @property
    def sending(self) -> bool:
        """Is a request on its way to the emulator right now?"""
        return self._sending

    @property
    def sends(self) -> int:
        """Requests handed to the worker since the pad was made."""
        return self._sends

    @property
    def locked(self) -> bool:
        """True while the AI is playing and the child's presses are refused."""
        return self._locked

    def set_locked(self, locked: bool) -> None:
        if locked == self._locked:
            return
        self._locked = locked
        if locked:
            self.release_all()
        self._grid_host.setEnabled(self._enabled and not locked)
        if locked:
            self._message.setText(AI_HAS_CONTROLS)
        else:
            self._message.setText(
                KEY_HINT if self._enabled else (NO_INPUT if self._buttons else NOT_CONNECTED)
            )

    def widget_for(self, name: str) -> QPushButton:
        return self._widgets[name]

    def press(self, name: str) -> bool:
        """Hold button ``name``. Returns False if the pad is inactive or unknown."""
        if self._locked or not self._enabled or name not in self._buttons or name in self._held:
            return False
        self._held.add(name)
        self._widgets[name].setDown(True)
        self._send()
        return True

    def release(self, name: str) -> bool:
        if name not in self._held:
            return False
        self._held.discard(name)
        if name in self._widgets:
            self._widgets[name].setDown(False)
        self._send()
        return True

    def release_all(self) -> None:
        if not self._held:
            return
        for name in list(self._held):
            self._held.discard(name)
            if name in self._widgets:
                self._widgets[name].setDown(False)
        self._send()

    def key_down(self, key: int, auto_repeat: bool = False) -> bool:
        if auto_repeat:
            return False
        name = self._keymap.get(key)
        return self.press(name) if name is not None else False

    def key_up(self, key: int, auto_repeat: bool = False) -> bool:
        if auto_repeat:
            return False
        name = self._keymap.get(key)
        return self.release(name) if name is not None else False

    # ----------------------------------------------------------- Qt hooks

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if not self.key_down(event.key(), event.isAutoRepeat()):
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if not self.key_up(event.key(), event.isAutoRepeat()):
            super().keyReleaseEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        """Lets a parent window forward its key events here."""
        if isinstance(event, QKeyEvent):
            if event.type() == QEvent.Type.KeyPress:
                return self.key_down(event.key(), event.isAutoRepeat())
            if event.type() == QEvent.Type.KeyRelease:
                return self.key_up(event.key(), event.isAutoRepeat())
        return False

    # ------------------------------------------------------------ events

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._clear_buttons()
        self._buttons = {b.name: b for b in event.buttons}
        for index, button in enumerate(event.buttons):
            widget = QPushButton(button.label, self._grid_host)
            widget.setObjectName(f"pad_{button.name}")
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            widget.pressed.connect(lambda n=button.name: self.press(n))
            widget.released.connect(lambda n=button.name: self.release(n))
            row, col = _LAYOUT.get(button.name, (4 + index // 8, index % 8))
            self._grid.addWidget(widget, row, col)
            self._widgets[button.name] = widget
        self._enabled = event.capabilities.input
        self._locked = False
        self._grid_host.setEnabled(self._enabled)
        self._message.setText(KEY_HINT if self._enabled else NO_INPUT)

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._held.clear()
        self._enabled = False
        self._locked = False
        self._clear_buttons()
        self._message.setText(NOT_CONNECTED)
        self.input_changed.emit(frozenset())

    def _clear_buttons(self) -> None:
        for widget in self._widgets.values():
            self._grid.removeWidget(widget)
            widget.deleteLater()
        self._widgets.clear()
        self._buttons.clear()

    def _send(self) -> None:
        held = self.held
        self.input_changed.emit(held)
        if not self._context.connection.is_connected:
            return
        self._context.observation.record_input((b.name for b in held), "human")
        self._queue(held)

    def _queue(self, held: frozenset[Button]) -> None:
        if self._sending:
            self._pending = held  # the newest set wins; it goes out when the worker is free
            return
        self._sending = True
        self._sends += 1
        task = _SendTask(self._context, held)
        task.signals.done.connect(self._on_sent)
        self._pool.start(task)

    def _on_sent(self, error: object) -> None:
        """Back on the GUI thread once the emulator has answered."""
        self._sending = False
        if isinstance(error, EmulatorDisconnectedError):
            self._pending = None  # the service announces the drop; the pad resets on that event
            return
        if isinstance(error, Exception):
            # Counted and carried on, so one glitch can't stick a key.
            self._send_errors += 1
        pending, self._pending = self._pending, None
        if pending is not None and self._context.connection.is_connected:
            self._queue(pending)


__all__ = ["DEFAULT_KEYMAP", "KEY_HINT", "NOT_CONNECTED", "NO_INPUT", "ControllerPad"]
