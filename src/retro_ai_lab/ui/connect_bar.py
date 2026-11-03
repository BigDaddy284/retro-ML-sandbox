"""Connect / disconnect controls with a plain-English capability display."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from retro_ai_lab.core import Capabilities
from retro_ai_lab.services import (
    ConnectionFailed,
    ConnectionState,
    ConnectionStateChanged,
    EmulatorConnected,
    EmulatorDisconnected,
)
from retro_ai_lab.ui.context import AppContext

CAPABILITY_LABELS: dict[str, str] = {
    "screen_capture": "see the screen",
    "memory_read": "look inside the game",
    "input": "press buttons",
    "frame_advance": "step one frame at a time",
    "speed_control": "change speed",
    "save_states": "save and rewind",
}
assert set(CAPABILITY_LABELS) == set(Capabilities.names())

STATUS_IDLE = "Pick a console and press Connect."
STATUS_CONNECTING = "Connecting…"


class _ConnectSignals(QObject):
    done = Signal(object)  # the exception, or None on success


class _ConnectTask(QRunnable):
    """Runs ``ConnectionService.connect`` off the GUI thread."""

    def __init__(self, context: AppContext, name: str, options: dict[str, Any]) -> None:
        super().__init__()
        self.signals = _ConnectSignals()
        self._context = context
        self._name = name
        self._options = options

    def run(self) -> None:
        try:
            self._context.connection.connect(self._name, **self._options)
        except Exception as exc:
            self.signals.done.emit(exc)
        else:
            self.signals.done.emit(None)


class ConnectBar(QWidget):
    """Adapter picker, Connect/Disconnect buttons, status and capabilities."""

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("connect_bar")
        self._context = context
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._busy = False
        self._last_error: str | None = None

        self._picker = QComboBox(self)
        self._picker.setObjectName("adapter_picker")
        for name, display in context.connection.available_adapters():
            self._picker.addItem(display, name)

        self._connect = QPushButton("Connect", self)
        self._connect.setObjectName("connect_button")
        self._connect.clicked.connect(self._on_connect_clicked)
        self._disconnect = QPushButton("Disconnect", self)
        self._disconnect.setObjectName("disconnect_button")
        self._disconnect.clicked.connect(self.disconnect_now)

        self._status = QLabel(STATUS_IDLE, self)
        self._status.setObjectName("connect_status")
        self._caps = QLabel("", self)
        self._caps.setObjectName("capability_strip")
        self._caps.setWordWrap(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(QLabel("Console:", self))
        layout.addWidget(self._picker)
        layout.addWidget(self._connect)
        layout.addWidget(self._disconnect)
        layout.addWidget(self._status, 1)
        layout.addWidget(self._caps, 2)

        bridge = context.bridge
        unsubscribe = (
            bridge.subscribe(ConnectionStateChanged, self._on_state),
            bridge.subscribe(EmulatorConnected, self._on_connected),
            bridge.subscribe(EmulatorDisconnected, self._on_disconnected),
            bridge.subscribe(ConnectionFailed, self._on_failed),
        )
        self.destroyed.connect(lambda *_: [off() for off in unsubscribe])
        self._apply_state(context.connection.state)

    # --------------------------------------------------------------- API

    @property
    def selected_adapter(self) -> str | None:
        data = self._picker.currentData()
        return str(data) if data is not None else None

    def select_adapter(self, name: str) -> None:
        index = self._picker.findData(name)
        if index < 0:
            raise ValueError(f"No adapter {name!r} in the picker.")
        self._picker.setCurrentIndex(index)

    @property
    def status_text(self) -> str:
        return self._status.text()

    @property
    def capability_text(self) -> str:
        return self._caps.text()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def connect_button(self) -> QPushButton:
        return self._connect

    @property
    def disconnect_button(self) -> QPushButton:
        return self._disconnect

    def connect_now(self, name: str | None = None, **options: Any) -> bool:
        """Start connecting in the background. Returns False if not possible now."""
        if self._busy or self._context.connection.state is not ConnectionState.DISCONNECTED:
            return False
        adapter = name or self.selected_adapter
        if adapter is None:
            self._status.setText("There is no console to connect to.")
            return False
        self._busy = True
        self._last_error = None
        self._status.setText(STATUS_CONNECTING)
        self._update_buttons()
        task = _ConnectTask(self._context, adapter, options)
        task.signals.done.connect(self._on_task_done)
        self._pool.start(task)
        return True

    @Slot()
    def disconnect_now(self) -> None:
        self._context.connection.disconnect()

    # ------------------------------------------------------------ internal

    def _on_connect_clicked(self) -> None:
        self.connect_now()

    def _on_task_done(self, error: object) -> None:
        self._busy = False
        if isinstance(error, Exception):
            self._last_error = str(error) or error.__class__.__name__
        self._update_buttons()

    def _on_state(self, event: ConnectionStateChanged) -> None:
        self._apply_state(event.new)

    def _apply_state(self, state: ConnectionState) -> None:
        if state is ConnectionState.CONNECTING:
            self._status.setText(STATUS_CONNECTING)
        elif state is ConnectionState.DISCONNECTED and self._status.text() == STATUS_CONNECTING:
            self._status.setText(STATUS_IDLE)
        self._update_buttons()

    def _update_buttons(self) -> None:
        state = self._context.connection.state
        self._connect.setEnabled(not self._busy and state is ConnectionState.DISCONNECTED)
        self._disconnect.setEnabled(state is ConnectionState.CONNECTED)
        self._picker.setEnabled(not self._busy and state is ConnectionState.DISCONNECTED)

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._status.setText(f"Connected to {event.game_info.title}.")
        self._caps.setText(describe_capabilities(event.capabilities))

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._caps.setText("")
        if event.expected:
            self._status.setText(STATUS_IDLE)
        else:
            self._status.setText(f"Lost the connection: {event.reason}")

    def _on_failed(self, event: ConnectionFailed) -> None:
        self._status.setText(f"Couldn't connect: {event.reason}")


def describe_capabilities(caps: Capabilities) -> str:
    """One child-friendly line: 'Can: see the screen. Can't: save and rewind.'"""
    can = [label for name, label in CAPABILITY_LABELS.items() if caps.supports(name)]
    cannot = [label for name, label in CAPABILITY_LABELS.items() if not caps.supports(name)]
    parts = []
    if can:
        parts.append("Can: " + ", ".join(can) + ".")
    if cannot:
        parts.append("Can't: " + ", ".join(cannot) + ".")
    return " ".join(parts)


__all__ = [
    "CAPABILITY_LABELS",
    "STATUS_CONNECTING",
    "STATUS_IDLE",
    "ConnectBar",
    "describe_capabilities",
]
