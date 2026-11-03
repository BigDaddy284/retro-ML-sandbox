"""The main window: all six panels visible together (DESIGN.md §10, §11)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QSplitter, QStatusBar, QToolBar, QWidget

from retro_ai_lab import __version__
from retro_ai_lab.education import PANEL_TIP
from retro_ai_lab.services import ConnectionFailed, EmulatorConnected, EmulatorDisconnected
from retro_ai_lab.ui.compare_panel import ComparePanel
from retro_ai_lab.ui.connect_bar import ConnectBar
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.discovery_panel import DiscoveryPanel
from retro_ai_lab.ui.game_panel import GamePanel
from retro_ai_lab.ui.goals_panel import GoalsPanel
from retro_ai_lab.ui.learning_panel import LearningPanel
from retro_ai_lab.ui.link_watchdog import LinkWatchdog
from retro_ai_lab.ui.panels import PANELS, Panel
from retro_ai_lab.ui.rewards_panel import RewardsPanel
from retro_ai_lab.ui.tips import TipDialog

WINDOW_TITLE = "Retro AI Lab"


class MainWindow(QMainWindow):
    def __init__(self, context: AppContext | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("main_window")
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1200, 800)
        self._context = context if context is not None else AppContext()

        self._panels: dict[str, Panel] = {spec.key: Panel(spec) for spec in PANELS}
        self._tips = TipDialog(self)
        for panel in self._panels.values():
            panel.help_requested.connect(self.explain)
        self._game_panel = GamePanel(self._context)
        self._panels["game"].set_body(self._game_panel)
        self._discovery_panel = DiscoveryPanel(self._context)
        self._panels["activity"].set_body(self._discovery_panel)
        self._rewards_panel = RewardsPanel(self._context)
        self._panels["rewards"].set_body(self._rewards_panel)
        self._goals_panel = GoalsPanel(self._context)
        self._panels["progress"].set_body(self._goals_panel)
        self._learning_panel = LearningPanel(self._context)
        self._panels["learning"].set_body(self._learning_panel)
        self._compare_panel = ComparePanel(self._context)
        self._panels["compare"].set_body(self._compare_panel)
        # Keys pressed anywhere in the window reach the controller pad.
        self.installEventFilter(self._game_panel.pad)

        toolbar = QToolBar("Connection", self)
        toolbar.setObjectName("connection_toolbar")
        toolbar.setMovable(False)
        self._connect_bar = ConnectBar(self._context)
        toolbar.addWidget(self._connect_bar)
        self.addToolBar(toolbar)

        # Layout: [ Game | Activity / Rewards ] over [ Progress | Learning | Compare ]
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setObjectName("top_splitter")
        top.addWidget(self._panels["game"])
        right = QSplitter(Qt.Orientation.Vertical)
        right.setObjectName("right_splitter")
        right.addWidget(self._panels["activity"])
        right.addWidget(self._panels["rewards"])
        top.addWidget(right)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.setObjectName("bottom_splitter")
        bottom.addWidget(self._panels["progress"])
        bottom.addWidget(self._panels["learning"])
        bottom.addWidget(self._panels["compare"])

        root = QSplitter(Qt.Orientation.Vertical)
        root.setObjectName("root_splitter")
        root.addWidget(top)
        root.addWidget(bottom)
        root.setStretchFactor(0, 3)
        root.setStretchFactor(1, 1)
        self.setCentralWidget(root)

        status = QStatusBar(self)
        status.setObjectName("status_bar")
        self.setStatusBar(status)
        self.set_status("Ready. Connect to a game to begin.")
        status.addPermanentWidget(_VersionLabel(self))

        self._watchdog = LinkWatchdog(self._context.connection, parent=self)
        self._watchdog.start()
        bridge = self._context.bridge
        unsubscribe = (
            bridge.subscribe(EmulatorConnected, self._on_connected),
            bridge.subscribe(EmulatorDisconnected, self._on_disconnected),
            bridge.subscribe(ConnectionFailed, self._on_failed),
        )
        self.destroyed.connect(lambda *_: [off() for off in unsubscribe])

    # ------------------------------------------------------------- API

    @property
    def context(self) -> AppContext:
        return self._context

    @property
    def discovery_panel(self) -> DiscoveryPanel:
        return self._discovery_panel

    @property
    def goals_panel(self) -> GoalsPanel:
        return self._goals_panel

    @property
    def learning_panel(self) -> LearningPanel:
        return self._learning_panel

    @property
    def compare_panel(self) -> ComparePanel:
        return self._compare_panel

    @property
    def rewards_panel(self) -> RewardsPanel:
        return self._rewards_panel

    @property
    def game_panel(self) -> GamePanel:
        return self._game_panel

    @property
    def connect_bar(self) -> ConnectBar:
        return self._connect_bar

    def panel(self, key: str) -> Panel:
        return self._panels[key]

    @property
    def tip_dialog(self) -> TipDialog:
        return self._tips

    def explain(self, panel_key: str) -> None:
        """Open the "What's this?" window on ``panel_key``'s tip."""
        self._tips.show_tip(PANEL_TIP[panel_key])

    @property
    def panels(self) -> tuple[Panel, ...]:
        return tuple(self._panels[spec.key] for spec in PANELS)

    @property
    def watchdog(self) -> LinkWatchdog:
        return self._watchdog

    def set_status(self, text: str) -> None:
        bar = self.statusBar()
        bar.showMessage(text)

    def _on_connected(self, event: EmulatorConnected) -> None:
        self.set_status(f"Connected to {event.game_info.title}. Have fun!")

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        if event.expected:
            self.set_status("Disconnected. Connect to a game to begin.")
        else:
            self.set_status(f"Lost the game: {event.reason} You can reconnect from the toolbar.")

    def _on_failed(self, event: ConnectionFailed) -> None:
        self.set_status(f"Couldn't connect: {event.reason}")


class _VersionLabel(QWidget):
    """Small permanent status-bar widget showing the version."""

    def __init__(self, parent: QWidget | None = None) -> None:
        from PySide6.QtWidgets import QHBoxLayout, QLabel

        super().__init__(parent)
        self.setObjectName("version_label")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 6, 0)
        label = QLabel(f"v{__version__}", self)
        label.setObjectName("version_text")
        layout.addWidget(label)


__all__ = ["WINDOW_TITLE", "MainWindow"]
