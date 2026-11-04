"""The main window: all six panels visible together (DESIGN.md §10, §11).

Each panel lives in a dock widget (T-837), so it can be dragged to another
edge, dropped on another panel to make tabs, pulled out as its own window,
resized, hidden and brought back from the toolbar's "Panels" menu. The
arrangement is saved when the window closes and put back on the next start.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QMenu,
    QPushButton,
    QStatusBar,
    QToolBar,
    QToolButton,
    QWidget,
)

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
from retro_ai_lab.ui.research_dialog import ResearchDialog
from retro_ai_lab.ui.rewards_panel import RewardsPanel
from retro_ai_lab.ui.tips import TipDialog

WINDOW_TITLE = "Retro AI Lab"
LAYOUT_FILE = "layout.bin"  # the dock arrangement, under the app's data folder (T-837)
LAYOUT_VERSION = 1  # bump when the set of docks or toolbars changes
PANELS_MENU = "Panels"
RESET_LAYOUT = "Reset layout"

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self,
        context: AppContext | None = None,
        parent: QWidget | None = None,
        *,
        layout_path: Path | None = None,
    ) -> None:
        """``layout_path`` is where the dock arrangement is kept; None forgets it."""
        super().__init__(parent)
        self.setObjectName("main_window")
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1200, 800)
        self._context = context if context is not None else AppContext()
        self._layout_path = layout_path

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
        self._research = ResearchDialog(self._context, self)
        self._research_button = QPushButton("Research", self)
        self._research_button.setObjectName("research_button")
        self._research_button.setToolTip("For grown-ups: settings, values and the step log.")
        self._research_button.clicked.connect(self.open_research)
        toolbar.addWidget(self._research_button)

        # Every panel is a dock (T-837): no central widget, so the docks fill the window.
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.GroupedDragging
        )
        self._docks: dict[str, QDockWidget] = {}
        for spec in PANELS:
            dock = QDockWidget(spec.title, self)
            dock.setObjectName(f"dock_{spec.key}")
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                | QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
            panel = self._panels[spec.key]
            dock.setWidget(panel)
            # One title per panel (T-881): the panel's header is the dock's title bar.
            dock.setTitleBarWidget(panel.header)
            panel.show_dock_buttons()
            panel.pop_out_requested.connect(lambda d=dock: d.setFloating(not d.isFloating()))
            panel.hide_requested.connect(dock.close)
            dock.topLevelChanged.connect(panel.set_floating)
            self._docks[spec.key] = dock
        self._panels_menu = QMenu(PANELS_MENU, self)
        self._panels_menu.setObjectName("panels_menu")
        for spec in PANELS:
            action = self._docks[spec.key].toggleViewAction()
            action.setObjectName(f"show_{spec.key}")
            action.setText(spec.title)
            self._panels_menu.addAction(action)
        self._panels_menu.addSeparator()
        self._reset_action = QAction(RESET_LAYOUT, self)
        self._reset_action.setObjectName("reset_layout")
        self._reset_action.triggered.connect(self.reset_layout)
        self._panels_menu.addAction(self._reset_action)
        panels_button = QToolButton(self)
        panels_button.setObjectName("panels_button")
        panels_button.setText(PANELS_MENU)
        panels_button.setToolTip("Hide or show a panel, or put them all back where they started.")
        panels_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        panels_button.setMenu(self._panels_menu)
        toolbar.addWidget(panels_button)
        self.addToolBar(toolbar)
        self._default_layout()
        self._restore_layout()

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

    def dock(self, key: str) -> QDockWidget:
        """The dock that holds panel ``key`` (T-837)."""
        return self._docks[key]

    @property
    def docks(self) -> tuple[QDockWidget, ...]:
        return tuple(self._docks[spec.key] for spec in PANELS)

    @property
    def panels_menu(self) -> QMenu:
        """The toolbar's "Panels" menu: one tick per panel, then "Reset layout"."""
        return self._panels_menu

    @property
    def layout_path(self) -> Path | None:
        return self._layout_path

    def reset_layout(self) -> None:
        """Put every panel back where it started, docked and visible."""
        for dock in self._docks.values():
            dock.setFloating(False)
            self.removeDockWidget(dock)
        self._default_layout()

    def _default_layout(self) -> None:
        """[ Game | Activity / Rewards ] over [ Progress | Learning | Compare ]."""
        d = self._docks
        area = Qt.DockWidgetArea
        self.addDockWidget(area.LeftDockWidgetArea, d["game"])
        self.addDockWidget(area.RightDockWidgetArea, d["activity"])
        self.splitDockWidget(d["activity"], d["rewards"], Qt.Orientation.Vertical)
        self.addDockWidget(area.BottomDockWidgetArea, d["progress"])
        self.splitDockWidget(d["progress"], d["learning"], Qt.Orientation.Horizontal)
        self.splitDockWidget(d["learning"], d["compare"], Qt.Orientation.Horizontal)
        for dock in d.values():
            dock.show()
        self.resizeDocks([d["game"], d["activity"]], [720, 480], Qt.Orientation.Horizontal)
        self.resizeDocks([d["game"], d["progress"]], [540, 220], Qt.Orientation.Vertical)

    def _restore_layout(self) -> None:
        path = self._layout_path
        if path is None or not path.is_file():
            return
        try:
            data = path.read_bytes()
        except OSError as exc:
            log.warning("Could not read the panel layout from %s: %s", path, exc)
            return
        if not self.restoreState(QByteArray(data), LAYOUT_VERSION):
            log.warning("The panel layout in %s was not usable; using the default.", path)
            self.reset_layout()

    def _save_layout(self) -> None:
        path = self._layout_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(self.saveState(LAYOUT_VERSION).data()))
        except OSError as exc:
            log.warning("Could not save the panel layout to %s: %s", path, exc)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt's name)
        self._save_layout()
        super().closeEvent(event)

    @property
    def tip_dialog(self) -> TipDialog:
        return self._tips

    @property
    def research_dialog(self) -> ResearchDialog:
        return self._research

    def open_research(self) -> None:
        """Bring the Research window up."""
        self._research.show()
        self._research.raise_()
        self._research.activateWindow()

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


__all__ = ["LAYOUT_FILE", "PANELS_MENU", "RESET_LAYOUT", "WINDOW_TITLE", "MainWindow"]
