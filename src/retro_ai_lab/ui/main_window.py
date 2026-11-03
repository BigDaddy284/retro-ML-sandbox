"""The main window: all five panels visible together (DESIGN.md §10)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QSplitter, QStatusBar, QToolBar, QWidget

from retro_ai_lab import __version__
from retro_ai_lab.ui.connect_bar import ConnectBar
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.game_panel import GamePanel
from retro_ai_lab.ui.panels import PANELS, Panel

WINDOW_TITLE = "Retro AI Lab"


class MainWindow(QMainWindow):
    def __init__(self, context: AppContext | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("main_window")
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1200, 800)
        self._context = context if context is not None else AppContext()

        self._panels: dict[str, Panel] = {spec.key: Panel(spec) for spec in PANELS}
        self._game_panel = GamePanel(self._context)
        self._panels["game"].set_body(self._game_panel)

        toolbar = QToolBar("Connection", self)
        toolbar.setObjectName("connection_toolbar")
        toolbar.setMovable(False)
        self._connect_bar = ConnectBar(self._context)
        toolbar.addWidget(self._connect_bar)
        self.addToolBar(toolbar)

        # Layout: [ Game | Activity / Rewards ] over [ Progress | Learning ]
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

    # ------------------------------------------------------------- API

    @property
    def context(self) -> AppContext:
        return self._context

    @property
    def game_panel(self) -> GamePanel:
        return self._game_panel

    @property
    def connect_bar(self) -> ConnectBar:
        return self._connect_bar

    def panel(self, key: str) -> Panel:
        return self._panels[key]

    @property
    def panels(self) -> tuple[Panel, ...]:
        return tuple(self._panels[spec.key] for spec in PANELS)

    def set_status(self, text: str) -> None:
        bar = self.statusBar()
        bar.showMessage(text)


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
