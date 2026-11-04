"""The app-wide look (T-882): colourful, calm and easy to read.

DESIGN.md §10 asks for an interface that is "colourful, modern and approachable".
One stylesheet does it, so the widgets keep their structure and only their
presentation changes: white card panels on a soft background, a coloured header
strip per panel so each one can be found at a glance, rounded buttons whose
disabled state is obvious, and tables with clear headers and alternating rows.

Panels carry their key as the ``panel`` property (see :class:`~retro_ai_lab.ui.panels.Panel`),
which is what the per-panel rules select on. Headings inside a panel are labels
with the ``heading`` property; grey explanations carry ``hint``.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QListView,
    QStyleFactory,
    QTableView,
    QTreeView,
    QWidget,
)

THEME_MARKER = "/* retro-ai-lab theme */"
THEME_END = "/* end of retro-ai-lab theme */"
STYLE_PROPERTY = "retroAiLabStyle"  # the base style the theme picked
MIN_POINT_SIZE = 10.0  # Windows starts at 9 pt, which is small for a child to read


@dataclass(frozen=True, slots=True)
class PanelColours:
    accent: str  # the strip down the header's left edge
    tint: str  # the header's background
    ink: str  # the header's title


PANEL_COLOURS: dict[str, PanelColours] = {
    "game": PanelColours("#3b82f6", "#e3edff", "#1d4ed8"),
    "activity": PanelColours("#8b5cf6", "#efe9ff", "#6d28d9"),
    "rewards": PanelColours("#f59e0b", "#fff3d6", "#b45309"),
    "progress": PanelColours("#10b981", "#dcf7ec", "#047857"),
    "learning": PanelColours("#ec4899", "#fde7f2", "#be185d"),
    "compare": PanelColours("#06b6d4", "#d8f6fb", "#0e7490"),
}

BACKGROUND = "#eef2f8"
CARD = "#ffffff"
LINE = "#d5dbe7"
INK = "#1f2430"
MUTED = "#64748b"
FOCUS = "#3b82f6"
SELECTION = "#cfe0ff"

_BASE = f"""
QMainWindow, QDialog {{ background: {BACKGROUND}; }}
QToolBar {{ background: {CARD}; border: none; border-bottom: 1px solid {LINE};
    padding: 4px 6px; spacing: 6px; }}
QStatusBar {{ background: {CARD}; border-top: 1px solid {LINE}; color: {MUTED}; }}
QMainWindow::separator {{ background: {BACKGROUND}; width: 6px; height: 6px; }}

QFrame[panel] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 10px; }}
QWidget[panelHeader="true"] {{ border-radius: 8px; border-left: 5px solid {LINE}; }}
QWidget[panelHeader="true"] QToolButton {{ border: none; background: transparent;
    padding: 3px 6px; border-radius: 5px; color: {MUTED}; }}
QWidget[panelHeader="true"] QToolButton:hover {{ background: rgba(255, 255, 255, 0.7); }}
QWidget[panelHeader="true"] QToolButton[panelTitle="true"] {{ font-weight: bold;
    font-size: 11pt; }}

QLabel {{ color: {INK}; background: transparent; }}
QLabel[heading="true"] {{ font-weight: bold; color: #334155; padding-top: 6px; }}
QLabel[hint="true"] {{ color: {MUTED}; }}

QPushButton {{ background: {CARD}; color: {INK}; border: 1px solid #b8c2d6;
    border-radius: 6px; padding: 5px 12px; min-height: 18px; }}
QPushButton:hover {{ background: #eaf1ff; border-color: {FOCUS}; }}
QPushButton:pressed {{ background: #d6e4ff; }}
QPushButton:checked {{ background: #d6e4ff; border-color: {FOCUS}; }}
QPushButton:disabled {{ color: #a3abbb; background: #f3f5f9; border-color: #e1e5ee; }}
QPushButton#research_button {{ background: #4f46e5; color: white; border-color: #4338ca;
    font-weight: bold; }}
QPushButton#research_button:hover {{ background: #4338ca; }}

QLineEdit {{ background: {CARD}; color: {INK}; border: 1px solid #c3cbdb;
    border-radius: 5px; padding: 3px 6px; min-height: 18px; }}
QLineEdit:focus {{ border-color: {FOCUS}; }}
QLineEdit:disabled {{ color: #a3abbb; background: #f3f5f9; border-color: #e1e5ee; }}
/* Combo and spin boxes keep Fusion's own drawing (their arrows vanish once a
   stylesheet gives them a border); the palette below colours them. */

QTreeView, QListView, QTableView {{ background: {CARD}; alternate-background-color: #f5f7fb;
    border: 1px solid {LINE}; border-radius: 6px; selection-background-color: {SELECTION};
    selection-color: {INK}; }}
QHeaderView::section {{ background: {BACKGROUND}; color: #334155; font-weight: bold;
    padding: 4px 6px; border: none; border-right: 1px solid {LINE};
    border-bottom: 1px solid {LINE}; }}

QGroupBox {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 8px;
    margin-top: 4px; padding: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
    font-weight: bold; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: #e1e7f1; color: #334155; padding: 6px 16px; margin-right: 3px;
    border-top-left-radius: 7px; border-top-right-radius: 7px; }}
QTabBar::tab:selected {{ background: {CARD}; color: #4338ca; font-weight: bold; }}
QTabBar::tab:hover:!selected {{ background: #eef2ff; }}

QToolTip {{ background: #1f2430; color: white; border: none; padding: 5px 7px; }}
"""


def _panel_rules() -> str:
    rules = []
    for key, c in PANEL_COLOURS.items():
        rules.append(
            f'QWidget[panelHeader="true"][panel="{key}"] {{ background: {c.tint}; '
            f"border-left: 5px solid {c.accent}; }}\n"
            f'QWidget[panelHeader="true"][panel="{key}"] QToolButton[panelTitle="true"] '
            f"{{ color: {c.ink}; }}\n"
        )
    return "".join(rules)


def stylesheet() -> str:
    """The whole stylesheet, starting with :data:`THEME_MARKER`."""
    return THEME_MARKER + "\n" + _BASE + _panel_rules() + THEME_END + "\n"


def apply_theme(app: QApplication) -> None:
    """Use the Fusion style, a readable font size and the stylesheet; safe to repeat."""
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
        # With a stylesheet set, app.style() is Qt's stylesheet wrapper and has no name.
        app.setProperty(STYLE_PROPERTY, fusion.name())
    palette = app.palette()
    for role, colour in (
        (QPalette.ColorRole.Window, BACKGROUND),
        (QPalette.ColorRole.Base, CARD),
        (QPalette.ColorRole.AlternateBase, "#f5f7fb"),
        (QPalette.ColorRole.Button, CARD),
        (QPalette.ColorRole.Highlight, FOCUS),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
        (QPalette.ColorRole.Text, INK),
        (QPalette.ColorRole.WindowText, INK),
        (QPalette.ColorRole.ButtonText, INK),
    ):
        # Active and inactive only: Fusion keeps its greyed-out colours for disabled widgets.
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            palette.setColor(group, role, QColor(colour))
    app.setPalette(palette)
    font = app.font()
    if font.pointSizeF() < MIN_POINT_SIZE:
        font.setPointSizeF(MIN_POINT_SIZE)
        app.setFont(font)
    current = app.styleSheet()
    start = current.find(THEME_MARKER)
    if start != -1:  # take our old copy out, keep anyone else's rules
        end = current.find(THEME_END, start)
        end = len(current) if end == -1 else end + len(THEME_END) + 1
        current = current[:start] + current[end:]
    app.setStyleSheet(current + stylesheet())


def polish_views(root: QWidget) -> None:
    """Give every table and list under ``root`` alternating row colours."""
    for view in root.findChildren(QAbstractItemView):
        if isinstance(view, (QTreeView, QTableView)) or (
            isinstance(view, QListView) and view.objectName()
        ):
            view.setAlternatingRowColors(True)


__all__ = [
    "PANEL_COLOURS",
    "STYLE_PROPERTY",
    "THEME_END",
    "THEME_MARKER",
    "PanelColours",
    "apply_theme",
    "polish_views",
    "stylesheet",
]
