"""The app-wide look (T-886): the theme of Stills & Reels and StackSpace.

The user's other two apps share one theme, so this one has it too. Every colour
is taken value for value from Stills & Reels' ``src/Theme.cs`` (StackSpace's
``lib/app/theme.dart`` copies the same values and adds error and success):
dark by default, a light mode, and "Match Windows" as StackSpace offers. The
controls follow Stills & Reels' ``src/UiControls.cs``: 8 px rounded buttons on
the Control fill, the primary button on the blue-to-purple accent gradient,
checked buttons on a faint gradient with an accent edge, small capital section
labels, the mode pill (here, the tabs), bars on the Bar colour with hairlines,
and the Windows title bar coloured to match.

Two choices differ from Stills & Reels on purpose, as StackSpace's do: small
text uses SubText rather than Muted (Muted is below 4.5:1), and each panel's
header keeps a thin strip in its own colour so a child can tell the panels
apart (the user agreed to that on 4 Nov).

Panels carry their key as the ``panel`` property, headers ``panelHeader``,
section headings ``heading``, grey explanations ``hint`` and warnings
``warning``; the stylesheet selects on those.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QFontDatabase, QGuiApplication, QPalette
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
THEME_FILE = "theme.json"  # the person's choice, under the app's data folder
POINT_SIZE = 10.0  # Stills & Reels' button text size
FONT_FAMILIES = ("Segoe UI Variable Text", "Segoe UI")  # the first one installed wins

log = logging.getLogger(__name__)


class ThemeMode(Enum):
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"  # follow Windows


MODE_ORDER: tuple[ThemeMode, ...] = (ThemeMode.DARK, ThemeMode.LIGHT, ThemeMode.SYSTEM)
MODE_LABELS: dict[ThemeMode, str] = {
    ThemeMode.DARK: "Dark mode",
    ThemeMode.LIGHT: "Light mode",
    ThemeMode.SYSTEM: "Match Windows",
}


@dataclass(frozen=True, slots=True)
class Palette:
    """Stills & Reels' ``Palette`` class, field for field (colours as #rrggbb)."""

    dark: bool
    window: str  # form background
    bar: str  # top bar, status bar, toolbars
    bar_border: str  # hairlines between bars and content
    canvas: str  # picture / graph area
    panel: str  # sidebars, pop-outs
    text: str
    sub_text: str
    muted: str  # disabled controls only (below 4.5:1 for text)
    hover: str
    pressed: str
    pill: str  # inside of the mode toolbar (here, the tabs)
    pill_sep: str
    accent_a: str  # gradient start (blue)
    accent_b: str  # gradient end (purple)
    selected_text: str
    control: str  # buttons, input fills, slider tracks
    control_border: str
    warning: str
    glow: str
    video_bg: str
    error: str  # StackSpace's addition
    success: str  # StackSpace's addition


def _hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


DARK = Palette(
    dark=True,
    window=_hex(10, 13, 26),
    bar=_hex(13, 17, 33),
    bar_border=_hex(30, 37, 62),
    canvas=_hex(9, 12, 24),
    panel=_hex(14, 19, 37),
    text=_hex(234, 238, 250),
    sub_text=_hex(172, 182, 210),
    muted=_hex(104, 114, 146),
    hover=_hex(26, 33, 62),
    pressed=_hex(34, 43, 80),
    pill=_hex(12, 18, 44),
    pill_sep=_hex(62, 74, 128),
    accent_a=_hex(38, 104, 255),
    accent_b=_hex(176, 72, 255),
    selected_text=_hex(255, 255, 255),
    control=_hex(32, 40, 70),
    control_border=_hex(46, 56, 92),
    warning=_hex(255, 190, 110),
    glow=_hex(40, 70, 150),
    video_bg=_hex(5, 7, 14),
    error=_hex(255, 120, 140),
    success=_hex(80, 200, 140),
)

LIGHT = Palette(
    dark=False,
    window=_hex(244, 246, 251),
    bar=_hex(255, 255, 255),
    bar_border=_hex(221, 226, 238),
    canvas=_hex(236, 240, 247),
    panel=_hex(250, 251, 254),
    text=_hex(20, 26, 45),
    sub_text=_hex(74, 85, 112),
    muted=_hex(138, 147, 168),
    hover=_hex(232, 237, 249),
    pressed=_hex(218, 226, 246),
    pill=_hex(243, 245, 255),
    pill_sep=_hex(196, 204, 232),
    accent_a=_hex(37, 99, 245),
    accent_b=_hex(150, 62, 240),
    selected_text=_hex(255, 255, 255),
    control=_hex(222, 228, 242),
    control_border=_hex(200, 208, 228),
    warning=_hex(196, 98, 0),
    glow=_hex(160, 185, 255),
    video_bg=_hex(12, 14, 22),
    error=_hex(190, 30, 60),
    success=_hex(20, 120, 70),
)

# One strip colour per panel, so the panels stay easy to tell apart (T-882's idea, kept).
PANEL_ACCENTS: dict[str, str] = {
    "game": "#3b82f6",
    "activity": "#8b5cf6",
    "rewards": "#f59e0b",
    "progress": "#10b981",
    "learning": "#ec4899",
    "compare": "#06b6d4",
}

_state: dict[str, object] = {"palette": DARK, "mode": ThemeMode.DARK}


def current() -> Palette:
    """The palette in use now."""
    palette = _state["palette"]
    assert isinstance(palette, Palette)
    return palette


def current_mode() -> ThemeMode:
    mode = _state["mode"]
    assert isinstance(mode, ThemeMode)
    return mode


def next_mode(mode: ThemeMode) -> ThemeMode:
    return MODE_ORDER[(MODE_ORDER.index(mode) + 1) % len(MODE_ORDER)]


# ------------------------------------------------------------------ colour maths


def _luminance(colour: str) -> float:
    c = QColor(colour)

    def channel(v: int) -> float:
        s = v / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(c.red()) + 0.7152 * channel(c.green()) + 0.0722 * channel(c.blue())


def contrast(a: str, b: str) -> float:
    """The WCAG contrast ratio between two colours (1 to 21)."""
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _rgba(colour: str, alpha: int) -> str:
    c = QColor(colour)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})"


def _blend(a: str, b: str, t: float) -> str:
    ca, cb = QColor(a), QColor(b)
    return _hex(
        round(ca.red() + (cb.red() - ca.red()) * t),
        round(ca.green() + (cb.green() - ca.green()) * t),
        round(ca.blue() + (cb.blue() - ca.blue()) * t),
    )


def _gradient(a: str, b: str) -> str:
    return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {a}, stop:1 {b})"


# ------------------------------------------------------------------ the stylesheet


def stylesheet(p: Palette) -> str:
    """The whole stylesheet for ``p``, between :data:`THEME_MARKER` and :data:`THEME_END`."""
    accent = _gradient(p.accent_a, p.accent_b)
    accent_hover = _gradient(
        _blend(p.accent_a, "#ffffff", 0.11), _blend(p.accent_b, "#ffffff", 0.11)
    )
    faint = 70 if p.dark else 40  # Stills & Reels' checked-button alpha
    faint_accent = _gradient(_rgba(p.accent_a, faint), _rgba(p.accent_b, faint))
    stripe = _blend(p.panel, p.hover, 0.5)
    rules = f"""
QMainWindow, QDialog {{ background: {p.window}; color: {p.text}; }}
QToolBar {{ background: {p.bar}; border: none; border-bottom: 1px solid {p.bar_border};
    padding: 6px 10px; spacing: 8px; }}
QStatusBar {{ background: {p.bar}; border-top: 1px solid {p.bar_border}; color: {p.sub_text};
    min-height: 28px; }}
QStatusBar QLabel {{ color: {p.sub_text}; }}
QMainWindow::separator {{ background: {p.window}; width: 6px; height: 6px; }}

QFrame[panel] {{ background: {p.panel}; border: 1px solid {p.bar_border}; border-radius: 10px; }}
QWidget[panelHeader="true"] {{ background: {p.bar}; border: 1px solid {p.bar_border};
    border-left: 4px solid {p.bar_border}; border-radius: 8px; }}
QWidget[panelHeader="true"] QToolButton {{ border: none; background: transparent;
    padding: 4px 8px; border-radius: 6px; color: {p.sub_text}; }}
QWidget[panelHeader="true"] QToolButton:hover {{ background: {p.hover}; color: {p.text}; }}
QWidget[panelHeader="true"] QToolButton:pressed {{ background: {p.pressed}; }}
QWidget[panelHeader="true"] QToolButton[panelTitle="true"] {{ color: {p.text};
    font-weight: bold; font-size: 11pt; }}

QLabel {{ color: {p.text}; background: transparent; }}
QLabel[heading="true"] {{ color: {p.sub_text}; font-weight: bold; font-size: 8.5pt;
    padding-top: 8px; }}
QLabel[hint="true"] {{ color: {p.sub_text}; }}
QLabel[warning="true"] {{ color: {p.warning}; }}

QPushButton {{ background: {p.control}; color: {p.text}; border: 1px solid transparent;
    border-radius: 8px; padding: 6px 14px; min-height: 20px; }}
QPushButton:hover {{ background: {p.hover}; }}
QPushButton:pressed {{ background: {p.pressed}; }}
QPushButton:checked {{ background: {faint_accent}; border: 1px solid {p.accent_a}; }}
QPushButton:focus {{ border: 1px dotted {p.accent_a}; }}
QPushButton:disabled {{ color: {p.muted}; background: {p.control}; border-color: transparent; }}
QPushButton#research_button, QPushButton[primary="true"] {{ background: {accent};
    color: {p.selected_text}; font-weight: bold; }}
QPushButton#research_button:hover, QPushButton[primary="true"]:hover {{
    background: {accent_hover}; }}

QToolButton {{ background: transparent; color: {p.text}; border: none; border-radius: 8px;
    padding: 6px 10px; }}
QToolButton:hover {{ background: {p.hover}; }}
QToolButton:pressed {{ background: {p.pressed}; }}

QLineEdit {{ background: {p.control}; color: {p.text}; border: 1px solid transparent;
    border-radius: 8px; padding: 5px 8px; min-height: 20px;
    selection-background-color: {p.accent_a}; selection-color: {p.selected_text}; }}
QLineEdit:focus {{ border: 1px solid {p.accent_a}; }}
QLineEdit:disabled {{ color: {p.muted}; }}
/* Combo and spin boxes keep Fusion's own drawing (their arrows vanish once a
   stylesheet gives them a border); the Qt palette below colours them. */

QTreeView, QListView, QTableView {{ background: {p.panel}; alternate-background-color: {stripe};
    color: {p.text}; border: 1px solid {p.bar_border}; border-radius: 8px;
    selection-background-color: {_rgba(p.accent_a, 110)}; selection-color: {p.text}; }}
QTreeView::item:hover, QListView::item:hover {{ background: {p.hover}; }}
QHeaderView::section {{ background: {p.bar}; color: {p.sub_text}; font-weight: bold;
    padding: 5px 8px; border: none; border-right: 1px solid {p.bar_border};
    border-bottom: 1px solid {p.bar_border}; }}
QTableCornerButton::section {{ background: {p.bar}; border: none; }}

QGroupBox {{ background: {p.panel}; border: 1px solid {p.bar_border}; border-radius: 10px;
    margin-top: 4px; padding: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px;
    color: {p.sub_text}; font-weight: bold; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: {p.pill}; color: {p.sub_text}; border: 1px solid {p.pill_sep};
    padding: 7px 18px; margin-right: 4px; border-radius: 8px; }}
QTabBar::tab:hover:!selected {{ background: {p.hover}; color: {p.text}; }}
QTabBar::tab:selected {{ background: {accent}; color: {p.selected_text}; font-weight: bold;
    border-color: transparent; }}

QMenu {{ background: {p.panel}; color: {p.text}; border: 1px solid {p.control_border};
    padding: 4px; }}
QMenu::item {{ padding: 6px 22px 6px 18px; border-radius: 5px; }}
QMenu::item:selected {{ background: {p.hover}; color: {p.text}; }}
QMenu::item:disabled {{ color: {p.muted}; }}
QMenu::separator {{ height: 1px; background: {p.bar_border}; margin: 4px 8px; }}
QToolTip {{ background: {p.panel}; color: {p.text}; border: 1px solid {p.control_border};
    padding: 5px 8px; }}

QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.control}; border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:horizontal {{ background: {p.control}; border-radius: 4px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {p.control_border}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""
    strips = "".join(
        f'QWidget[panelHeader="true"][panel="{key}"] {{ border-left: 4px solid {colour}; }}\n'
        for key, colour in PANEL_ACCENTS.items()
    )
    return THEME_MARKER + "\n" + rules + strips + THEME_END + "\n"


def _qt_palette(app: QApplication, p: Palette) -> QPalette:
    palette = app.palette()
    roles = {
        QPalette.ColorRole.Window: p.window,
        QPalette.ColorRole.WindowText: p.text,
        QPalette.ColorRole.Base: p.control,
        QPalette.ColorRole.AlternateBase: _blend(p.panel, p.hover, 0.5),
        QPalette.ColorRole.Button: p.control,
        QPalette.ColorRole.ButtonText: p.text,
        QPalette.ColorRole.Text: p.text,
        QPalette.ColorRole.BrightText: p.selected_text,
        QPalette.ColorRole.Highlight: p.accent_a,
        QPalette.ColorRole.HighlightedText: p.selected_text,
        QPalette.ColorRole.ToolTipBase: p.panel,
        QPalette.ColorRole.ToolTipText: p.text,
        QPalette.ColorRole.PlaceholderText: p.sub_text,
        QPalette.ColorRole.Link: p.accent_a,
        QPalette.ColorRole.Light: p.control_border,
        QPalette.ColorRole.Midlight: p.control_border,
        QPalette.ColorRole.Mid: p.bar_border,
        QPalette.ColorRole.Dark: p.bar,
        QPalette.ColorRole.Shadow: p.window,
    }
    for role, colour in roles.items():
        palette.setColor(QPalette.ColorGroup.All, role, QColor(colour))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(p.muted))
    return palette


def system_prefers_dark() -> bool:
    """Is Windows (or the desktop) set to dark? Light when it can't be told."""
    try:
        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:  # Qt before 6.5
        return False


def palette_for(mode: ThemeMode) -> Palette:
    if mode is ThemeMode.SYSTEM:
        return DARK if system_prefers_dark() else LIGHT
    return DARK if mode is ThemeMode.DARK else LIGHT


def apply_theme(app: QApplication, mode: ThemeMode = ThemeMode.DARK) -> Palette:
    """Use Fusion, Segoe UI Variable, the Qt palette and the stylesheet; safe to repeat."""
    p = palette_for(mode)
    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
        # With a stylesheet set, app.style() is Qt's stylesheet wrapper and has no name.
        app.setProperty(STYLE_PROPERTY, fusion.name())
    font = app.font()
    families = set(QFontDatabase.families())
    for family in FONT_FAMILIES:
        if family in families:
            font.setFamily(family)
            break
    if font.pointSizeF() < POINT_SIZE:
        font.setPointSizeF(POINT_SIZE)
    app.setFont(font)
    app.setPalette(_qt_palette(app, p))
    current_sheet = app.styleSheet()
    start = current_sheet.find(THEME_MARKER)
    if start != -1:  # take our old copy out, keep anyone else's rules
        end = current_sheet.find(THEME_END, start)
        end = len(current_sheet) if end == -1 else end + len(THEME_END) + 1
        current_sheet = current_sheet[:start] + current_sheet[end:]
    app.setStyleSheet(current_sheet + stylesheet(p))
    _state["palette"], _state["mode"] = p, mode
    for widget in app.topLevelWidgets():
        if widget.isVisible():
            set_title_bar(widget, p)
        widget.update()
    return p


# ------------------------------------------------------------------ the choice on disk


def load_mode(path: Path | None) -> ThemeMode:
    """The saved choice, or dark when there is none or it can't be read."""
    if path is None:
        return ThemeMode.DARK
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ThemeMode(data["mode"])
    except (OSError, ValueError, KeyError, TypeError):
        return ThemeMode.DARK


def save_mode(path: Path | None, mode: ThemeMode) -> bool:
    if path is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mode": mode.value}), encoding="utf-8")
    except OSError as exc:
        log.warning("Could not save the theme to %s: %s", path, exc)
        return False
    return True


# ------------------------------------------------------------------ the Windows title bar

_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19
_DWMWA_BORDER_COLOR = 34
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36


def _colorref(colour: str) -> int:
    c = QColor(colour)
    return c.red() | (c.green() << 8) | (c.blue() << 16)


def set_title_bar(widget: QWidget, p: Palette) -> bool:
    """Dark or light title bar (Windows 10 2004+) in the Bar colour (Windows 11)."""
    if sys.platform != "win32" or not widget.isWindow():
        return False
    try:
        import ctypes

        hwnd = int(widget.winId())
        if not hwnd:
            return False
        dwm = getattr(ctypes, "windll").dwmapi  # noqa: B009 (windll exists only on Windows)

        def put(attribute: int, value: int) -> int:
            v = ctypes.c_int(value)
            return int(dwm.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(v), 4))

        dark = 1 if p.dark else 0
        if put(_DWMWA_USE_IMMERSIVE_DARK_MODE, dark) != 0:
            put(_DWMWA_USE_IMMERSIVE_DARK_MODE_OLD, dark)
        put(_DWMWA_CAPTION_COLOR, _colorref(p.bar))
        put(_DWMWA_TEXT_COLOR, _colorref(p.text))
        put(_DWMWA_BORDER_COLOR, _colorref("#283054" if p.dark else p.bar_border))
    except (OSError, AttributeError, ValueError) as exc:
        log.debug("Could not colour the title bar: %s", exc)
        return False
    return True


class ThemeWatcher(QObject):
    """Colours each window's title bar as it opens; follows Windows in "system" mode."""

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._app = app
        app.installEventFilter(self)
        with contextlib.suppress(AttributeError):  # Qt before 6.5 has no colour scheme
            QGuiApplication.styleHints().colorSchemeChanged.connect(self._on_scheme)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget) and watched.isWindow():
            set_title_bar(watched, current())
        return False

    def _on_scheme(self, *_: object) -> None:
        if current_mode() is ThemeMode.SYSTEM:
            apply_theme(self._app, ThemeMode.SYSTEM)


def install_watcher(app: QApplication) -> ThemeWatcher:
    """One watcher per app; repeat calls return it."""
    existing = app.findChild(ThemeWatcher)
    if isinstance(existing, ThemeWatcher):
        return existing
    return ThemeWatcher(app)


def polish_views(root: QWidget) -> None:
    """Give every table and list under ``root`` alternating row colours."""
    for view in root.findChildren(QAbstractItemView):
        if isinstance(view, (QTreeView, QTableView)) or (
            isinstance(view, QListView) and view.objectName()
        ):
            view.setAlternatingRowColors(True)


__all__ = [
    "DARK",
    "LIGHT",
    "MODE_LABELS",
    "MODE_ORDER",
    "PANEL_ACCENTS",
    "STYLE_PROPERTY",
    "THEME_END",
    "THEME_FILE",
    "THEME_MARKER",
    "Palette",
    "ThemeMode",
    "ThemeWatcher",
    "apply_theme",
    "contrast",
    "current",
    "current_mode",
    "install_watcher",
    "load_mode",
    "next_mode",
    "palette_for",
    "polish_views",
    "save_mode",
    "set_title_bar",
    "stylesheet",
    "system_prefers_dark",
]
