"""T-886 (replacing T-882's look): the Stills & Reels / StackSpace theme."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QAbstractItemView, QApplication, QToolButton, QWidget
from pytestqt.qtbot import QtBot

from retro_ai_lab.ui import theme
from retro_ai_lab.ui.app import create_app
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.main_window import MainWindow
from retro_ai_lab.ui.panels import PANELS
from retro_ai_lab.ui.theme import (
    DARK,
    LIGHT,
    MODE_LABELS,
    PANEL_ACCENTS,
    THEME_END,
    THEME_MARKER,
    ThemeMode,
    apply_theme,
    contrast,
    load_mode,
    polish_views,
    save_mode,
    set_title_bar,
    stylesheet,
)


@pytest.fixture(autouse=True)
def back_to_dark(qapp: QApplication):  # type: ignore[no-untyped-def]
    """Every test leaves the app in the default look."""
    yield
    apply_theme(qapp, ThemeMode.DARK)


# Stills & Reels src/Theme.cs, value for value (StackSpace lib/app/theme.dart copies it).
SR_DARK = {
    "window": (10, 13, 26), "bar": (13, 17, 33), "bar_border": (30, 37, 62),
    "canvas": (9, 12, 24), "panel": (14, 19, 37), "text": (234, 238, 250),
    "sub_text": (172, 182, 210), "muted": (104, 114, 146), "hover": (26, 33, 62),
    "pressed": (34, 43, 80), "pill": (12, 18, 44), "pill_sep": (62, 74, 128),
    "accent_a": (38, 104, 255), "accent_b": (176, 72, 255), "control": (32, 40, 70),
    "control_border": (46, 56, 92), "warning": (255, 190, 110), "glow": (40, 70, 150),
    "video_bg": (5, 7, 14),
}  # fmt: skip
SR_LIGHT = {
    "window": (244, 246, 251), "bar": (255, 255, 255), "bar_border": (221, 226, 238),
    "canvas": (236, 240, 247), "panel": (250, 251, 254), "text": (20, 26, 45),
    "sub_text": (74, 85, 112), "muted": (138, 147, 168), "hover": (232, 237, 249),
    "pressed": (218, 226, 246), "pill": (243, 245, 255), "pill_sep": (196, 204, 232),
    "accent_a": (37, 99, 245), "accent_b": (150, 62, 240), "control": (222, 228, 242),
    "control_border": (200, 208, 228), "warning": (196, 98, 0), "glow": (160, 185, 255),
    "video_bg": (12, 14, 22),
}  # fmt: skip


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    c = QColor(hex_colour)
    return (c.red(), c.green(), c.blue())


@pytest.mark.parametrize(("palette", "expected"), [(DARK, SR_DARK), (LIGHT, SR_LIGHT)])
def test_the_palettes_are_stills_and_reels_value_for_value(
    palette: theme.Palette, expected: dict[str, tuple[int, int, int]]
) -> None:
    for name, rgb in expected.items():
        assert _rgb(getattr(palette, name)) == rgb, name
    assert palette.dark is (palette is DARK)
    assert _rgb(palette.selected_text) == (255, 255, 255)


@pytest.mark.parametrize("palette", [DARK, LIGHT])
def test_text_is_readable(palette: theme.Palette) -> None:
    for surface in (palette.window, palette.bar, palette.panel, palette.control):
        assert contrast(palette.text, surface) >= 4.5
    for surface in (palette.panel, palette.bar, palette.window):
        assert contrast(palette.sub_text, surface) >= 4.5
    assert contrast("#ffffff", palette.accent_a) >= 4.5
    # Kept exactly as Stills & Reels has it: 3.98:1 in light mode, the large-text level.
    assert contrast(palette.warning, palette.panel) >= 3.0
    assert contrast(palette.success, palette.panel) >= 4.5


def test_contrast_is_the_wcag_ratio() -> None:
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#777777", "#777777") == pytest.approx(1.0)


@pytest.mark.parametrize("palette", [DARK, LIGHT])
def test_the_stylesheet_uses_the_palette(palette: theme.Palette) -> None:
    sheet = stylesheet(palette)
    assert sheet.startswith(THEME_MARKER) and sheet.rstrip().endswith(THEME_END)
    for colour in (palette.window, palette.bar, palette.panel, palette.text, palette.control):
        assert colour in sheet
    assert "qlineargradient" in sheet and palette.accent_a in sheet and palette.accent_b in sheet
    assert set(PANEL_ACCENTS) == {spec.key for spec in PANELS}
    for key, accent in PANEL_ACCENTS.items():
        assert f'[panel="{key}"]' in sheet and accent in sheet


def test_apply_theme_twice_keeps_one_copy_and_sets_the_qt_palette(qapp: QApplication) -> None:
    before = qapp.styleSheet()
    try:
        qapp.setStyleSheet("QLabel#mine { color: red; }")
        assert apply_theme(qapp, ThemeMode.LIGHT) is LIGHT
        assert apply_theme(qapp, ThemeMode.DARK) is DARK
        sheet = qapp.styleSheet()
        assert sheet.count(THEME_MARKER) == 1 and "QLabel#mine { color: red; }" in sheet
        assert theme.current() is DARK and theme.current_mode() is ThemeMode.DARK
        pal = qapp.palette()
        assert pal.color(QPalette.ColorRole.Window).name() == DARK.window
        assert pal.color(QPalette.ColorRole.Text).name() == DARK.text
        assert pal.color(QPalette.ColorRole.Highlight).name() == DARK.accent_a
        disabled = pal.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)
        assert disabled.name() == DARK.muted
        assert str(qapp.property(theme.STYLE_PROPERTY)).lower() == "fusion"
        assert qapp.font().pointSizeF() >= 10
    finally:
        qapp.setStyleSheet(before)
        apply_theme(qapp, ThemeMode.DARK)


def test_match_windows_follows_the_colour_scheme(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(theme, "system_prefers_dark", lambda: False)
    assert apply_theme(qapp, ThemeMode.SYSTEM) is LIGHT
    assert theme.current_mode() is ThemeMode.SYSTEM
    monkeypatch.setattr(theme, "system_prefers_dark", lambda: True)
    assert apply_theme(qapp, ThemeMode.SYSTEM) is DARK


def test_create_app_applies_the_theme(qapp: QApplication) -> None:
    assert create_app() is qapp
    assert qapp.styleSheet().count(THEME_MARKER) == 1
    assert theme.current() is DARK


@pytest.mark.parametrize("mode", [ThemeMode.DARK, ThemeMode.LIGHT])
def test_the_stylesheet_parses_on_every_window(
    qtbot: QtBot,
    qapp: QApplication,
    tmp_path: Path,
    qtlog,  # type: ignore[no-untyped-def]
    mode: ThemeMode,
) -> None:
    apply_theme(qapp, mode)
    w = MainWindow(AppContext(data_dir=tmp_path))
    qtbot.addWidget(w)
    w.show()
    w.open_research()
    for tab in range(w.research_dialog.tabs.count()):
        w.research_dialog.tabs.setCurrentIndex(tab)
        qapp.processEvents()
    w.grab()
    w.research_dialog.grab()
    assert [r.message for r in qtlog.records if "style" in r.message.lower()] == []
    w.research_dialog.close()


def test_tables_alternate_rows(qtbot: QtBot, tmp_path: Path) -> None:
    w = MainWindow(AppContext(data_dir=tmp_path))
    qtbot.addWidget(w)
    views = [*w.findChildren(QAbstractItemView), *w.research_dialog.findChildren(QAbstractItemView)]
    trees = [v for v in views if v.objectName()]
    assert len(trees) >= 8 and all(v.alternatingRowColors() for v in trees)
    polish_views(w)
    assert all(v.alternatingRowColors() for v in trees)


def test_the_theme_button_cycles_and_is_remembered(
    qtbot: QtBot, qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(theme, "system_prefers_dark", lambda: False)
    path = tmp_path / "theme.json"
    w = MainWindow(AppContext(data_dir=tmp_path), theme_path=path)
    qtbot.addWidget(w)
    button = w.findChild(QToolButton, "theme_button")
    assert button is not None and button.text() == MODE_LABELS[ThemeMode.DARK]
    button.click()
    assert theme.current_mode() is ThemeMode.LIGHT and theme.current() is LIGHT
    assert button.text() == MODE_LABELS[ThemeMode.LIGHT] and load_mode(path) is ThemeMode.LIGHT
    button.click()
    assert theme.current_mode() is ThemeMode.SYSTEM and theme.current() is LIGHT  # Windows: light
    assert load_mode(path) is ThemeMode.SYSTEM
    button.click()
    assert theme.current_mode() is ThemeMode.DARK and load_mode(path) is ThemeMode.DARK
    # The next window starts with the saved choice.
    save_mode(path, ThemeMode.LIGHT)
    again = MainWindow(AppContext(data_dir=tmp_path), theme_path=path)
    qtbot.addWidget(again)
    assert theme.current() is LIGHT
    assert again.findChild(QToolButton, "theme_button").text() == MODE_LABELS[ThemeMode.LIGHT]  # type: ignore[union-attr]


def test_a_missing_or_broken_theme_file_means_dark(tmp_path: Path) -> None:
    assert load_mode(tmp_path / "nowhere.json") is ThemeMode.DARK
    bad = tmp_path / "theme.json"
    for text in ("not json", '{"mode": "purple"}', "[1, 2]", '{"mode": 3}'):
        bad.write_text(text, encoding="utf-8")
        assert load_mode(bad) is ThemeMode.DARK
    save_mode(tmp_path / "deep" / "theme.json", ThemeMode.SYSTEM)
    assert load_mode(tmp_path / "deep" / "theme.json") is ThemeMode.SYSTEM
    assert save_mode(tmp_path, ThemeMode.LIGHT) is False  # a folder: refused quietly


def test_the_title_bar_is_left_alone_off_windows(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = QWidget()
    qtbot.addWidget(w)
    monkeypatch.setattr(theme.sys, "platform", "linux")
    w.show()
    assert set_title_bar(w, DARK) is False


@pytest.mark.parametrize("mode", [ThemeMode.DARK, ThemeMode.LIGHT])
def test_graphs_paint_on_the_canvas_colour(
    qtbot: QtBot, qapp: QApplication, mode: ThemeMode
) -> None:
    from retro_ai_lab.ui.comparison_graph import ComparisonGraph
    from retro_ai_lab.ui.learning_graph import LearningGraph

    palette = apply_theme(qapp, mode)
    for graph in (LearningGraph(), ComparisonGraph()):
        qtbot.addWidget(graph)
        graph.resize(200, 120)
        image = graph.grab().toImage()
        assert image.pixelColor(2, 2).name() == palette.canvas
