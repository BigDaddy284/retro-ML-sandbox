"""T-882: the app-wide look."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QAbstractItemView, QApplication
from pytestqt.qtbot import QtBot

from retro_ai_lab.ui.app import create_app
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.main_window import MainWindow
from retro_ai_lab.ui.panels import PANELS
from retro_ai_lab.ui.theme import (
    PANEL_COLOURS,
    STYLE_PROPERTY,
    THEME_MARKER,
    PanelColours,
    apply_theme,
    polish_views,
    stylesheet,
)


def test_every_panel_has_colours_and_the_stylesheet_uses_them() -> None:
    assert set(PANEL_COLOURS) == {spec.key for spec in PANELS}
    sheet = stylesheet()
    for key, colours in PANEL_COLOURS.items():
        assert isinstance(colours, PanelColours)
        assert f'[panel="{key}"]' in sheet
        for colour in (colours.accent, colours.tint, colours.ink):
            assert colour.startswith("#") and len(colour) == 7 and colour in sheet
    # Distinct accents, so the panels can be told apart at a glance.
    assert len({c.accent for c in PANEL_COLOURS.values()}) == len(PANEL_COLOURS)
    assert sheet.startswith(THEME_MARKER)


def test_apply_theme_twice_keeps_one_copy(qapp: QApplication) -> None:
    before = qapp.styleSheet()
    try:
        qapp.setStyleSheet("QLabel#mine { color: red; }")
        apply_theme(qapp)
        apply_theme(qapp)
        sheet = qapp.styleSheet()
        assert sheet.count(THEME_MARKER) == 1
        assert "QLabel#mine { color: red; }" in sheet  # someone else's rules stay
        assert str(qapp.property(STYLE_PROPERTY)).lower() == "fusion"
        assert qapp.font().pointSizeF() >= 10
    finally:
        qapp.setStyleSheet(before)


def test_create_app_applies_the_theme(qapp: QApplication) -> None:
    assert create_app() is qapp
    assert THEME_MARKER in qapp.styleSheet()
    assert qapp.styleSheet().count(THEME_MARKER) == 1


def test_the_stylesheet_parses_on_every_window(
    qtbot: QtBot,
    qapp: QApplication,
    tmp_path: Path,
    qtlog,  # type: ignore[no-untyped-def]
) -> None:
    apply_theme(qapp)
    w = MainWindow(AppContext(data_dir=tmp_path))
    qtbot.addWidget(w)
    w.show()
    w.open_research()
    for tab in range(w.research_dialog.tabs.count()):
        w.research_dialog.tabs.setCurrentIndex(tab)
        qapp.processEvents()
    w.grab()
    w.research_dialog.grab()
    bad = [r.message for r in qtlog.records if "style" in r.message.lower()]
    assert bad == []
    w.research_dialog.close()


def test_tables_alternate_rows(qtbot: QtBot, tmp_path: Path) -> None:
    w = MainWindow(AppContext(data_dir=tmp_path))
    qtbot.addWidget(w)
    views = [*w.findChildren(QAbstractItemView), *w.research_dialog.findChildren(QAbstractItemView)]
    trees = [v for v in views if v.objectName()]  # combo-box pop-ups have no name
    assert len(trees) >= 8
    assert all(v.alternatingRowColors() for v in trees)
    # polish_views is safe to run again on a widget it already did.
    polish_views(w)
    assert all(v.alternatingRowColors() for v in trees)
