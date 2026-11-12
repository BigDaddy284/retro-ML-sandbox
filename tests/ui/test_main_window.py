"""T-007: the application shell."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDockWidget, QLabel, QMainWindow, QToolButton
from pytestqt.qtbot import QtBot

from retro_ai_lab import __version__
from retro_ai_lab.ui.app import APP_NAME, create_app
from retro_ai_lab.ui.main_window import PANELS_MENU, RESET_LAYOUT, WINDOW_TITLE, MainWindow
from retro_ai_lab.ui.panels import PANELS, Panel, PanelSpec


def test_window_opens_with_title_and_closes(qtbot: QtBot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    assert w.isVisible()
    assert w.windowTitle() == WINDOW_TITLE == "Retro AI Lab"
    assert isinstance(w, QMainWindow)
    assert w.close()
    assert not w.isVisible()


def test_the_window_watches_the_link_and_the_gui_thread(qtbot: QtBot) -> None:
    """T-894: the stall watch logs the app's own work next to the emulator link's."""
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.watchdog.running and w.stall_watch.running
    w.close()
    assert not w.stall_watch.running  # it would otherwise log for a window that is gone


def test_all_six_panels_present_with_child_friendly_titles(qtbot: QtBot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    assert [p.key for p in w.panels] == [
        "game",
        "activity",
        "rewards",
        "progress",
        "learning",
        "compare",
    ]
    assert [p.title for p in w.panels] == [
        "Game",
        "What the AI is doing",
        "Rewards",
        "Progress",
        "Learning",
        "Compare AIs",
    ]
    for p in w.panels:
        assert p.isVisible()
        assert p.expanded
        assert p.toggle_button.text() == p.title
        assert w.panel(p.key) is p
        placeholder = p.content.findChild(QLabel, f"panel_{p.key}_placeholder")
        assert placeholder is not None and placeholder.text() == p.spec.blurb
        # T-837: every panel sits in a dock named after it.
        dock = w.dock(p.key)
        assert isinstance(dock, QDockWidget) and dock.widget() is p
        assert dock.objectName() == f"dock_{p.key}" and dock.windowTitle() == p.title
    assert w.centralWidget() is None


AREA = Qt.DockWidgetArea
DEFAULT_AREAS = {
    "game": AREA.LeftDockWidgetArea,
    "activity": AREA.RightDockWidgetArea,
    "rewards": AREA.RightDockWidgetArea,
    "progress": AREA.BottomDockWidgetArea,
    "learning": AREA.BottomDockWidgetArea,
    "compare": AREA.BottomDockWidgetArea,
}


def _layout(w: MainWindow) -> dict[str, tuple[bool, bool, Qt.DockWidgetArea]]:
    """(visible, floating, area) per panel key."""
    return {
        d.objectName()[5:]: (d.isVisible(), d.isFloating(), w.dockWidgetArea(d)) for d in w.docks
    }


def test_default_layout_shows_all_six_together(qtbot: QtBot) -> None:
    """T-837: DESIGN §10's "all visible together" is the default arrangement."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    assert _layout(w) == {key: (True, False, area) for key, area in DEFAULT_AREAS.items()}
    feature = QDockWidget.DockWidgetFeature
    for dock in w.docks:
        for wanted in (
            feature.DockWidgetMovable,
            feature.DockWidgetFloatable,
            feature.DockWidgetClosable,
        ):
            assert dock.features() & wanted
        assert dock.allowedAreas() == AREA.AllDockWidgetAreas
    assert w.isDockNestingEnabled()
    assert w.dockOptions() & QMainWindow.DockOption.AllowTabbedDocks
    assert w.layout_path is None  # tests forget the layout


def test_a_hidden_panel_comes_back_from_the_panels_menu(qtbot: QtBot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    menu = w.panels_menu
    assert menu.title() == PANELS_MENU
    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert [a.text() for a in actions] == [*(p.title for p in w.panels), RESET_LAYOUT]
    tick = actions[2]
    assert tick.isCheckable() and tick.isChecked()
    w.dock("rewards").close()
    assert not w.dock("rewards").isVisible() and not w.panel("rewards").isVisible()
    assert not tick.isChecked()
    assert all(w.dock(k).isVisible() for k in DEFAULT_AREAS if k != "rewards")
    tick.trigger()
    assert w.dock("rewards").isVisible() and w.panel("rewards").isVisible()
    button = w.findChild(QToolButton, "panels_button")
    assert button is not None and button.menu() is menu and button.text() == PANELS_MENU


def test_the_layout_is_remembered_between_runs(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "layout.bin"
    w = MainWindow(layout_path=path)
    qtbot.addWidget(w)
    w.show()
    assert w.layout_path == path and not path.exists()
    w.dock("compare").setFloating(True)
    w.dock("rewards").close()
    w.removeDockWidget(w.dock("learning"))
    w.addDockWidget(AREA.LeftDockWidgetArea, w.dock("learning"))
    w.dock("learning").show()
    assert w.close() and path.is_file()

    again = MainWindow(layout_path=path)
    qtbot.addWidget(again)
    again.show()
    layout = _layout(again)
    assert layout["compare"][1] and layout["compare"][0]
    assert not layout["rewards"][0]
    assert layout["learning"] == (True, False, AREA.LeftDockWidgetArea)
    assert layout["game"] == (True, False, AREA.LeftDockWidgetArea)
    again.close()


def test_reset_layout_puts_every_panel_back(qtbot: QtBot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.dock("compare").setFloating(True)
    w.dock("rewards").close()
    w.removeDockWidget(w.dock("game"))
    w.addDockWidget(AREA.BottomDockWidgetArea, w.dock("game"))
    w.dock("game").show()
    assert _layout(w) != {key: (True, False, area) for key, area in DEFAULT_AREAS.items()}
    reset = next(a for a in w.panels_menu.actions() if a.text() == RESET_LAYOUT)
    reset.trigger()
    assert _layout(w) == {key: (True, False, area) for key, area in DEFAULT_AREAS.items()}
    assert all(a.isChecked() for a in w.panels_menu.actions() if a.isCheckable())


def test_a_bad_layout_file_is_ignored(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "layout.bin"
    path.write_bytes(b"not a layout at all")
    w = MainWindow(layout_path=path)
    qtbot.addWidget(w)
    w.show()
    assert _layout(w) == {key: (True, False, area) for key, area in DEFAULT_AREAS.items()}
    missing = MainWindow(layout_path=tmp_path / "nowhere" / "layout.bin")
    qtbot.addWidget(missing)
    missing.show()
    assert _layout(missing) == {key: (True, False, area) for key, area in DEFAULT_AREAS.items()}
    assert missing.close() and (tmp_path / "nowhere" / "layout.bin").is_file()


def test_panel_specs_are_unique_and_plain() -> None:
    assert len({p.key for p in PANELS}) == len(PANELS) == 6
    for spec in PANELS:
        assert spec.key.isidentifier() and spec.key.islower()
        assert spec.title.strip() and spec.blurb.endswith(".")
        for jargon in ("hex", "0x", "memory address", "tensor"):
            assert jargon not in spec.blurb.lower()


def test_panel_collapse_and_expand_by_click(qtbot: QtBot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    panel = w.panel("rewards")
    seen: list[bool] = []
    panel.toggled.connect(seen.append)

    qtbot.mouseClick(panel.toggle_button, Qt.MouseButton.LeftButton)
    assert not panel.expanded
    assert not panel.content.isVisible()
    assert panel.toggle_button.arrowType() == Qt.ArrowType.RightArrow
    assert panel.isVisible()  # header stays

    qtbot.mouseClick(panel.toggle_button, Qt.MouseButton.LeftButton)
    assert panel.expanded
    assert panel.content.isVisible()
    assert panel.toggle_button.arrowType() == Qt.ArrowType.DownArrow
    assert seen == [False, True]


def test_panel_toggle_api_and_idempotent_set(qtbot: QtBot) -> None:
    panel = Panel(PanelSpec("x", "X", "Blurb."))
    qtbot.addWidget(panel)
    panel.show()
    seen: list[bool] = []
    panel.toggled.connect(seen.append)
    panel.set_expanded(True)  # already expanded: no signal
    panel.toggle()
    panel.toggle()
    panel.set_expanded(False)
    panel.set_expanded(False)
    assert seen == [False, True, False]
    assert not panel.expanded


def test_collapsing_one_panel_leaves_the_others(qtbot: QtBot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.panel("game").toggle()
    assert not w.panel("game").expanded
    assert all(w.panel(k).expanded for k in ("activity", "rewards", "progress", "learning"))


def test_set_body_replaces_placeholder(qtbot: QtBot) -> None:
    panel = Panel(PanelSpec("x", "X", "Blurb."))
    qtbot.addWidget(panel)
    body = QLabel("real content")
    panel.set_body(body)
    assert body.parent() is panel.content
    layout = panel.content.layout()
    assert layout is not None
    assert layout.count() == 1
    assert layout.itemAt(0).widget() is body


def test_status_bar_shows_ready_message_and_version(qtbot: QtBot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    assert "Connect to a game" in w.statusBar().currentMessage()
    w.set_status("Hello")
    assert w.statusBar().currentMessage() == "Hello"
    version = w.findChild(QLabel, "version_text")
    assert version is not None and version.text() == f"v{__version__}"


def test_create_app_is_a_singleton_and_sets_metadata(qapp: QApplication) -> None:
    assert create_app() is qapp
    assert create_app(["--anything"]) is qapp
    assert qapp.applicationName() == APP_NAME
    assert qapp.applicationVersion() == __version__


def _run_in_fresh_process(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter.

    ``QApplication.exec()`` must not run inside the test process: once it has
    returned, Qt refuses to run further event loops on that thread, which
    silently breaks every timer-driven test that follows.
    """
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_run_app_shows_window_and_exits_zero() -> None:
    code = """
from retro_ai_lab.ui.app import APP_NAME, run_app
from retro_ai_lab.ui.main_window import WINDOW_TITLE

def hook(app, window):
    assert window.isVisible()
    assert window.windowTitle() == WINDOW_TITLE
    assert app.applicationName() == APP_NAME
    window.close()
    app.quit()

raise SystemExit(run_app(before_exec=hook))
"""
    result = _run_in_fresh_process(code)
    assert result.returncode == 0, result.stderr


def test_run_app_exit_after_ms() -> None:
    result = _run_in_fresh_process(
        "from retro_ai_lab.ui.app import run_app; raise SystemExit(run_app(exit_after_ms=10))"
    )
    assert result.returncode == 0, result.stderr


def test_real_entry_point_launches_headless_and_exits_zero(tmp_path: Path) -> None:
    """The launch smoke test: the real CLI, Qt and window in a fresh process."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "RETRO_AI_LAB_DATA_DIR": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-m", "retro_ai_lab", "--exit-after-ms", "200"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # T-815: the run leaves a log naming the startup milestones.
    text = (tmp_path / "logs" / "retro-ai-lab.log").read_text(encoding="utf-8")
    for line in ("starting", "Qt application created", "Main window shown", "Event loop finished"):
        assert line in text
    # T-995: each stage is named before it begins, so a start that stops says where.
    stages = (
        "starting",
        "Loading Qt",
        "Loading the app's own code",
        "Starting the app",
        "Creating the Qt application",
        "Qt application created",
        "Building the main window",
        "Main window shown",
        "Event loop finished",
    )
    places = [text.index(stage) for stage in stages]
    assert places == sorted(places)
    # It started in time and nothing crashed: no trace is left behind, or spoken of.
    assert list((tmp_path / "logs").glob("start-*.txt")) == []
    assert "left a trace" not in text


# ------------------------------------------------- T-881: one title per panel


def test_each_dock_has_one_header(qtbot: QtBot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    for p in w.panels:
        header = w.dock(p.key).titleBarWidget()
        assert header is p.header and header.objectName() == f"panel_{p.key}_header"
        for button in (p.toggle_button, p.help_button, p.pop_out_button, p.hide_button):
            assert header.isAncestorOf(button) and button.isVisible()
        # Nothing in the panel itself repeats the title.
        assert not p.isAncestorOf(p.toggle_button)
        assert p.toggle_button.text() == p.title == w.dock(p.key).windowTitle()


def test_pop_out_and_back(qtbot: QtBot) -> None:
    from retro_ai_lab.ui.panels import DOCK_BACK, POP_OUT

    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    dock, panel = w.dock("learning"), w.panel("learning")
    assert not dock.isFloating() and panel.pop_out_button.toolTip() == POP_OUT
    panel.pop_out_button.click()
    assert dock.isFloating() and panel.pop_out_button.toolTip() == DOCK_BACK
    panel.pop_out_button.click()
    assert not dock.isFloating() and panel.pop_out_button.toolTip() == POP_OUT
    # Floated some other way (a drag, a saved layout), the button still follows.
    dock.setFloating(True)
    assert panel.pop_out_button.toolTip() == DOCK_BACK
    w.reset_layout()
    assert panel.pop_out_button.toolTip() == POP_OUT


def test_hide_button_hides_and_the_menu_restores(qtbot: QtBot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.panel("compare").hide_button.click()
    assert not w.dock("compare").isVisible()
    tick = next(a for a in w.panels_menu.actions() if a.text() == w.panel("compare").title)
    assert not tick.isChecked()
    tick.trigger()
    assert w.dock("compare").isVisible() and w.panel("compare").header.isVisible()


def test_a_panel_without_a_dock_keeps_its_header(qtbot: QtBot) -> None:
    panel = Panel(PanelSpec("x", "X", "Blurb."))
    qtbot.addWidget(panel)
    panel.show()
    assert panel.isAncestorOf(panel.header) and panel.toggle_button.isVisible()
    assert panel.help_button.isVisible()
    # The dock-only buttons stay out of the way until a dock wires them.
    assert not panel.pop_out_button.isVisible() and not panel.hide_button.isVisible()


# ------------------------------------------- T-899: panels smaller than their content


def test_a_panel_can_be_dragged_smaller_than_what_it_holds(qtbot: QtBot) -> None:
    """The user could not shrink Learning: its chart, text and table set a floor."""
    from retro_ai_lab.ui.panels import MIN_CONTENT_HEIGHT

    panel = Panel(PanelSpec("x", "X", "Blurb."))
    qtbot.addWidget(panel)
    tall = QLabel("tall content")
    tall.setMinimumHeight(600)
    panel.set_body(tall)
    panel.show()
    assert panel.minimumSizeHint().height() < 200  # the content no longer sets the floor
    panel.resize(300, 150)
    qtbot.waitUntil(lambda: panel.height() == 150)
    assert tall.height() >= 600  # it scrolls rather than squashing
    assert panel.scroll_area.minimumHeight() == MIN_CONTENT_HEIGHT
    assert panel.scroll_area.isAncestorOf(tall) and panel.scroll_area.widgetResizable()
    # Collapsing hides the scroll area, header stays.
    panel.set_expanded(False)
    assert not panel.scroll_area.isVisible() and panel.header.isVisible()


def test_the_window_is_not_pinned_tall_by_its_panels(qtbot: QtBot) -> None:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    for p in w.panels:
        assert w.dock(p.key).minimumSizeHint().height() < 200, p.key
    assert w.minimumSizeHint().height() < 600
