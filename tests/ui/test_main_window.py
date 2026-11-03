"""T-007: the application shell."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QSplitter
from pytestqt.qtbot import QtBot

from retro_ai_lab import __version__
from retro_ai_lab.ui.app import APP_NAME, create_app
from retro_ai_lab.ui.main_window import WINDOW_TITLE, MainWindow
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
    assert isinstance(w.centralWidget(), QSplitter)


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
