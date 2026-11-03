"""T-804: the Compare AIs panel and the shared-scale graph."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from retro_ai_lab.training import EpisodeRecord, EpisodeResult, LearningCurve, Mode
from retro_ai_lab.ui.compare_panel import (
    COLUMNS,
    IDLE_TEXT,
    LEADER_MARK,
    NO_SAVED_TEXT,
    ComparePanel,
)
from retro_ai_lab.ui.comparison_graph import EMPTY_CAPTION, ComparisonGraph, shared_bounds
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.learning_graph import plot_points
from retro_ai_lab.ui.main_window import MainWindow
from tests.ui.test_learning_panel import connect


def go(n: int, reward: float) -> EpisodeRecord:
    return EpisodeRecord(n, Mode.GUIDED, EpisodeResult(10, 40, reward, "max_steps", 3), n * 40)


def curve(*rewards: float) -> LearningCurve:
    return LearningCurve([go(i + 1, r) for i, r in enumerate(rewards)], window=2)


@pytest.fixture
def context(tmp_path: Path) -> AppContext:
    return AppContext(data_dir=tmp_path)


@pytest.fixture
def panel(qtbot: QtBot, context: AppContext) -> ComparePanel:
    p = ComparePanel(context)
    qtbot.addWidget(p)
    p.show()
    return p


def widget(panel: ComparePanel, name: str) -> object:
    return panel.findChild(object, name)


def test_shared_bounds_put_the_same_value_at_the_same_height() -> None:
    a, b = curve(1, 5), curve(-3, 2)
    assert shared_bounds([a, b]) == (-3.0, 5.0)
    assert shared_bounds([LearningCurve([])]) is None and shared_bounds([]) is None
    bounds = shared_bounds([a, b])
    la = plot_points(a, 200, 100, bounds=bounds)
    lb = plot_points(b, 200, 100, bounds=bounds)
    # a's own scale would put its lowest dot at the bottom; shared, it sits higher
    own = plot_points(a, 200, 100)
    assert la.dots[0][1] < own.dots[0][1]
    # 5 is the top of the shared range for both, so a's second dot is at the top margin
    assert la.dots[1][1] == pytest.approx(12.0)
    assert lb.dots[0][1] == pytest.approx(88.0)  # -3 at the bottom margin
    assert la.zero_y == lb.zero_y


def test_graph_draws_each_ai_in_its_own_colour(qtbot: QtBot) -> None:
    g = ComparisonGraph()
    qtbot.addWidget(g)
    g.resize(240, 180)
    g.show()
    assert g.caption == EMPTY_CAPTION and g.layouts() == ()
    blank = g.grab().toImage()
    g.set_curves([("AI A", curve(4, 6, 8)), ("AI B", curve(-2, 2, 0)), ("Empty", curve())])
    assert [name for name, _ in g.layouts()] == ["AI A", "AI B"]
    assert "AI A" in g.caption and "AI B" in g.caption and "Empty" not in g.caption
    drawn = g.grab().toImage()
    assert drawn != blank
    (_, la), (_, lb) = g.layouts()
    x, y = la.dots[1]
    assert drawn.pixelColor(int(x), int(y)).name() == "#1e6fd9"
    x, y = lb.dots[1]
    assert drawn.pixelColor(int(x), int(y)).name() == "#d2691e"


def test_idle_panel(panel: ComparePanel) -> None:
    assert panel.comparison_text == IDLE_TEXT and panel.table_rows() == ()
    assert panel.saved_names() == () and panel.selected_name is None
    for name in ("ai_name_edit", "save_ai_button", "load_ai_button", "delete_ai_button"):
        assert not widget(panel, name).isEnabled(), name  # type: ignore[attr-defined]
    assert not panel.save_now("AI A") and "Connect to a game" in panel.note_text
    assert not panel.load_selected() and not panel.delete_selected()


def test_save_compare_load_and_delete(
    qtbot: QtBot, panel: ComparePanel, context: AppContext
) -> None:
    connect(context)
    assert panel.comparison_text == NO_SAVED_TEXT
    assert widget(panel, "save_ai_button").isEnabled()  # type: ignore[attr-defined]
    assert not widget(panel, "load_ai_button").isEnabled()  # type: ignore[attr-defined]
    assert not panel.save_now("   ") and panel.note_text == "Give the AI a name."
    t = context.training
    t.train(2)
    panel.set_name("AI A")
    assert panel.save_now() and panel.note_text == "Saved as AI A." and panel.name_text == ""
    assert panel.saved_names() == ("AI A",) and panel.selected_name == "AI A"
    rows = panel.table_rows()
    assert len(rows) == 1 and len(rows[0]) == len(COLUMNS)
    assert rows[0][0] == "AI A" and rows[0][COLUMNS.index("Goes")] == "2"
    assert LEADER_MARK not in "".join(rows[0])  # no leader with one AI
    assert panel.comparison_text == "Only AI A is saved so far. Save another AI to compare them."
    assert widget(panel, "load_ai_button").isEnabled()  # type: ignore[attr-defined]

    t.forget()
    profile = context.rewards.profile
    assert profile is not None
    for behaviour in profile.ids():
        context.rewards.set_value(behaviour, -1.0)
    t.train(3)
    assert panel.save_now("AI B")
    assert panel.saved_names() == ("AI A", "AI B")
    rows = panel.table_rows()
    assert [r[0] for r in rows] == ["AI A", "AI B"]
    goes = COLUMNS.index("Goes")
    assert rows[1][goes] == LEADER_MARK + "3"  # AI B practised more
    assert any(LEADER_MARK in cell for cell in rows[0] + rows[1])
    assert "AI A" in panel.comparison_text and "against" in panel.comparison_text
    assert [name for name, _ in panel.graph.layouts()] == ["AI A", "AI B"]

    # loading AI A brings its two goes back
    panel.select("AI A")
    assert panel.load_selected()
    assert panel.note_text.startswith("Loaded AI A")
    assert t.episodes == 2
    # not while training runs
    t.start()
    assert not panel.load_selected()
    assert panel.note_text == "Stop training before loading another AI."
    t.stop()
    panel.select("AI B")
    assert panel.delete_selected() and panel.note_text == "Deleted AI B."
    assert panel.saved_names() == ("AI A",) and [r[0] for r in panel.table_rows()] == ["AI A"]
    assert [name for name, _ in panel.graph.layouts()] == ["AI A"]
    panel.select(None)
    assert not widget(panel, "load_ai_button").isEnabled()  # type: ignore[attr-defined]
    assert not panel.load_selected() and panel.note_text == "Pick a saved AI to load."
    with pytest.raises(ValueError):
        panel.select("Nope")
    context.connection.disconnect()
    assert panel.comparison_text == IDLE_TEXT and panel.table_rows() == ()
    assert not widget(panel, "save_ai_button").isEnabled()  # type: ignore[attr-defined]


def test_main_window_has_the_panel_and_its_tip(qtbot: QtBot, context: AppContext) -> None:
    w = MainWindow(context)
    qtbot.addWidget(w)
    w.show()
    assert isinstance(w.compare_panel, ComparePanel)
    assert w.compare_panel.parent() is w.panel("compare").content
    w.explain("compare")
    assert w.tip_dialog.current_key == "panel.compare"


def test_share_and_load_a_shared_ai(
    qtbot: QtBot, panel: ComparePanel, context: AppContext, tmp_path: Path
) -> None:
    assert not widget(panel, "import_ai_button").isEnabled()  # type: ignore[attr-defined]
    assert not panel.export_selected_to(tmp_path / "x.json")
    assert panel.note_text == "Pick a saved AI to share."
    assert not panel.import_from(tmp_path / "x.json") and "Connect to a game" in panel.note_text
    connect(context)
    assert widget(panel, "import_ai_button").isEnabled()  # type: ignore[attr-defined]
    assert not widget(panel, "share_ai_button").isEnabled()  # type: ignore[attr-defined]
    context.training.train(1)
    assert panel.save_now("AI A")
    assert widget(panel, "share_ai_button").isEnabled()  # type: ignore[attr-defined]
    shared = tmp_path / "ai-a.json"
    assert panel.export_selected_to(shared) and panel.note_text == "Shared AI A as ai-a.json."
    assert panel.import_from(shared) and panel.note_text == "Loaded the shared AI as AI A (2)."
    assert panel.saved_names() == ("AI A", "AI A (2)") and panel.selected_name == "AI A (2)"
    assert "AI A (2)" in panel.comparison_text or "about as well" in panel.comparison_text
    assert not panel.import_from(tmp_path / "missing.json")
    assert panel.note_text.startswith("Can't read the shared AI")
    assert not panel.export_selected_to(tmp_path / "nope" / "deep" / "x.json")
    assert panel.note_text.startswith("Can't write the shared AI")
