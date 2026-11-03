"""T-602: the learning graph."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from retro_ai_lab.training import EpisodeRecord, EpisodeResult, LearningCurve, Mode
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.learning_graph import EMPTY_CAPTION, LearningGraph, plot_points
from retro_ai_lab.ui.learning_panel import LearningPanel
from tests.ui.test_learning_panel import connect


def go(n: int, reward: float) -> EpisodeRecord:
    return EpisodeRecord(n, Mode.GUIDED, EpisodeResult(10, 40, reward, "max_steps", 2), n * 40)


def curve(*rewards: float, window: int = 2) -> LearningCurve:
    return LearningCurve([go(i + 1, r) for i, r in enumerate(rewards)], window)


def test_geometry_by_hand() -> None:
    # Rewards 4, -4, 8 with window 2: averages 4, 0, 2. Values span -4..8.
    plot = plot_points(curve(4, -4, 8), width=120, height=70, margin=10)
    xs = [x for x, _ in plot.dots]
    assert xs == [10.0, 60.0, 110.0]  # evenly spaced across 100 px
    ys = [y for _, y in plot.dots]
    assert ys[2] == 10.0  # the highest reward sits on the top margin
    assert ys[1] == 60.0  # the lowest on the bottom margin
    assert ys[0] == pytest.approx(10 + 50 * (8 - 4) / 12)
    assert [y for _, y in plot.line] == pytest.approx([ys[0], 10 + 50 * 8 / 12, 10 + 50 * 6 / 12])
    assert plot.zero_y == pytest.approx(10 + 50 * 8 / 12)
    assert plot.best == 2 and not plot.is_empty


def test_geometry_edge_cases() -> None:
    assert plot_points(LearningCurve([]), 100, 50).is_empty
    assert plot_points(LearningCurve([]), 100, 50).best is None
    one = plot_points(curve(3), width=100, height=50, margin=10)
    assert one.dots == ((50.0, 25.0),) and one.best == 0  # centred: flat curves sit mid-way
    assert one.zero_y is None  # 3 +- 1 doesn't straddle zero
    positive = plot_points(curve(2, 6), 100, 50, 10)
    assert positive.zero_y is None
    negative = plot_points(curve(-2, -6), 100, 50, 10)
    assert negative.zero_y is None and negative.best == 0
    straddling = plot_points(curve(-2, 2), 100, 50, 10)
    assert straddling.zero_y == 25.0
    tiny = plot_points(curve(1, 2), width=5, height=5, margin=10)  # smaller than its margins
    assert len(tiny.dots) == 2  # never divides by zero


def test_widget_renders_and_captions(qtbot: QtBot) -> None:
    g = LearningGraph()
    qtbot.addWidget(g)
    g.resize(240, 160)
    g.show()
    assert g.caption == EMPTY_CAPTION and g.curve.attempts == 0
    assert g.layout_points().is_empty and g.minimumHeight() == 120
    blank = g.grab().toImage()
    g.set_curve(curve(4, -4, 8))
    assert g.caption == g.curve.describe() and g.toolTip() == g.caption
    assert not g.layout_points().is_empty
    drawn = g.grab().toImage()
    assert drawn != blank
    # The middle dot is drawn in the "loss" colour, the others in the "gain" colour.
    plot = g.layout_points()
    x, y = plot.dots[1]
    assert drawn.pixelColor(int(x), int(y)).name() == "#d2691e"
    x, y = plot.dots[2]
    assert drawn.pixelColor(int(x), int(y)).name() == "#2e8b57"


def test_panel_shows_and_updates_the_graph(qtbot: QtBot, tmp_path: Path) -> None:
    context = AppContext(data_dir=tmp_path)
    panel = LearningPanel(context)
    qtbot.addWidget(panel)
    panel.show()
    assert panel.graph.curve.attempts == 0 and panel.graph.caption == EMPTY_CAPTION
    connect(context)
    context.training.train(2)
    assert panel.graph.curve.attempts == 2
    assert panel.graph.caption.startswith("2 goes so far")
    context.connection.disconnect()
    assert panel.graph.curve.attempts == 2  # the picture stays for review
    connect(context)
    assert panel.graph.curve.attempts == 2  # loaded back with the brain
