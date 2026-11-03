"""Several learning curves on one graph, one colour per saved AI."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from retro_ai_lab.training.curve import LearningCurve
from retro_ai_lab.ui.learning_graph import (
    BACKGROUND,
    INK,
    MARGIN,
    MIN_HEIGHT,
    ZERO,
    PlotLayout,
    plot_points,
    value_range,
)

EMPTY_CAPTION = "Save two AIs and their learning curves will appear here together."
PALETTE: tuple[QColor, ...] = (
    QColor("#1e6fd9"),
    QColor("#d2691e"),
    QColor("#2e8b57"),
    QColor("#8e44ad"),
    QColor("#c2185b"),
    QColor("#00838f"),
)


def colour_for(index: int) -> QColor:
    return PALETTE[index % len(PALETTE)]


def shared_bounds(curves: Sequence[LearningCurve]) -> tuple[float, float] | None:
    """One value range covering every curve that has points."""
    ranges = [r for r in (value_range(c) for c in curves) if r is not None]
    if not ranges:
        return None
    return min(low for low, _ in ranges), max(high for _, high in ranges)


class ComparisonGraph(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("comparison_graph")
        self._curves: tuple[tuple[str, LearningCurve], ...] = ()
        self.setMinimumHeight(MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

    @property
    def curves(self) -> tuple[tuple[str, LearningCurve], ...]:
        return self._curves

    @property
    def caption(self) -> str:
        drawn = [name for name, c in self._curves if c.attempts]
        if not drawn:
            return EMPTY_CAPTION
        return "Lines: average of recent goes.  " + "  ".join(
            f"{name}: {colour_for(i).name()}" for i, name in enumerate(drawn)
        )

    def set_curves(self, curves: Sequence[tuple[str, LearningCurve]]) -> None:
        self._curves = tuple(curves)
        self.setToolTip(self.caption)
        self.update()

    def layouts(self) -> tuple[tuple[str, PlotLayout], ...]:
        """One layout per curve with points, all on the same scale."""
        text_band = self.fontMetrics().height() * 2 + 6
        height = max(1, self.height() - text_band)
        bounds = shared_bounds([c for _, c in self._curves])
        return tuple(
            (name, plot_points(curve, self.width(), height, bounds=bounds))
            for name, curve in self._curves
            if curve.attempts
        )

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), BACKGROUND)
        band = self.fontMetrics().height() * 2 + 6
        layouts = self.layouts()
        painter.setPen(INK)
        painter.drawText(
            QRectF(MARGIN, self.height() - band, self.width() - 2 * MARGIN, band),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            self._legend(layouts),
        )
        zero = next((p.zero_y for _, p in layouts if p.zero_y is not None), None)
        if zero is not None:
            painter.setPen(QPen(ZERO, 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(MARGIN, zero), QPointF(self.width() - MARGIN, zero))
        for i, (_, plot) in enumerate(layouts):
            colour = colour_for(i)
            painter.setPen(QPen(colour, 2))
            for (x0, y0), (x1, y1) in zip(plot.line, plot.line[1:], strict=False):
                painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))
            painter.setBrush(colour)
            painter.setPen(QPen(colour, 1))
            for x, y in plot.dots:
                painter.drawEllipse(QPointF(x, y), 2.5, 2.5)
        painter.end()

    def _legend(self, layouts: Sequence[tuple[str, PlotLayout]]) -> str:
        if not layouts:
            return EMPTY_CAPTION
        return "Lines: average of recent goes.  " + "  ".join(f"■ {name}" for name, _ in layouts)


__all__ = ["EMPTY_CAPTION", "PALETTE", "ComparisonGraph", "colour_for", "shared_bounds"]
