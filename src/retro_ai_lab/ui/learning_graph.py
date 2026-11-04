"""A small graph of how the AI's goes are going (DESIGN.md §10, §16 Phase 6).

One dot per go for the points it scored, a line for the rolling average and
a ring round the best go. The geometry lives in :func:`plot_points` so it
can be checked without drawing anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from retro_ai_lab.training.curve import LearningCurve
from retro_ai_lab.ui import theme

EMPTY_CAPTION = "No goes yet. Press Start and the graph will grow here."
LEGEND = "Dots: points each go.  Line: average of recent goes.  Ring: the best go."
MARGIN = 12
MIN_HEIGHT = 120


def background() -> QColor:
    """The graph's ground: the theme's Canvas colour (T-886)."""
    return QColor(theme.current().canvas)


def ink() -> QColor:
    return QColor(theme.current().text)


def zero_line() -> QColor:
    return QColor(theme.current().muted)


GAIN = QColor("#2e8b57")
LOSS = QColor("#d2691e")
AVERAGE = QColor("#1e6fd9")


@dataclass(frozen=True, slots=True)
class PlotLayout:
    dots: tuple[tuple[float, float], ...]  # (x, y) per go, in pixels
    line: tuple[tuple[float, float], ...]  # the rolling average, same x
    zero_y: float | None  # where zero points is drawn, if it is inside the range
    best: int | None  # index into ``dots`` of the best go

    @property
    def is_empty(self) -> bool:
        return not self.dots


def value_range(curve: LearningCurve) -> tuple[float, float] | None:
    """The lowest and highest value the curve draws (dots and average), or None."""
    points = curve.points
    if not points:
        return None
    values = [p.reward for p in points] + [p.average for p in points]
    return min(values), max(values)


def plot_points(
    curve: LearningCurve,
    width: float,
    height: float,
    margin: float = MARGIN,
    *,
    bounds: tuple[float, float] | None = None,
) -> PlotLayout:
    """Map the curve onto a ``width`` x ``height`` box, keeping ``margin`` clear.

    ``bounds`` fixes the value range (so several curves share one scale);
    without it the curve's own range is used.
    """
    points = curve.points
    if not points:
        return PlotLayout((), (), None, None)
    inner_w = max(1.0, width - 2 * margin)
    inner_h = max(1.0, height - 2 * margin)
    own = value_range(curve)
    assert own is not None
    low, high = bounds if bounds is not None else own
    if high == low:
        high, low = high + 1.0, low - 1.0  # a flat curve sits in the middle
    span = high - low

    def x_of(i: int) -> float:
        if len(points) == 1:
            return margin + inner_w / 2
        return margin + inner_w * i / (len(points) - 1)

    def y_of(value: float) -> float:
        return margin + inner_h * (high - value) / span

    dots = tuple((x_of(i), y_of(p.reward)) for i, p in enumerate(points))
    line = tuple((x_of(i), y_of(p.average)) for i, p in enumerate(points))
    zero_y = y_of(0.0) if low <= 0.0 <= high else None
    best = curve.best
    best_index = None
    if best is not None:
        best_index = next(i for i, p in enumerate(points) if p.number == best.number)
    return PlotLayout(dots, line, zero_y, best_index)


class LearningGraph(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("learning_graph")
        self._curve = LearningCurve([])
        self.setMinimumHeight(MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

    @property
    def curve(self) -> LearningCurve:
        return self._curve

    @property
    def caption(self) -> str:
        return self._curve.describe() if self._curve.attempts else EMPTY_CAPTION

    def set_curve(self, curve: LearningCurve) -> None:
        self._curve = curve
        self.setToolTip(self.caption)
        self.update()

    def layout_points(self) -> PlotLayout:
        """The geometry for the current size (what :meth:`paintEvent` draws)."""
        text_band = self.fontMetrics().height() * 2 + 6
        return plot_points(self._curve, self.width(), max(1, self.height() - text_band))

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), background())
        metrics = self.fontMetrics()
        band = metrics.height() * 2 + 6
        plot = self.layout_points()
        painter.setPen(ink())
        caption_rect = QRectF(MARGIN, self.height() - band, self.width() - 2 * MARGIN, band)
        painter.drawText(
            caption_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            self.caption if plot.is_empty else LEGEND,
        )
        if plot.is_empty:
            painter.end()
            return
        if plot.zero_y is not None:
            painter.setPen(QPen(zero_line(), 1, Qt.PenStyle.DashLine))
            painter.drawLine(
                QPointF(MARGIN, plot.zero_y), QPointF(self.width() - MARGIN, plot.zero_y)
            )
        if len(plot.line) > 1:
            painter.setPen(QPen(AVERAGE, 2))
            for (x0, y0), (x1, y1) in zip(plot.line, plot.line[1:], strict=False):
                painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))
        for i, ((x, y), point) in enumerate(zip(plot.dots, self._curve.points, strict=True)):
            colour = GAIN if point.reward >= 0 else LOSS
            painter.setPen(QPen(colour, 1))
            painter.setBrush(colour)
            painter.drawEllipse(QPointF(x, y), 3.5, 3.5)
            if i == plot.best:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(ink(), 1.5))
                painter.drawEllipse(QPointF(x, y), 7.0, 7.0)
        painter.end()


__all__ = ["EMPTY_CAPTION", "LEGEND", "LearningGraph", "PlotLayout", "plot_points", "value_range"]
