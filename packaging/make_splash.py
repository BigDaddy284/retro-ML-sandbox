"""Draw packaging/splash.png, the picture of the built app's "starting" window (T-1002).

Usage: python packaging/make_splash.py [output.png]

The build tool's bootloader shows this picture before Python is loaded and
writes the step the start is on under it (see ``TEXT_POS`` in the spec, which
must stay inside the picture and clear of what is drawn here). The picture is
kept in the repository, so a build needs neither this script nor its fonts;
run it again only to change the picture. The colours are the app's dark theme.
Pure magenta must not appear: on Windows the bootloader treats it as see-through.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WIDTH, HEIGHT = 520, 200
TITLE = "Retro AI Lab"
NOTE = "Starting. The first start after an update can take a minute."


def draw(path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import (
        QColor,
        QFont,
        QGuiApplication,
        QImage,
        QLinearGradient,
        QPainter,
        QPen,
    )

    app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
    image = QImage(WIDTH, HEIGHT, QImage.Format.Format_RGB32)
    image.fill(QColor(10, 13, 26))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    bar = QLinearGradient(QPointF(0, 0), QPointF(WIDTH, 0))
    bar.setColorAt(0.0, QColor(38, 104, 255))
    bar.setColorAt(1.0, QColor(176, 72, 255))
    painter.fillRect(QRectF(0, 0, WIDTH, 6), bar)
    painter.setPen(QPen(QColor(46, 56, 92), 1))
    painter.drawRect(QRectF(0.5, 0.5, WIDTH - 1, HEIGHT - 1))

    title = QFont("Inter")
    title.setPixelSize(30)
    title.setWeight(QFont.Weight.DemiBold)
    painter.setFont(title)
    painter.setPen(QColor(234, 238, 250))
    painter.drawText(QRectF(28, 34, WIDTH - 56, 44), Qt.AlignmentFlag.AlignVCenter, TITLE)

    note = QFont("Inter")
    note.setPixelSize(15)
    painter.setFont(note)
    painter.setPen(QColor(172, 182, 210))
    painter.drawText(QRectF(28, 88, WIDTH - 56, 26), Qt.AlignmentFlag.AlignVCenter, NOTE)

    # A quiet line over the place the bootloader writes the step in.
    painter.setPen(QPen(QColor(30, 37, 62), 1))
    painter.drawLine(QPointF(28, 132.5), QPointF(WIDTH - 28, 132.5))
    painter.end()
    if not image.save(str(path), "PNG"):
        raise SystemExit(f"could not write {path}")
    del app


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("splash.png")
    draw(target)
    print(f"wrote {target}")
