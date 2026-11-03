"""A widget that shows one emulator :class:`Frame`, scaled like pixel art."""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from retro_ai_lab.core import ConfigurationError, Frame

BACKGROUND = QColor(24, 24, 32)
PLACEHOLDER_TEXT = "No game yet"


def frame_to_image(frame: Frame) -> QImage:
    """Convert a :class:`Frame` to a ``QImage`` (deep copy, safe to keep)."""
    if not isinstance(frame, Frame):
        raise ConfigurationError("frame_to_image needs a Frame.")
    if frame.channels == 1:
        fmt = QImage.Format.Format_Grayscale8
    elif frame.channels == 3:
        fmt = QImage.Format.Format_RGB888
    else:
        fmt = QImage.Format.Format_RGBA8888
    stride = frame.width * frame.channels
    image = QImage(frame.pixels, frame.width, frame.height, stride, fmt)
    return image.copy()  # detach from the bytes buffer


class FrameView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("frame_view")
        self._frame: Frame | None = None
        self._image: QImage | None = None
        self._placeholder = PLACEHOLDER_TEXT
        self.setMinimumSize(128, 96)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

    # --------------------------------------------------------------- API

    @property
    def frame(self) -> Frame | None:
        return self._frame

    @property
    def placeholder(self) -> str:
        return self._placeholder

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        if self._frame is None:
            self.update()

    def set_frame(self, frame: Frame | None) -> None:
        if frame is not None and not isinstance(frame, Frame):
            raise ConfigurationError("FrameView.set_frame needs a Frame or None.")
        self._frame = frame
        self._image = frame_to_image(frame) if frame is not None else None
        self.update()

    def clear(self) -> None:
        self.set_frame(None)

    def image_rect(self) -> QRect:
        """Where the frame is drawn inside the widget (aspect ratio kept)."""
        if self._image is None:
            return QRect()
        w, h = self.width(), self.height()
        iw, ih = self._image.width(), self._image.height()
        scale = min(w / iw, h / ih)
        # Integer scale when we can, for crisp pixels; otherwise fit.
        if scale >= 1:
            scale = float(int(scale))
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        return QRect((w - dw) // 2, (h - dh) // 2, dw, dh)

    # ------------------------------------------------------------ painting

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        if self._image is None:
            painter.setPen(QColor(200, 200, 210))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
        else:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawImage(self.image_rect(), self._image)
        painter.end()


__all__ = ["BACKGROUND", "PLACEHOLDER_TEXT", "FrameView", "frame_to_image"]
