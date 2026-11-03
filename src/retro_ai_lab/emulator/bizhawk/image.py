"""Screenshots as :class:`Frame` objects.

EmuHawk writes PNG files; QtGui's ``QImage`` (already a dependency of the
app) decodes them in well under a millisecond and copes with every variant
a .NET encoder might produce. Only QtGui is used here: nothing from the UI
layer, and no application object is needed.
"""

from __future__ import annotations

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QImage, QImageWriter

from retro_ai_lab.core import ConfigurationError, Frame, ProtocolError

CHANNELS = 3  # frames come out as packed RGB
_FORMATS = {
    1: QImage.Format.Format_Grayscale8,
    3: QImage.Format.Format_RGB888,
    4: QImage.Format.Format_RGBA8888,
}


def decode_image(data: bytes, *, number: int = 0) -> Frame:
    """A PNG (or any image Qt reads) as an RGB frame numbered ``number``."""
    image = QImage.fromData(data)
    if image.isNull():
        raise ProtocolError("The emulator's screenshot could not be read as an image.")
    rgb = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = rgb.width(), rgb.height()
    stride = rgb.bytesPerLine()
    row_bytes = width * CHANNELS
    raw = bytes(rgb.constBits())
    if stride == row_bytes:
        pixels = raw[: row_bytes * height]
    else:  # Qt pads rows to 4 bytes; repack them
        pixels = b"".join(raw[y * stride : y * stride + row_bytes] for y in range(height))
    return Frame(width, height, CHANNELS, pixels, number=number)


def encode_png(frame: Frame) -> bytes:
    """A frame as PNG bytes (used by the simulator and the test host)."""
    fmt = _FORMATS[frame.channels]  # Frame allows 1, 3 or 4
    stride = frame.width * frame.channels
    image = QImage(frame.pixels, frame.width, frame.height, stride, fmt).copy()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    # QImageWriter rather than QImage.save: the save() stubs and runtime
    # disagree on the format argument's type across PySide6 releases.
    if not QImageWriter(buffer, QByteArray(b"png")).write(image):
        raise ConfigurationError("Could not encode the frame as PNG.")
    return bytes(buffer.data().data())


__all__ = ["CHANNELS", "decode_image", "encode_png"]
