"""T-008: FrameView and frame_to_image."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QImage
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.ui.frame_view import BACKGROUND, PLACEHOLDER_TEXT, FrameView, frame_to_image


def test_frame_to_image_grey() -> None:
    f = Frame(2, 1, 1, bytes([0, 255]))
    img = frame_to_image(f)
    assert (img.width(), img.height()) == (2, 1)
    assert img.format() == QImage.Format.Format_Grayscale8
    assert img.pixelColor(0, 0).red() == 0
    assert img.pixelColor(1, 0).red() == 255


def test_frame_to_image_rgb_and_rgba() -> None:
    rgb = Frame(1, 1, 3, bytes([10, 20, 30]))
    img = frame_to_image(rgb)
    c = img.pixelColor(0, 0)
    assert (c.red(), c.green(), c.blue()) == (10, 20, 30)
    rgba = Frame(1, 1, 4, bytes([10, 20, 30, 40]))
    img = frame_to_image(rgba)
    c = img.pixelColor(0, 0)
    assert (c.red(), c.green(), c.blue(), c.alpha()) == (10, 20, 30, 40)


def test_frame_to_image_is_a_deep_copy() -> None:
    f = Frame(1, 1, 1, bytes([7]))
    img = frame_to_image(f)
    del f
    assert img.pixelColor(0, 0).red() == 7


@pytest.mark.parametrize("bad", [None, b"\x00", "frame", 3])
def test_frame_to_image_rejects_non_frames(bad: object) -> None:
    with pytest.raises(ConfigurationError):
        frame_to_image(bad)  # type: ignore[arg-type]


def test_view_starts_empty_with_placeholder(qtbot: QtBot) -> None:
    v = FrameView()
    qtbot.addWidget(v)
    v.show()
    assert v.frame is None
    assert v.placeholder == PLACEHOLDER_TEXT
    assert v.image_rect().isNull()
    v.set_placeholder("Waiting")
    assert v.placeholder == "Waiting"
    shot = v.grab().toImage()
    # Background colour painted, and some lighter text pixels present.
    assert shot.pixelColor(0, 0) == BACKGROUND
    assert any(shot.pixelColor(x, shot.height() // 2).red() > 100 for x in range(shot.width()))


def test_view_draws_frame_centred_with_integer_scale(qtbot: QtBot) -> None:
    v = FrameView()
    qtbot.addWidget(v)
    v.resize(400, 200)
    v.show()
    # 4x2 frame: left half black, right half white.
    f = Frame(4, 2, 1, bytes([0, 0, 255, 255, 0, 0, 255, 255]))
    v.set_frame(f)
    assert v.frame is f
    rect = v.image_rect()
    # scale = min(400/4, 200/2) = 100 -> 400x200 drawn, centred at origin.
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (0, 0, 400, 200)
    shot = v.grab().toImage()
    assert shot.pixelColor(50, 100).red() == 0
    assert shot.pixelColor(350, 100).red() == 255

    v.resize(600, 200)
    rect = v.image_rect()
    assert (rect.width(), rect.height()) == (400, 200)
    assert rect.x() == 100  # centred horizontally


def test_view_fits_large_frames_down(qtbot: QtBot) -> None:
    v = FrameView()
    qtbot.addWidget(v)
    v.resize(128, 96)
    v.show()
    v.set_frame(Frame(640, 480, 1, bytes(640 * 480)))
    rect = v.image_rect()
    assert rect.width() <= 128 and rect.height() <= 96
    assert abs(rect.width() / rect.height() - 640 / 480) < 0.05


def test_view_clear_restores_placeholder(qtbot: QtBot) -> None:
    v = FrameView()
    qtbot.addWidget(v)
    v.set_frame(Frame(1, 1, 1, b"\x00"))
    v.clear()
    assert v.frame is None
    assert v.image_rect().isNull()
    v.set_frame(None)
    assert v.frame is None


def test_view_rejects_non_frames(qtbot: QtBot) -> None:
    v = FrameView()
    qtbot.addWidget(v)
    with pytest.raises(ConfigurationError):
        v.set_frame("nope")  # type: ignore[arg-type]
