"""T-113: screenshots to frames and back."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import Frame, ProtocolError
from retro_ai_lab.emulator.bizhawk.image import CHANNELS, decode_image, encode_png
from retro_ai_lab.emulator.fake.toy_racer import ToyRacer


def _rgb(width: int, height: int) -> bytes:
    return bytes(
        (x * 7 + y * 3 + c * 50) % 256
        for y in range(height)
        for x in range(width)
        for c in range(3)
    )


def test_rgb_round_trip_keeps_pixels_and_number() -> None:
    frame = Frame(5, 3, 3, _rgb(5, 3), number=1)  # 5 wide: rows are not 4-byte aligned
    back = decode_image(encode_png(frame), number=42)
    assert (back.width, back.height, back.channels, back.number) == (5, 3, CHANNELS, 42)
    assert back.pixels == frame.pixels


def test_grey_and_rgba_decode_to_rgb() -> None:
    grey = ToyRacer(0).render()
    assert grey.channels == 1
    back = decode_image(encode_png(grey))
    assert (back.width, back.height, back.channels) == (grey.width, grey.height, 3)
    assert back.pixels == bytes(v for v in grey.pixels for _ in range(3))
    rgba = Frame(2, 2, 4, bytes([10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 1, 2, 3, 255]))
    back = decode_image(encode_png(rgba))
    assert back.channels == 3
    assert back.pixels == bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 2, 3])


def test_bad_images_are_reported() -> None:
    with pytest.raises(ProtocolError, match="could not be read as an image"):
        decode_image(b"definitely not a png")
    with pytest.raises(ProtocolError):
        decode_image(b"")
