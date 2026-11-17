"""T-1053: the most a layout can hold, which a count taken below nought reads."""

from __future__ import annotations

import pytest

from retro_ai_lab.understanding import Encoding
from retro_ai_lab.understanding.knowledge import Variable


@pytest.mark.parametrize(
    ("layout", "length", "most"),
    [
        (Encoding.U8, 0, 255),
        (Encoding.U16LE, 0, 65_535),
        (Encoding.U16BE, 0, 65_535),
        (Encoding.BCD, 0, 99),
        (Encoding.UINT_LE, 1, 255),
        (Encoding.UINT_LE, 2, 65_535),
        (Encoding.UINT_BE, 3, 16_777_215),
        (Encoding.BCD_LE, 1, 99),
        (Encoding.BCD_LE, 2, 9_999),
        (Encoding.BCD_BE, 3, 999_999),
        (Encoding.DIGITS_LE, 1, 9),
        (Encoding.DIGITS_LE, 2, 99),
        (Encoding.DIGITS_BE, 3, 999),
        (Encoding.S8, 0, None),  # it reads -1 below nought: a fall as it is
        (Encoding.FLAG, 0, None),  # on or off: no number
    ],
)
def test_the_most_each_layout_holds(layout: Encoding, length: int, most: int | None) -> None:
    assert layout.most(length) == most
    held = Variable("v1", "ram", 0, layout, length=length)
    assert held.most == most
    if most is not None:
        assert type(held.most) is int


def test_every_layout_is_spoken_for() -> None:
    # A layout added later has to say what it holds, or that it holds no such thing.
    for layout in Encoding:
        most = layout.most(2 if layout.sized else 0)
        assert most is None or most in (255, 65_535, 99, 9_999)


def test_a_layout_with_its_own_width_takes_no_length() -> None:
    # The length is a sized layout's alone: a byte is a byte whatever is passed.
    assert Encoding.U8.most() == Encoding.U8.most(3) == 255
    assert Encoding.U16LE.most() == Encoding.U16LE.most(1) == 65_535
    assert Encoding.BCD.most() == Encoding.BCD.most(2) == 99
