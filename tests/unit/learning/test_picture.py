"""T-1055: the screen, shrunk to one small grey picture whatever the console."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.learning import (
    NO_MOVE,
    PICTURE_HEIGHT,
    PICTURE_SIZE,
    PICTURE_WIDTH,
    Sight,
    shrink,
)

SIZES = [(64, 48), (256, 240), (160, 144), (320, 224), (33, 29), (7, 5), (1, 1)]
EVEN = [(64, 48), (256, 240), (160, 144), (320, 224)]


def frame(width: int, height: int, channels: int, grey: Callable[[int, int], int]) -> Frame:
    """A screen whose pixel at (x, y) is ``grey(x, y)`` in every colour channel; a
    fourth channel, alpha, is 255."""
    out = bytearray()
    for y in range(height):
        for x in range(width):
            value = grey(x, y)
            out += bytes([value] * min(channels, 3))
            if channels == 4:
                out.append(255)
    return Frame(width, height, channels, bytes(out))


def rows(picture: bytes) -> list[bytes]:
    return [picture[r * PICTURE_WIDTH : (r + 1) * PICTURE_WIDTH] for r in range(PICTURE_HEIGHT)]


def test_the_picture_is_32_by_28() -> None:
    assert (PICTURE_WIDTH, PICTURE_HEIGHT, PICTURE_SIZE) == (32, 28, 896)


@pytest.mark.parametrize(("width", "height"), SIZES)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_any_screen_becomes_a_picture_of_the_one_size(
    width: int, height: int, channels: int
) -> None:
    picture = shrink(frame(width, height, channels, lambda x, y: (3 * x + 5 * y) % 256))
    assert isinstance(picture, bytes) and len(picture) == PICTURE_SIZE
    # The same screen, the same picture.
    assert picture == shrink(frame(width, height, channels, lambda x, y: (3 * x + 5 * y) % 256))
    # One plain colour stays that colour everywhere.
    assert shrink(frame(width, height, channels, lambda x, y: 77)) == bytes([77]) * PICTURE_SIZE


@pytest.mark.parametrize(("width", "height"), EVEN)
@pytest.mark.parametrize("channels", [1, 3])
def test_a_screen_in_two_halves_is_a_picture_in_two_halves(
    width: int, height: int, channels: int
) -> None:
    sides = shrink(frame(width, height, channels, lambda x, y: 255 if x < width // 2 else 0))
    for row in rows(sides):
        assert row == bytes([255]) * 16 + bytes([0]) * 16
    over = shrink(frame(width, height, channels, lambda x, y: 255 if y < height // 2 else 0))
    assert rows(over) == [bytes([255]) * 32] * 14 + [bytes([0]) * 32] * 14


@pytest.mark.parametrize(("width", "height"), [(33, 29), (7, 5)])
def test_an_odd_sized_screen_keeps_its_sides_and_nothing_is_cropped(
    width: int, height: int
) -> None:
    sides = rows(shrink(frame(width, height, 1, lambda x, y: 255 if x < width // 2 else 0)))
    for row in sides:
        assert row[0] == 255 and row[-1] == 0
        assert list(row) == sorted(row, reverse=True)  # bright to dark, once, left to right
    # The last column and the last row of the screen are in the picture: a line down
    # the right edge and one along the bottom both show.
    edge = rows(shrink(frame(width, height, 1, lambda x, y: 255 if x == width - 1 else 0)))
    assert all(row[-1] > 0 and row[0] == 0 for row in edge)
    foot = rows(shrink(frame(width, height, 1, lambda x, y: 255 if y == height - 1 else 0)))
    assert all(v > 0 for v in foot[-1]) and all(v == 0 for v in foot[0])


def test_a_cell_is_the_average_of_what_it_covers() -> None:
    # 64 by 56: every cell covers two pixels by two.
    values = {(0, 0): 10, (1, 0): 20, (0, 1): 30, (1, 1): 40}
    picture = shrink(frame(64, 56, 1, lambda x, y: values[(x % 2, y % 2)]))
    assert picture == bytes([25]) * PICTURE_SIZE
    # 256 by 240: one bright pixel lights its own cell only, by its share of the cell.
    one = rows(shrink(frame(256, 240, 1, lambda x, y: 255 if (x, y) == (100, 50) else 0)))
    row, col = 5, 12  # rows 42 to 50 and columns 96 to 103 are that cell
    lit = [(r, c) for r in range(PICTURE_HEIGHT) for c in range(PICTURE_WIDTH) if one[r][c]]
    assert lit == [(row, col)]
    assert one[row][col] == round(255 / (8 * 9))


def test_a_screen_smaller_than_the_picture_has_its_pixels_repeated() -> None:
    source = [[(40 * y + 7 * x) % 256 for x in range(7)] for y in range(5)]
    picture = rows(shrink(frame(7, 5, 1, lambda x, y: source[y][x])))
    for r in range(PICTURE_HEIGHT):
        for c in range(PICTURE_WIDTH):
            assert picture[r][c] == source[r * 5 // PICTURE_HEIGHT][c * 7 // PICTURE_WIDTH]


def test_grey_is_the_mean_of_the_colours_and_alpha_is_left_out() -> None:
    rgb = Frame(64, 56, 3, bytes([30, 60, 90]) * (64 * 56))
    assert shrink(rgb) == bytes([60]) * PICTURE_SIZE
    for alpha in (0, 200, 255):
        rgba = Frame(64, 56, 4, bytes([30, 60, 90, alpha]) * (64 * 56))
        assert shrink(rgba) == bytes([60]) * PICTURE_SIZE
    # Colour by colour, not the first channel alone.
    assert shrink(Frame(64, 56, 3, bytes([255, 0, 0]) * (64 * 56))) == bytes([85]) * PICTURE_SIZE
    assert shrink(Frame(64, 56, 3, bytes([0, 0, 255]) * (64 * 56))) == bytes([85]) * PICTURE_SIZE


@pytest.mark.parametrize("not_a_screen", [None, "screen", b"\x00" * 896, 7, [1, 2, 3]])
def test_what_is_not_a_screen_is_refused(not_a_screen: Any) -> None:
    with pytest.raises(ConfigurationError, match="Only a captured screen can be shrunk"):
        shrink(not_a_screen)


# ----------------------------------------------------------------------- sight

DARK = bytes([20]) * PICTURE_SIZE
BRIGHT = bytes([200]) * PICTURE_SIZE


def test_a_sight_is_the_picture_now_the_one_before_and_the_move_made() -> None:
    sight = Sight(BRIGHT, DARK, 3)
    assert (sight.now, sight.before, sight.held) == (BRIGHT, DARK, 3)
    assert Sight(BRIGHT, DARK).held == NO_MOVE == -1
    assert Sight(BRIGHT, DARK, 0).held == 0
    assert Sight(BRIGHT, DARK, 3) == sight and Sight(BRIGHT, BRIGHT, 3) != sight
    with pytest.raises(FrozenInstanceError):
        sight.held = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "picture", [BRIGHT[:-1], BRIGHT + b"\x00", b"", bytearray(BRIGHT), None, "x" * 896]
)
def test_a_sight_refuses_a_picture_of_the_wrong_size_or_kind(picture: Any) -> None:
    with pytest.raises(ConfigurationError, match=r"A sight's picture \(now\) must be 896 bytes"):
        Sight(picture, DARK)
    with pytest.raises(ConfigurationError, match=r"A sight's picture \(before\) must be 896"):
        Sight(DARK, picture)


@pytest.mark.parametrize("held", [-2, -100, True, False, 1.0, "1", None])
def test_a_sight_refuses_what_is_no_move(held: Any) -> None:
    with pytest.raises(ConfigurationError, match="A sight's move must be a move's number"):
        Sight(BRIGHT, DARK, held)
