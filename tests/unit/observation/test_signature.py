"""T-987: knowing a thing by its colours, and finding it on one screen with no memory."""

from __future__ import annotations

import time

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.signature import COMMON, REACH, ColourSignature

SKY = (92, 148, 252)
BROWN = (200, 76, 12)
BLACK = (0, 0, 0)
RED = (228, 0, 88)
SKIN = (252, 188, 176)
DARK = (120, 40, 0)
GOLD = (252, 188, 60)

W, H = 256, 240
GROUND = 208
HERO_W, HERO_H = 12, 16


def pack(colour: tuple[int, int, int]) -> int:
    return (colour[0] << 16) | (colour[1] << 8) | colour[2]


def scene() -> np.ndarray:
    """Sky, and brick ground with black outlines from y=208."""
    a = np.zeros((H, W, 3), np.uint8)
    a[:] = SKY
    a[GROUND:] = BROWN
    a[GROUND] = BLACK
    a[GROUND:, ::16] = BLACK
    return a


def hero(a: np.ndarray, x: int, y: int) -> None:
    """12 by 16: a skin head with black eyes over a red body (drawn clipped at the edges)."""

    def fill(x0: int, y0: int, x1: int, y1: int, colour: tuple[int, int, int]) -> None:
        a[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = colour

    fill(x, y + 4, x + HERO_W, y + HERO_H, RED)
    fill(x + 2, y, x + 10, y + 4, SKIN)
    fill(x + 3, y + 1, x + 5, y + 2, BLACK)
    fill(x + 7, y + 1, x + 9, y + 2, BLACK)


def walker(a: np.ndarray, x: int, y: int) -> None:
    a[y : y + 14, x : x + 14] = DARK
    a[y + 12 : y + 14, x : x + 14] = BLACK


def frame(a: np.ndarray, number: int = 0) -> Frame:
    return Frame(a.shape[1], a.shape[0], 3, a.tobytes(), number)


def standing(x: int, y: int = GROUND - HERO_H) -> Frame:
    a = scene()
    hero(a, x, y)
    return frame(a)


@pytest.fixture(scope="module")
def sign() -> ColourSignature:
    found = ColourSignature.learn([(standing(40), (40, GROUND - HERO_H, HERO_W, HERO_H))])
    assert found is not None
    return found


def test_learns_the_colours_and_size_from_whole_boxes(sign: ColourSignature) -> None:
    # Red body 12x12 and skin head 8x4 less four eye pixels; the eyes' black is the
    # ground's outline colour too, so it says nothing and is left out.
    assert sign.colours == ((pack(RED), 144), (pack(SKIN), 28))
    assert (sign.width, sign.height, sign.channels) == (HERO_W, HERO_H, 3)
    assert sign.pixels == 172
    assert sign.describe() == "2 colours of its own, 12 by 16 pixels"


def test_learns_the_whole_thing_from_slivers_of_its_edges(sign: ColourSignature) -> None:
    # Walking right, only its leading and trailing edges change on the screen: the
    # boxes are slivers three pixels wide, some of them holding nothing but sky.
    samples = []
    for i, x in enumerate(range(40, 80, 5)):
        lead = (x + HERO_W - 3, GROUND - HERO_H, 3, HERO_H)
        trail = (x - 3, GROUND - HERO_H, 3, HERO_H)
        samples.append((standing(x), lead if i % 2 == 0 else trail))
    assert ColourSignature.learn(samples) == sign


def test_scenery_in_the_box_is_left_out(sign: ColourSignature) -> None:
    # The box takes in the ground under its feet: brown and black are scenery (the
    # patch they make is the whole floor), and the hero is still made out whole.
    box = (38, GROUND - HERO_H - 2, HERO_W + 4, HERO_H + 8)
    assert ColourSignature.learn([(standing(40), box)]) == sign
    # Were a thing at most 8 pixels across, the 12-wide red body would be scenery too
    # and the head alone the thing; at 3 pixels nothing is left.
    small = ColourSignature.learn([(standing(40), box)], max_size=8)
    assert small is not None
    assert (small.colours, small.width, small.height) == (((pack(SKIN), 28),), 8, 4)
    assert ColourSignature.learn([(standing(40), box)], max_size=3) is None


def test_a_colour_mostly_elsewhere_is_left_out() -> None:
    # Three coins in the head's colour hang in the sky: skin is mostly elsewhere.
    a = scene()
    hero(a, 40, GROUND - HERO_H)
    for x in (100, 130, 160):
        a[150:158, x : x + 6] = SKIN
    found = ColourSignature.learn([(frame(a), (40, GROUND - HERO_H, HERO_W, HERO_H))])
    assert found is not None
    assert found.colours == ((pack(RED), 144),)
    assert (found.width, found.height) == (12, 12)  # the red body alone


def test_nothing_distinctive_is_no_signature() -> None:
    # A thing drawn only in the ground's colours has none of its own.
    a = scene()
    a[GROUND - 16 : GROUND, 40:52] = BROWN
    assert ColourSignature.learn([(frame(a), (40, GROUND - 16, 12, 16))]) is None
    # Nothing but sky in the boxes; and no samples at all.
    assert ColourSignature.learn([(frame(scene()), (40, 100, 12, 16))]) is None
    assert ColourSignature.learn([]) is None


def test_bad_samples_are_refused() -> None:
    good = standing(40)
    for bad in (
        [good],
        [(good, (1, 2, 3))],
        [(good, (1, 2, 0, 4))],
        [(good, (1.5, 2, 3, 4))],
        [(good, (True, 2, 3, 4))],
        [("screen", (1, 2, 3, 4))],
        [(good, (1, 2, 3, 4)), (Frame(8, 8, 3, bytes(192)), (1, 2, 3, 4))],
    ):
        with pytest.raises(ConfigurationError):
            ColourSignature.learn(bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        ColourSignature.learn([(good, (1, 2, 3, 4))], max_size=0)
    for args in (
        ((), 12, 16),
        (((1, 0),), 12, 16),
        (((1, 5), (1, 6)), 12, 16),
        (((1, 5),), 0, 16),
        (((1, 5),), 12, 16, 2),
        ([[1, 5]], 12, 16),
    ):
        with pytest.raises(ConfigurationError):
            ColourSignature(*args)  # type: ignore[arg-type]


def test_finds_it_on_scenery_in_the_air_touching_and_half_off_screen(
    sign: ColourSignature,
) -> None:
    # Standing on the ground (one patch with it for the object finder), anywhere.
    for x in (0, 40, 133, W - HERO_W):
        assert sign.find(standing(x)) == (x, GROUND - HERO_H, HERO_W, HERO_H)
    # In the air.
    assert sign.find(standing(90, 120)) == (90, 120, HERO_W, HERO_H)
    # Touching a walker.
    a = scene()
    hero(a, 90, GROUND - HERO_H)
    walker(a, 102, GROUND - 14)
    assert sign.find(frame(a)) == (90, GROUND - HERO_H, HERO_W, HERO_H)
    # The walker covers its right half from two rows down: what shows is 10 wide.
    walker(a, 96, GROUND - 14)
    assert sign.find(frame(a)) == (90, GROUND - HERO_H, 10, HERO_H)
    # Half off the left edge: the half in sight is found.
    assert sign.find(standing(-6)) == (0, GROUND - HERO_H, 6, HERO_H)
    # A sliver of it is not enough to say it is there.
    assert sign.find(standing(-10)) is None


def test_of_two_alike_the_nearest_to_where_it_was(sign: ColourSignature) -> None:
    a = scene()
    hero(a, 40, GROUND - HERO_H)
    hero(a, 200, 100)
    both = frame(a)
    assert sign.find(both, near=(206, 108)) == (200, 100, HERO_W, HERO_H)
    assert sign.find(both, near=(46, 200)) == (40, GROUND - HERO_H, HERO_W, HERO_H)
    # It may have moved a good way since: anything within reach is still it.
    assert sign.find(both, near=(46 + REACH - 2, 200)) == (40, GROUND - HERO_H, HERO_W, HERO_H)
    # With nowhere to start from, one of the two (the first met) is given.
    assert sign.find(both) in ((40, GROUND - HERO_H, HERO_W, HERO_H), (200, 100, HERO_W, HERO_H))


def test_found_across_the_screen_when_gone_from_where_it_was(sign: ColourSignature) -> None:
    # A new life starts at the other end of the screen: nothing near where it was.
    there = (200, GROUND - HERO_H, HERO_W, HERO_H)
    assert sign.find(standing(200), near=(20, 100)) == there
    assert sign.find(standing(200), near=(20, 100), reach=0) == there
    # Where it was last seen can be off the screen altogether.
    assert sign.find(standing(200), near=(-500, 9000)) == there


def test_a_colour_that_floods_the_screen_is_ignored_for_that_look(sign: ColourSignature) -> None:
    # A wall in its shirt's red: for this look the head's colour alone says where it is.
    a = scene()
    a[60:100, 100:200] = RED
    hero(a, 40, GROUND - HERO_H)
    assert COMMON * 144 < 40 * 100
    assert sign.find(frame(a)) == (42, GROUND - HERO_H, 8, 4)  # the head
    # A little more red than its own (a mushroom) floods nothing: found whole.
    b = scene()
    b[60:70, 100:110] = RED
    hero(b, 40, GROUND - HERO_H)
    assert sign.find(frame(b), near=(46, 200)) == (40, GROUND - HERO_H, HERO_W, HERO_H)
    # Every colour flooding: it cannot say, and says so rather than guess.
    a[110:150, 100:200] = SKIN
    assert sign.find(frame(a)) is None


def test_hidden_is_not_found(sign: ColourSignature) -> None:
    assert sign.find(frame(scene())) is None
    assert sign.find(frame(scene()), near=(46, 200)) is None
    # A speck of its colour is not it.
    a = scene()
    a[100:102, 50:52] = RED
    assert sign.find(frame(a)) is None
    # A screen with other channels is not looked at; nonsense is refused.
    assert sign.find(Frame(W, H, 1, bytes(W * H))) is None
    with pytest.raises(ConfigurationError):
        sign.find("screen")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        sign.find(standing(40), reach=-1)


def test_a_signature_is_saved_and_read_back(sign: ColourSignature) -> None:
    saved = sign.to_dict()
    assert saved == {
        "colours": [[pack(RED), 144], [pack(SKIN), 28]],
        "width": 12,
        "height": 16,
        "channels": 3,
    }
    assert ColourSignature.from_dict(saved) == sign
    for bad in (
        {},
        {**saved, "colours": []},
        {**saved, "colours": [[1]]},
        {**saved, "colours": "red"},
        {**saved, "width": "wide"},
        {**saved, "channels": 7},
    ):
        with pytest.raises(ConfigurationError):
            ColourSignature.from_dict(bad)


def test_finding_is_quick_enough_for_every_look(sign: ColourSignature) -> None:
    screens = [standing(x) for x in range(0, 240, 12)]
    began = time.perf_counter()
    for _ in range(5):
        for screen in screens:
            assert sign.find(screen, near=(100, 200)) is not None
    each = (time.perf_counter() - began) / (5 * len(screens))
    assert each < 0.05  # a look already costs ~15 ms; this adds a millisecond or two
