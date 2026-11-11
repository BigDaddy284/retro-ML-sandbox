"""T-971: the AI finds the things on a screen by itself and groups them into kinds."""

from __future__ import annotations

import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.observation.objects import (
    ObjectCatalogue,
    ScreenObject,
    background_colour,
    find_objects,
)
from retro_ai_lab.observation.sprites import SpriteFinder

SKY = (92, 148, 252)
BROWN = (200, 76, 12)
BLACK = (0, 0, 0)
WHITE = (252, 252, 252)
GOLD = (252, 188, 60)


def walker() -> np.ndarray:
    s = np.zeros((16, 16, 3), np.uint8)
    s[:] = SKY
    s[2:10, 3:13] = BROWN
    s[1, 5:11] = BROWN
    s[5:7, 5] = WHITE
    s[5:7, 9] = WHITE
    s[10:14, 6:10] = WHITE
    s[14:16, 2:7] = BLACK
    s[15, 10:13] = BLACK
    return s


def coin() -> np.ndarray:
    c = np.zeros((8, 6, 3), np.uint8)
    c[:] = SKY
    c[1:7, 1:5] = GOLD
    c[3:5, 2:4] = WHITE
    return c


def scene(placements: list[tuple[int, int, np.ndarray]], *, number: int = 0) -> Frame:
    a = np.zeros((240, 256, 3), np.uint8)
    a[:] = SKY
    for x, y, spr in placements:
        h, w = spr.shape[:2]
        solid = (spr != np.array(SKY, np.uint8)).any(axis=2)
        a[y : y + h, x : x + w][solid] = spr[solid]
    return Frame(256, 240, 3, a.tobytes(), number=number)


def boxes(objs: tuple[ScreenObject, ...]) -> list[tuple[int, int, int, int]]:
    return sorted((o.x, o.y, o.width, o.height) for o in objs)


# ------------------------------------------------------------ background


def test_background_is_the_commonest_colour() -> None:
    assert background_colour(scene([(10, 10, walker())])) == SKY


def test_background_of_a_grey_frame() -> None:
    f = Frame(4, 4, 1, bytes([7] * 12 + [9] * 4))
    assert background_colour(f) == (7,)


# ------------------------------------------------------------ one screen


def test_finds_each_thing_with_its_box() -> None:
    objs = find_objects(scene([(10, 20, walker()), (100, 50, coin())]))
    # The walker's solid pixels span x 2..12, y 1..15 of its 16x16 cell.
    assert boxes(objs) == [(12, 21, 11, 15), (101, 51, 4, 6)]


def test_an_empty_screen_has_nothing() -> None:
    assert find_objects(scene([])) == ()


def test_parts_a_pixel_apart_are_one_thing() -> None:
    """The walker's feet do not touch its body diagonally everywhere; it is still one thing."""
    s = np.zeros((10, 10, 3), np.uint8)
    s[:] = SKY
    s[0:3, 0:3] = BLACK
    s[4:7, 0:3] = BLACK  # a one-pixel gap below the first part
    objs = find_objects(scene([(30, 30, s)]))
    assert boxes(objs) == [(30, 30, 3, 7)]


def test_things_further_apart_stay_separate() -> None:
    s = np.zeros((10, 10, 3), np.uint8)
    s[:] = SKY
    s[0:3, 0:3] = BLACK
    s[6:9, 0:3] = BLACK  # three pixels apart
    assert len(find_objects(scene([(30, 30, s)]))) == 2


def test_specks_are_ignored() -> None:
    s = np.zeros((1, 1, 3), np.uint8)
    s[:] = BLACK
    assert find_objects(scene([(30, 30, s)])) == ()
    assert len(find_objects(scene([(30, 30, s)]), min_pixels=1)) == 1


def test_big_areas_are_scenery_not_things() -> None:
    a = np.zeros((240, 256, 3), np.uint8)
    a[:] = SKY
    a[200:] = BROWN  # the ground: one huge area
    a[100:116, 100:116] = walker()
    objs = find_objects(Frame(256, 240, 3, a.tobytes()))
    assert [o.scenery for o in sorted(objs, key=lambda o: o.y)] == [False, True]


def test_a_given_background_colour_is_used() -> None:
    objs = find_objects(scene([(10, 20, walker())]), background=BROWN)
    assert any(o.scenery for o in objs)  # now the sky is the thing


def test_settings_are_checked() -> None:
    with pytest.raises(ConfigurationError):
        find_objects(scene([]), min_pixels=0)
    with pytest.raises(ConfigurationError):
        find_objects(scene([]), max_size=0)
    with pytest.raises(ConfigurationError):
        find_objects(scene([]), background=(1, 2))


def test_works_on_the_toy_racer() -> None:
    """The practice console's car and obstacles are found with no help."""
    game = ToyRacer(seed=3)
    for _ in range(8):  # off the start line, which is drawn through the car
        game.step(frozenset({"up"}))
    frame = game.render()
    objs = find_objects(frame, min_pixels=2)
    assert objs  # the car at least
    car = [o for o in objs if o.y >= frame.height - 5]
    assert car and car[0].width == 3


@settings(max_examples=30, deadline=None)
@given(
    x=st.integers(min_value=0, max_value=240),
    y=st.integers(min_value=0, max_value=224),
)
def test_found_wherever_it_is(x: int, y: int) -> None:
    objs = find_objects(scene([(x, y, walker())]))
    assert boxes(objs) == [(x + 2, y + 1, 11, 15)]


def test_fast_enough() -> None:
    frame = scene([(x, y, walker()) for x in range(0, 240, 40) for y in range(0, 200, 50)])
    find_objects(frame)
    start = time.perf_counter()
    for _ in range(10):
        find_objects(frame)
    per = (time.perf_counter() - start) / 10
    assert per < 0.05, f"{per * 1000:.1f} ms"


# ------------------------------------------------------------ kinds across screens


def test_copies_are_one_kind_and_kinds_keep_their_names() -> None:
    cat = ObjectCatalogue()
    first = cat.observe(scene([(10, 20, walker()), (100, 20, walker()), (50, 100, coin())]))
    kinds = {o.kind for o in first}
    assert len(kinds) == 2
    walker_kind = next(o.kind for o in first if o.width == 11)
    later = cat.observe(scene([(200, 150, walker())]))
    assert [o.kind for o in later] == [walker_kind]
    k = cat.kind(walker_kind)
    assert k.sightings == 3 and k.screens == 2


def test_a_mirrored_copy_is_the_same_kind() -> None:
    cat = ObjectCatalogue()
    a = cat.observe(scene([(10, 20, walker())]))
    b = cat.observe(scene([(10, 20, walker()[:, ::-1])]))
    assert a[0].kind == b[0].kind


def test_kind_names_are_plain_and_numbered_in_order_seen() -> None:
    cat = ObjectCatalogue()
    cat.observe(scene([(10, 20, walker())]))
    cat.observe(scene([(50, 100, coin())]))
    assert [k.name for k in cat.kinds()] == ["thing 1", "thing 2"]


def test_a_kind_can_be_found_again_with_the_sprite_finder() -> None:
    cat = ObjectCatalogue()
    obj = cat.observe(scene([(10, 20, walker())]))[0]
    template = cat.kind(obj.kind).template
    found = SpriteFinder([template]).find(scene([(150, 60, walker())]))
    assert [(m.x, m.y) for m in found] == [(152, 61)]


def test_scenery_is_not_catalogued() -> None:
    a = np.zeros((240, 256, 3), np.uint8)
    a[:] = SKY
    a[200:] = BROWN
    cat = ObjectCatalogue()
    objs = cat.observe(Frame(256, 240, 3, a.tobytes()))
    assert objs[0].scenery and objs[0].kind is None
    assert cat.kinds() == ()


def test_unknown_kind_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        ObjectCatalogue().kind("thing-9")


def test_catalogue_has_a_limit() -> None:
    cat = ObjectCatalogue(most=2)
    s3 = np.zeros((4, 4, 3), np.uint8)
    s3[:] = BLACK
    cat.observe(scene([(10, 10, walker()), (60, 60, coin()), (120, 120, s3)]))
    assert len(cat.kinds()) == 2
    assert cat.dropped == 1


def test_animation_frames_are_one_kind() -> None:
    """T-977: a walker's two steps are drawn differently in the same colours: one kind.
    The user's game made a new kind of every frame (over 250 in an hour)."""
    step = walker()
    other = walker()
    other[14:16, 2:7] = SKY
    other[14:16, 9:14] = BLACK  # the other foot forward
    cat = ObjectCatalogue()
    a = cat.observe(scene([(10, 20, step)]))
    b = cat.observe(scene([(10, 20, other)]))
    assert a[0].kind == b[0].kind
    assert len(cat.kinds()) == 1


def test_different_colours_are_different_kinds() -> None:
    cat = ObjectCatalogue()
    found = cat.observe(scene([(10, 20, walker()), (100, 50, coin())]))
    assert len({o.kind for o in found}) == 2
