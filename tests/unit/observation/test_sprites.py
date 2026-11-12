"""T-970: find a sprite a person tagged (an enemy, the player, an item) on the screen."""

from __future__ import annotations

import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.observation.sprites import (
    SpriteFinder,
    SpriteMatch,
    SpriteTag,
    SpriteTemplate,
)

SKY = (92, 148, 252)
BUSH = (0, 168, 0)
BROWN = (200, 76, 12)
BLACK = (0, 0, 0)
WHITE = (252, 252, 252)


def walker() -> np.ndarray:
    """A 16x16 'enemy': a brown mushroom with eyes and feet, sky around it, not symmetric."""
    s = np.zeros((16, 16, 3), np.uint8)
    s[:] = SKY
    s[2:10, 3:13] = BROWN
    s[1, 5:11] = BROWN
    s[5:7, 5] = WHITE
    s[5:7, 9] = WHITE
    s[10:14, 6:10] = WHITE
    s[14:16, 2:7] = BLACK  # one foot forward: the left/right mirror differs
    s[15, 10:13] = BLACK
    return s


def scene(
    placements: list[tuple[int, int, np.ndarray]],
    *,
    width: int = 256,
    height: int = 240,
    background: tuple[int, int, int] = SKY,
    number: int = 0,
) -> Frame:
    """A frame with each sprite drawn where its pixels differ from SKY (see-through)."""
    a = np.zeros((height, width, 3), np.uint8)
    a[:] = background
    a[200:] = BROWN  # the ground
    for x, y, spr in placements:
        h, w = spr.shape[:2]
        solid = (spr != np.array(SKY, np.uint8)).any(axis=2)
        region = a[y : y + h, x : x + w]
        region[solid[: region.shape[0], : region.shape[1]]] = spr[
            : region.shape[0], : region.shape[1]
        ][solid[: region.shape[0], : region.shape[1]]]
    return Frame(width, height, 3, a.tobytes(), number=number)


def enemy_template(**kw: object) -> SpriteTemplate:
    f = scene([(40, 184, walker())])
    return SpriteTemplate.from_frame(
        f,
        40,
        184,
        16,
        16,
        name="walking mushroom",
        tag=SpriteTag.ENEMY,
        **kw,  # type: ignore[arg-type]
    )


def spots(matches: tuple[SpriteMatch, ...]) -> list[tuple[int, int]]:
    return sorted((m.x, m.y) for m in matches)


# ------------------------------------------------------------ template


def test_template_from_frame_masks_the_background_out() -> None:
    t = enemy_template()
    assert (t.width, t.height, t.channels) == (16, 16, 3)
    assert t.name == "walking mushroom" and t.tag is SpriteTag.ENEMY
    mask = np.frombuffer(t.mask, np.uint8).reshape(16, 16)
    expected = (walker() != np.array(SKY, np.uint8)).any(axis=2)
    assert (mask.astype(bool) == expected).all()
    assert 0 < t.solid_pixels < 256


def test_template_with_an_explicit_background_colour() -> None:
    f = scene([(40, 184, walker())])
    t = SpriteTemplate.from_frame(
        f, 40, 184, 16, 16, name="m", tag=SpriteTag.ENEMY, background=BROWN
    )
    mask = np.frombuffer(t.mask, np.uint8).reshape(16, 16).astype(bool)
    brown = (walker() == np.array(BROWN, np.uint8)).all(axis=2)
    assert not mask[brown].any()  # brown was declared see-through
    assert mask[5, 5]  # the white eye is still solid
    assert mask[0, 0]  # sky is now part of the sprite


def test_a_crop_with_no_background_keeps_every_pixel() -> None:
    f = scene([])
    t = SpriteTemplate.from_frame(f, 0, 0, 8, 8, name="sky", tag=SpriteTag.OTHER)
    assert t.solid_pixels == 64


@pytest.mark.parametrize(
    ("x", "y", "w", "h"),
    [(-1, 0, 4, 4), (0, -1, 4, 4), (250, 0, 8, 4), (0, 238, 4, 4), (0, 0, 0, 4), (0, 0, 4, 0)],
)
def test_template_crop_must_lie_inside_the_frame(x: int, y: int, w: int, h: int) -> None:
    with pytest.raises(ConfigurationError):
        SpriteTemplate.from_frame(scene([]), x, y, w, h, name="x", tag=SpriteTag.ENEMY)


@pytest.mark.parametrize("name", ["", "   ", 3])
def test_template_needs_a_name(name: object) -> None:
    with pytest.raises(ConfigurationError):
        SpriteTemplate.from_frame(scene([]), 0, 0, 4, 4, name=name, tag=SpriteTag.ENEMY)  # type: ignore[arg-type]


def test_template_tag_must_be_a_sprite_tag() -> None:
    with pytest.raises(ConfigurationError):
        SpriteTemplate.from_frame(scene([]), 0, 0, 4, 4, name="x", tag="baddie")  # type: ignore[arg-type]


def test_template_background_must_match_the_channels() -> None:
    with pytest.raises(ConfigurationError):
        SpriteTemplate.from_frame(
            scene([]), 0, 0, 4, 4, name="x", tag=SpriteTag.ENEMY, background=(1, 2)
        )


def test_template_rejects_inconsistent_fields() -> None:
    t = enemy_template()
    with pytest.raises(ConfigurationError):
        SpriteTemplate(t.name, t.tag, 16, 16, 3, t.pixels[:-1], t.mask)
    with pytest.raises(ConfigurationError):
        SpriteTemplate(t.name, t.tag, 16, 16, 3, t.pixels, t.mask[:-1])
    with pytest.raises(ConfigurationError):
        SpriteTemplate(t.name, t.tag, 16, 16, 3, t.pixels, bytes(256))  # nothing solid


def test_template_round_trips_through_a_dict() -> None:
    t = enemy_template()
    back = SpriteTemplate.from_dict(t.to_dict())
    assert back == t
    import json

    assert SpriteTemplate.from_dict(json.loads(json.dumps(t.to_dict()))) == t


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"name": "x"},
        "not a dict",
        {
            "name": "x",
            "tag": "enemy",
            "width": 2,
            "height": 1,
            "channels": 1,
            "pixels": "!!",
            "mask": "AQE=",
        },
        {
            "name": "x",
            "tag": "boss",
            "width": 2,
            "height": 1,
            "channels": 1,
            "pixels": "AQE=",
            "mask": "AQE=",
        },
    ],
)
def test_template_from_a_broken_dict_is_a_configuration_error(broken: object) -> None:
    with pytest.raises(ConfigurationError):
        SpriteTemplate.from_dict(broken)  # type: ignore[arg-type]


# ------------------------------------------------------------ finding


def test_finds_every_copy_of_the_enemy_and_tags_it() -> None:
    finder = SpriteFinder([enemy_template()])
    frame = scene([(10, 184, walker()), (120, 184, walker()), (230, 100, walker())])
    found = finder.find(frame)
    assert spots(found) == [(10, 184), (120, 184), (230, 100)]
    for m in found:
        assert m.tag is SpriteTag.ENEMY and m.name == "walking mushroom"
        assert m.score == 1.0 and not m.mirrored
        assert (m.width, m.height) == (16, 16)
        assert m.centre == (m.x + 8, m.y + 8)


def test_an_empty_screen_has_no_enemies() -> None:
    assert SpriteFinder([enemy_template()]).find(scene([])) == ()


def test_see_through_pixels_ignore_what_is_behind_the_enemy() -> None:
    """Tagged against the sky, still found in front of a bush (a plain crop would fail)."""
    frame = scene([(60, 120, walker())], background=BUSH)
    assert spots(SpriteFinder([enemy_template()]).find(frame)) == [(60, 120)]


def test_finds_the_enemy_facing_the_other_way() -> None:
    frame = scene([(60, 184, walker()[:, ::-1])])
    found = SpriteFinder([enemy_template()]).find(frame)
    assert spots(found) == [(60, 184)]
    assert found[0].mirrored


def test_mirrors_can_be_turned_off() -> None:
    frame = scene([(60, 184, walker()[:, ::-1])])
    assert SpriteFinder([enemy_template()], mirrored=False).find(frame) == ()


def test_partly_off_the_edge_is_not_reported_but_whole_at_the_edge_is() -> None:
    finder = SpriteFinder([enemy_template()])
    assert spots(finder.find(scene([(240, 0, walker())]))) == [(240, 0)]
    assert finder.find(scene([(248, 0, walker())])) == ()


def test_a_slightly_different_palette_still_matches_within_tolerance() -> None:
    tinted = np.clip(walker().astype(int) + 10, 0, 255).astype(np.uint8)
    tinted[(walker() == np.array(SKY, np.uint8)).all(axis=2)] = SKY
    frame = scene([(30, 30, tinted)])
    assert spots(SpriteFinder([enemy_template()], tolerance=16).find(frame)) == [(30, 30)]
    assert SpriteFinder([enemy_template()], tolerance=4).find(frame) == ()


def test_a_partly_covered_enemy_scores_below_one_and_min_score_decides() -> None:
    frame_arr = (
        np.frombuffer(scene([(30, 30, walker())]).pixels, np.uint8).reshape(240, 256, 3).copy()
    )
    frame_arr[30:32, 30:46] = BUSH  # cover the top two rows of the enemy
    frame = Frame(256, 240, 3, frame_arr.tobytes())
    loose = SpriteFinder([enemy_template()], min_score=0.8).find(frame)
    assert spots(loose) == [(30, 30)]
    assert 0.8 <= loose[0].score < 1.0
    assert SpriteFinder([enemy_template()], min_score=0.99).find(frame) == ()


def test_overlapping_hits_collapse_to_the_best_one() -> None:
    """A plain block also scores well one pixel off; only the exact spot is reported."""
    block = np.zeros((8, 8, 3), np.uint8)
    block[:] = SKY
    block[1:7, 1:7] = BLACK
    t = SpriteTemplate.from_frame(
        scene([(0, 0, block)]), 0, 0, 8, 8, name="block", tag=SpriteTag.OTHER
    )
    frame = scene([(50, 50, block)])
    loose = SpriteFinder([t], min_score=0.8)
    found = loose.find(frame)
    assert [(m.x, m.y, m.score) for m in found] == [(50, 50, 1.0)]


def test_two_enemies_overlapping_a_little_are_both_kept() -> None:
    frame = scene([(50, 184, walker()), (62, 184, walker())])  # 4 of 16 columns shared
    assert spots(SpriteFinder([enemy_template()], min_score=0.8).find(frame)) == [
        (50, 184),
        (62, 184),
    ]


def test_several_tags_in_one_pass() -> None:
    player = np.zeros((8, 8, 3), np.uint8)
    player[:] = SKY
    player[1:7, 2:6] = (248, 56, 0)
    player[0, 2:6] = BLACK
    pt = SpriteTemplate.from_frame(
        scene([(0, 0, player)]), 0, 0, 8, 8, name="me", tag=SpriteTag.PLAYER
    )
    finder = SpriteFinder([enemy_template(), pt])
    found = finder.find(scene([(100, 184, walker()), (20, 192, player)]))
    by_tag = {m.tag: (m.x, m.y) for m in found}
    assert by_tag == {SpriteTag.ENEMY: (100, 184), SpriteTag.PLAYER: (20, 192)}
    assert finder.find_tag(scene([(100, 184, walker())]), SpriteTag.PLAYER) == ()


def test_results_are_in_a_stable_order() -> None:
    finder = SpriteFinder([enemy_template()])
    frame = scene([(200, 10, walker()), (10, 150, walker()), (100, 10, walker())])
    found = finder.find(frame)
    assert [(m.x, m.y) for m in found] == [(100, 10), (200, 10), (10, 150)]  # top row first
    assert finder.find(frame) == found


def test_a_template_bigger_than_the_frame_finds_nothing() -> None:
    t = enemy_template()
    tiny = Frame(8, 8, 3, bytes(8 * 8 * 3))
    assert SpriteFinder([t]).find(tiny) == ()


def test_channels_must_match_the_frame() -> None:
    grey = Frame(32, 32, 1, bytes(32 * 32))
    with pytest.raises(ConfigurationError):
        SpriteFinder([enemy_template()]).find(grey)


@pytest.mark.parametrize(
    "kwargs",
    [{"tolerance": -1}, {"tolerance": 256}, {"min_score": 0.0}, {"min_score": 1.5}, {"anchors": 0}],
)
def test_finder_settings_are_checked(kwargs: dict[str, float]) -> None:
    with pytest.raises(ConfigurationError):
        SpriteFinder([enemy_template()], **kwargs)  # type: ignore[arg-type]


def test_finder_needs_templates() -> None:
    with pytest.raises(ConfigurationError):
        SpriteFinder([])
    with pytest.raises(ConfigurationError):
        SpriteFinder(["not a template"])  # type: ignore[list-item]


def test_finder_does_not_change_the_frame() -> None:
    frame = scene([(10, 184, walker())])
    before = frame.pixels
    SpriteFinder([enemy_template()]).find(frame)
    assert frame.pixels == before


@settings(max_examples=40, deadline=None)
@given(
    x=st.integers(min_value=0, max_value=256 - 16),
    y=st.integers(min_value=0, max_value=240 - 16),
    flip=st.booleans(),
)
def test_found_wherever_it_is_put(x: int, y: int, flip: bool) -> None:
    spr = walker()[:, ::-1] if flip else walker()
    found = SpriteFinder([enemy_template()]).find(scene([(x, y, spr)]))
    assert [(m.x, m.y, m.mirrored) for m in found] == [(x, y, flip)]


# ------------------------------------------------------------ the practice console


def test_tags_an_obstacle_on_the_toy_racer_and_follows_it_down_the_screen() -> None:
    game = ToyRacer(seed=3)
    first = game.render()
    arr = np.frombuffer(first.pixels, np.uint8).reshape(first.height, first.width)
    from retro_ai_lab.emulator.fake.toy_racer import PX_OBSTACLE

    ys, xs = np.nonzero(arr == PX_OBSTACLE)
    if len(ys) == 0:  # pragma: no cover - seed 3 shows an obstacle at power-on
        pytest.fail("seed 3 should show an obstacle")
    y0, x0 = int(ys.min()), int(xs.min())
    width = int(xs[ys == y0].max()) - x0 + 1
    t = SpriteTemplate.from_frame(first, x0, y0, width, 2, name="block", tag=SpriteTag.ENEMY)
    finder = SpriteFinder([t], mirrored=False)
    assert (x0, y0) in [(m.x, m.y) for m in finder.find(first)]

    rows = []
    for _ in range(6):
        game.step(frozenset({"up"}))
        found = [m for m in finder.find(game.render()) if m.x == x0]
        rows.append(min((m.y for m in found), default=None))
    seen = [r for r in rows if r is not None]
    assert seen and seen[-1] > y0  # the obstacle comes towards the car


# ------------------------------------------------------------ cost


def test_fast_enough_for_every_step() -> None:
    """Four tags on a 256x240 colour screen with five enemies: well under a frame's time."""
    templates = [enemy_template()]
    for i in range(3):
        other = walker().copy()
        other[2:10, 3:13] = (i * 40, 200, 50)
        templates.append(
            SpriteTemplate.from_frame(
                scene([(0, 0, other)]), 0, 0, 16, 16, name=f"t{i}", tag=SpriteTag.ENEMY
            )
        )
    finder = SpriteFinder(templates)
    frame = scene([(x, 184, walker()) for x in (10, 60, 110, 160, 210)])
    finder.find(frame)
    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        finder.find(frame)
    per_frame = (time.perf_counter() - start) / runs
    assert per_frame < 0.03, f"{per_frame * 1000:.1f} ms per frame"


# ------------------------------------------------------------ the surroundings, not only enemies


def brick() -> np.ndarray:
    b = np.zeros((16, 16, 3), np.uint8)
    b[:] = BROWN
    b[0, :] = BLACK
    b[8, :] = BLACK
    b[0:8, 7] = BLACK
    b[8:16, 15] = BLACK
    return b


def test_ground_tiles_are_each_found_and_tagged_solid() -> None:
    """A row of ground: one hit per tile, next to each other, none merged away."""
    a = np.zeros((240, 256, 3), np.uint8)
    a[:] = SKY
    for x in range(0, 256, 16):
        a[208:224, x : x + 16] = brick()
        a[224:240, x : x + 16] = brick()
    frame = Frame(256, 240, 3, a.tobytes())
    t = SpriteTemplate.from_frame(frame, 0, 208, 16, 16, name="ground", tag=SpriteTag.SOLID)
    found = SpriteFinder([t], mirrored=False).find(frame)
    assert {(m.x, m.y) for m in found} == {(x, y) for x in range(0, 256, 16) for y in (208, 224)}
    assert all(m.tag is SpriteTag.SOLID for m in found)


def test_every_tag_has_child_words() -> None:
    from retro_ai_lab.observation.sprites import TAG_WORDS

    assert set(TAG_WORDS) == set(SpriteTag)
    assert all(w and w == w.lower() for w in TAG_WORDS.values())
