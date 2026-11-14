"""T-1023: the tracker can start from a screen the caller kept, so that the first
look after a game is put back to an earlier moment is measured like any other."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.tracking import ObjectTracker
from tests.unit.observation.test_tracking import hero, level, view, walker

WORLD = level()


def screen(camera: int, number: int, *, with_hero: bool = False) -> Frame:
    """The camera moves 16 a look; a walker comes the other way at 4. The hero, where
    it is drawn, is held at one place on the screen."""
    things = [(500 - number * 4, 192, walker())]
    if with_hero:
        things.append((camera + 100, 180, hero()))
    return view(WORLD, camera, things, number)


def test_the_tracker_starts_from_a_screen() -> None:
    t = ObjectTracker(max_shift=32)
    for i in range(4):
        t.update(screen(300 + 16 * i, i))
    assert t.scroll == (16, 0) and t.camera == (48, 0)
    followed = t.tracks()
    assert len(followed) == 1 and followed[0].kind is not None  # the walker

    # Started afresh with nothing, the next look has no screen to be measured
    # against: whatever the camera did, no move is read. (As it was.)
    t.reset()
    assert t.tracks() == ()
    t.update(screen(364, 4))
    assert (t.scroll, t.sighted, t.unexplained, t.tracks()) == ((0, 0), (0, 0), 0.0, ())

    # Started from the screen of the moment gone back to: what was followed is
    # forgotten and no move is measured, as after a first look ...
    for i in range(5, 8):
        t.update(screen(300 + 16 * i, i))
    assert len(t.tracks()) == 1 and t.camera != (0, 0)
    assert t.start_from(screen(300, 0)) == ()
    assert (t.scroll, t.sighted, t.unexplained, t.camera) == ((0, 0), (0, 0), 0.0, (0, 0))
    assert t.tracks() == ()
    # ... and the next look is measured against it: the camera's 16, and the walker
    # found moving by its own 4 with a kind the tracker knew before.
    t.update(screen(316, 1))
    assert (t.scroll, t.sighted, t.camera) == ((16, 0), (16, 0), (16, 0))
    found = t.tracks()
    assert len(found) == 1 and found[0].kind == followed[0].kind
    assert found[0].id != followed[0].id

    # What is hidden on that screen is known to have been hidden when the next look
    # is measured: the player, which the caller follows, is no thing and leaves no
    # trail, as at any other look.
    box = (99, 179, 14, 18)
    t.start_from(screen(300, 0, with_hero=True), hide=box)
    t.update(screen(316, 1, with_hero=True), hide=box)
    assert t.scroll == (16, 0) and len(t.tracks()) == 1
    plain = ObjectTracker(max_shift=32)
    plain.start_from(screen(300, 0, with_hero=True))
    plain.update(screen(316, 1, with_hero=True))
    assert plain.scroll == (16, 0)

    # What is no screen, or no box, is refused, and nothing is forgotten for it.
    before = t.tracks()
    with pytest.raises(ConfigurationError, match="needs a Frame"):
        t.start_from("a screen")  # type: ignore[arg-type]
    for nonsense in ((1, 2, 3), (0, 0, 0, 5), (0, 0, 5, 1.5), (True, 0, 5, 5)):
        with pytest.raises(ConfigurationError, match="hide must be a box"):
            t.start_from(screen(300, 0), hide=nonsense)  # type: ignore[arg-type]
    assert t.tracks() == before and t.scroll == (16, 0)
