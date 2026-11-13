"""T-1023: the first look after the game is put back to a saved moment is measured
against the screen of that moment, where it used to be measured against nothing."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.tracking import ObjectTracker
from retro_ai_lab.training.vision import HIDE_MARGIN, Mark, ScreenWatch
from retro_ai_lab.understanding import Confidence
from retro_ai_lab.understanding.player import PlayerFinding
from tests.unit.training.test_trial import FRAMES, Hopper
from tests.unit.training.test_vision_camera_memory import (
    STEP,
    Spy,
    a_watch,
    camera,
    frame_of,
    memory_of,
)

LOOK_EVERY = 2  # steps between two looks, as in a go
A_MOVE = 16  # what the camera moves between two looks once it follows the player


def on(hopper: Hopper, *watches: ScreenWatch, looks: int = 1, memory: bool = False) -> Frame:
    """Hold right for so many looks' worth of steps, every watch looking each time."""
    right = [b for b in hopper.adapter.buttons() if b.name == "right"]
    frame = frame_of(hopper)
    for _ in range(looks):
        hopper.adapter.step(FRAMES * LOOK_EVERY, pressed=right)
        frame = frame_of(hopper)
        for watch in watches:
            watch.look(frame, memory=memory_of(hopper.game) if memory else None)
    return frame


def walking(hopper: Hopper, *, memory: bool = False) -> tuple[ScreenWatch, ScreenWatch]:
    """Two watches of one walk to the right, far enough for the camera to follow."""
    new = a_watch(hopper)
    old = ScreenWatch(new.player)
    for watch in (new, old):
        watch.begin_go()
        watch.look(frame_of(hopper), memory=memory_of(hopper.game) if memory else None)
    on(hopper, new, old, looks=6, memory=memory)  # (the level's first stretch is short)
    assert new.tracker.scroll == old.tracker.scroll == (A_MOVE, 0)
    return new, old


def test_a_mark_carries_the_screen_it_was_taken_at(hopper: Hopper) -> None:
    watch = a_watch(hopper)
    watch.begin_go()
    # Before the go's first look there is no screen to carry.
    first = watch.mark()
    assert (first.screen, first.hide) == (None, None)
    frame = frame_of(hopper)
    watch.look(frame)
    mark = watch.mark()
    assert watch.me is not None
    x, y, w, h = watch.me
    assert mark.screen is frame
    assert mark.hide == (x - HIDE_MARGIN, y - HIDE_MARGIN, w + 2 * HIDE_MARGIN, h + 2 * HIDE_MARGIN)
    # It is the last screen looked at, whatever the game did since.
    later = on(hopper, watch, looks=3)
    hopper.adapter.step(FRAMES, pressed=[])
    assert watch.mark().screen is later and watch.mark().screen is not frame
    assert mark.screen is frame  # a mark is what it was
    # A new go: nothing is carried over from the last.
    watch.begin_go()
    assert (watch.mark().screen, watch.mark().hide) == (None, None)
    # A mark made by hand in the old way is still a mark, with no screen.
    plain = Mark(mark.ground, mark.last_me)
    assert (plain.screen, plain.hide) == (None, None)
    watch.back_to(plain)
    with pytest.raises(ConfigurationError, match="Only a moment this watch marked"):
        watch.back_to(frame)  # type: ignore[arg-type]

    # A watch that knows the player by no colours hides nothing, and carries the
    # screen all the same.
    blind = ScreenWatch(PlayerFinding(None, (46, 200), frozenset({"a"}), 2, Confidence.MEDIUM))
    blind.begin_go()
    frame = frame_of(hopper)
    blind.look(frame)
    assert (blind.mark().screen, blind.mark().hide) == (frame, None)
    assert blind.mark().screen is frame


def test_the_first_look_after_going_back_is_measured(hopper: Hopper) -> None:
    new, old = walking(hopper)
    game, adapter = hopper.game, hopper.adapter
    mark = new.mark()
    adapter.save_state("here")
    here = camera(game)
    assert new.ground.reach[0] == old.ground.reach[0] == here
    on(hopper, new, old, looks=2)
    assert camera(game) == here + 2 * A_MOVE
    # Put back, and on: the look after is measured against the screen of the moment
    # gone back to, and is the camera's own move ...
    adapter.load_state("here")
    new.back_to(mark)
    old.back_to(Mark(mark.ground, mark.last_me))  # as it was: the tracker afresh
    assert new.ground.reach[0] == here + 2 * A_MOVE  # what was reached stays reached
    on(hopper, new, old)
    assert camera(game) == here + A_MOVE
    assert (new.tracker.scroll, new.tracker.unexplained < 0.02) == ((A_MOVE, 0), True)
    assert new.ground.place == (here + A_MOVE, 0)
    # ... where it used to read as no move at all.
    assert old.tracker.scroll == (0, 0) and old.ground.place == (here, 0)
    # On past where it had been: the ground counted is the game's own. The old way
    # is a move short.
    on(hopper, new, old, looks=4)
    gone = camera(game)
    assert gone == here + 5 * A_MOVE
    assert new.ground.reach[0] == gone
    assert old.ground.reach[0] == gone - A_MOVE
    # ... and stays so however often it goes back.
    mark = new.mark()
    adapter.save_state("here")
    on(hopper, new, old)
    adapter.load_state("here")
    new.back_to(mark)
    on(hopper, new)
    assert new.ground.place == (camera(game), 0) and new.ground.reach[0] == gone + A_MOVE
    assert A_MOVE == STEP


def test_showing_the_screen_counts_as_nothing(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    new, old = walking(hopper, memory=True)
    adapter = hopper.adapter
    mark = new.mark()
    adapter.save_state("here")
    on(hopper, new, old, looks=3, memory=True)
    new.take_new_ground()
    looks, seen, kinds = new.looks, new.seen_me, new.describe()
    assert new.around is not None and new.me is not None
    spy = Spy(monkeypatch)
    started: list[tuple[Frame, object]] = []
    start_from = ObjectTracker.start_from

    def starting(tracker: ObjectTracker, frame: Frame, *, hide: object = None) -> object:
        started.append((frame, hide))
        return start_from(tracker, frame, hide=hide)  # type: ignore[arg-type]

    monkeypatch.setattr(ObjectTracker, "start_from", starting)
    adapter.load_state("here")
    new.back_to(mark)
    # The tracker starts from the mark's screen, with what was hidden in it (the
    # player, which the watch follows itself) ...
    assert len(started) == 1 and started[0][0] is mark.screen
    assert started[0][1] == mark.hide and mark.hide is not None
    # ... which is no look of the tracker's, and shows nothing to the camera's memory.
    assert spy.since() == (0, [])
    assert (new.looks, new.seen_me, new.take_new_ground()) == (looks, seen, 0)
    assert new.describe() == kinds
    # What going back dropped it still drops: the player is looked for again, where
    # the mark last saw it.
    assert new.around is None and new.me is None
    assert new.tracker.scroll == (0, 0)
    # And the mark can be gone back to again and again.
    on(hopper, new, looks=2, memory=True)
    adapter.load_state("here")
    new.back_to(mark)
    on(hopper, new, memory=True)
    assert new.tracker.scroll == (A_MOVE, 0)


def test_a_mark_from_before_any_look(hopper: Hopper) -> None:
    watch = a_watch(hopper)
    watch.begin_go()
    hopper.adapter.save_state("start")
    mark = watch.mark()
    watch.look(frame_of(hopper))
    on(hopper, watch, looks=7)
    assert watch.tracker.scroll == (A_MOVE, 0)
    # Back to before the go's first look: there is no screen, and the tracker starts
    # afresh as it always did. The next look measures no move.
    hopper.adapter.load_state("start")
    watch.back_to(mark)
    assert watch.mark().screen is None
    watch.look(frame_of(hopper))
    assert watch.tracker.scroll == (0, 0)
    on(hopper, watch, looks=7)
    assert watch.tracker.scroll == (A_MOVE, 0)


def test_the_games_memory_is_not_read_across(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    new, _ = walking(hopper, memory=True)
    adapter = hopper.adapter
    spy = Spy(monkeypatch)
    on(hopper, new, memory=True)
    assert spy.since() == (1, [(None, None)])  # learnt from; no byte found yet to tell
    mark = new.mark()
    adapter.save_state("here")
    on(hopper, new, looks=2, memory=True)
    adapter.load_state("here")
    new.back_to(mark)
    spy.since()
    # The first look after: measured, by the picture alone. The tracker is told
    # nothing from the game's memory and nothing is learnt of the camera's byte.
    on(hopper, new, memory=True)
    assert new.tracker.scroll == new.tracker.sighted == (A_MOVE, 0)
    assert spy.since() == (0, [None])
    # From the second on, as ever.
    on(hopper, new, memory=True)
    assert spy.since() == (1, [(None, None)])


def test_a_try_starts_afresh_as_before(hopper: Hopper) -> None:
    new, _ = walking(hopper)
    adapter = hopper.adapter
    mark = new.mark()
    adapter.save_state("here")
    looks, place, reach = new.looks, new.ground.place, new.ground.reach
    # A try from the mark: its watch has no screen from before, and its first look
    # measures no move (out of this task's scope; a try is paid as it was).
    other = new.trial(mark)
    assert other.mark().screen is None
    on(hopper, other)
    assert other.tracker.scroll == (0, 0)
    on(hopper, other)
    assert other.tracker.scroll == (A_MOVE, 0)
    # Nothing of it reaches the watch it came from.
    assert (new.looks, new.ground.place, new.ground.reach) == (looks, place, reach)
    assert new.mark() == mark
    adapter.load_state("here")
    new.back_to(mark)
    on(hopper, new)
    assert new.tracker.scroll == (A_MOVE, 0)
