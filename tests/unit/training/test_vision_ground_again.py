"""T-1028: the watch hands over the ground a go walks again after it was put back."""

from __future__ import annotations

from retro_ai_lab.training.vision import ScreenWatch
from tests.unit.training import test_trial as trial
from tests.unit.training.test_trial import Hopper
from tests.unit.training.test_vision_camera_memory import a_watch, camera, frame_of
from tests.unit.training.test_vision_first_look_back import A_MOVE, on

hopper = trial.hopper  # the practice game behind its adapter (pytest fixture)


def walking(hopper: Hopper) -> ScreenWatch:
    """A watch of a walk to the right, far enough for the camera to follow."""
    watch = a_watch(hopper)
    watch.begin_go()
    watch.look(frame_of(hopper))
    on(hopper, watch, looks=6)
    assert watch.tracker.scroll == (A_MOVE, 0) and watch.follows_camera
    return watch


def test_the_watch_hands_over_ground_walked_again(hopper: Hopper) -> None:
    watch = walking(hopper)
    game, adapter = hopper.game, hopper.adapter
    unit = watch.ground.unit
    assert unit is not None and unit <= A_MOVE  # a look's move is a step at least
    assert watch.take_new_ground() > 0
    assert watch.take_ground_again() == 0  # played straight, nothing is walked again
    mark = watch.mark()
    adapter.save_state("here")
    here = camera(game)
    on(hopper, watch, looks=3)
    paid = watch.take_new_ground()
    assert paid == (here + 3 * A_MOVE) // unit - here // unit
    assert watch.take_ground_again() == 0
    # Put back, and on over the same ground: nothing new, and what the go had been
    # paid for it is handed over as it is walked, look by look.
    adapter.load_state("here")
    watch.back_to(mark)
    assert watch.take_ground_again() == 0
    again = []
    for _ in range(3):
        on(hopper, watch)
        assert watch.take_new_ground() == 0
        again.append(watch.take_ground_again())
        assert watch.take_ground_again() == 0  # given once
    assert sum(again) == paid and all(again)
    # Past where it had been: new ground, and nothing walked again.
    on(hopper, watch, looks=2)
    assert watch.take_new_ground() > 0 and watch.take_ground_again() == 0

    # A try from the mark counts on its own copy: what it walks again is not the go's.
    adapter.load_state("here")
    other = watch.trial(mark)
    on(hopper, other, looks=2)
    assert other.take_new_ground() == 0  # what the try would be paid (T-1007)
    # (A try's tracker starts afresh, so its first look measures no move.)
    assert 0 < other.take_ground_again() <= again[0] + again[1]
    assert watch.take_ground_again() == 0

    # What nobody took adds up over the looks, and a new go starts with none of it.
    adapter.load_state("here")
    watch.back_to(mark)
    on(hopper, watch, looks=2)
    assert watch.take_ground_again() == again[0] + again[1]
    adapter.load_state("here")
    watch.back_to(mark)
    on(hopper, watch, looks=2)
    watch.begin_go()
    assert watch.take_ground_again() == 0 and watch.take_new_ground() == 0
