"""T-1013: a life lost that also reads as the game starting over is searched first."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.services import DiscoveryUpdated
from retro_ai_lab.training import EpisodeRunner
from retro_ai_lab.training.episode import NOW_SLOT
from retro_ai_lab.understanding.events import EventKind, GameEvent
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_search_at_a_loss import GAP_ENDS, searching

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

WENT_HOME = "Progress, Flag all went back to their starting values."
HELD = (
    "The game put the player back to the start when that life went, which used to end "
    "the go. The AI has gone back to before it and carries on."
)


class Restarts:
    """Reports a restart on the bus in the very step a life is seen to go, as the
    engine does once it knows the values that return to their start."""

    def __init__(self, lab: Lab, *, late: bool = False) -> None:
        self.lab = lab
        self.late = late  # report it with the step after the loss instead
        self.due = False
        self.told = 0
        lab.bus.subscribe(DiscoveryUpdated, self._on_update)

    def _on_update(self, update: DiscoveryUpdated) -> None:
        if any(e.kind is EventKind.SETBACK for e in update.events):
            if self.late:
                self.due = True
            else:
                self.tell()

    def tell(self) -> None:
        self.due = False
        self.told += 1
        reset = GameEvent(EventKind.RESET, self.lab.observation.frame, None, WENT_HOME)
        self.lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))


def test_a_life_lost_that_reads_as_a_restart_is_searched_first(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, moments = searching(lab)
    restarts = Restarts(lab)
    r.begin()
    steps(r, DIED_AT - 1)
    assert restarts.told == 0 and r.ways_sought == 0
    start = moments.kept[0]
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        assert r.step()  # the life goes, the restart is reported, and the go is not over
    assert restarts.told == 1 and r.last_reset.startswith(WENT_HOME)
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    assert lab.loaded[-1] == start.slot and lab.game.state.lives == toy_hopper.START_LIVES
    # The step of the loss was learned once, as the end of that story, with its penalty.
    assert r.steps == DIED_AT and len(lab.agent.learned) == DIED_AT
    *_, (_, action, reward, _, done, _) = lab.agent.learned
    assert (action, done) == (lab.move("right"), True) and reward <= -10.0
    said = [rec.getMessage() for rec in caplog.records]
    assert said.count(HELD) == 1 and said[-1] == HELD
    # The way is played and gets the player across with every life.
    while lab.game.state.x <= GAP_ENDS:
        assert r.step()
        assert lab.game.state.lives == toy_hopper.START_LIVES
    assert restarts.told == 1
    # Its one search is used. Walking on, it loses a life to the walker further along:
    # with nothing left to search with, that restart ends the go, as it always did.
    made = (list(lab.saved).count(NOW_SLOT), len(lab.loaded))
    while r.step():
        assert restarts.told == 1
    assert restarts.told == 2 and lab.game.state.lives == toy_hopper.START_LIVES - 1
    assert (list(lab.saved).count(NOW_SLOT), len(lab.loaded)) == made
    result = r.finish()
    assert (result.ended_by, result.setbacks) == ("reset", 2)
    assert (result.ways_sought, result.ways_found) == (1, 1) and result.steps > DIED_AT + 20


def test_with_no_way_found_the_go_ends_as_a_restart(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, _ = searching(lab, way_tries=3)
    restarts = Restarts(lab)
    r.begin()
    steps(r, DIED_AT - 1)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        assert not r.step()  # searched, nothing found: the restart stands
    assert restarts.told == 1
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 0, 0)
    # The game is where it was after the loss, a life down, and the go is over.
    assert lab.loaded[-1] == NOW_SLOT
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1
    assert not r.step()
    result = r.finish()
    assert (result.ended_by, result.steps, result.setbacks) == ("reset", DIED_AT, 1)
    assert r.last_reset.startswith(WENT_HOME)
    said = [rec.getMessage() for rec in caplog.records]
    assert HELD not in said and any("none got past" in line for line in said)
    # The step was learned once, as final.
    assert len(lab.agent.learned) == DIED_AT and lab.agent.learned[-1][4] is True


def test_where_nothing_can_be_searched_it_ends_as_before(lab: Lab) -> None:
    def falls_in(r: EpisodeRunner, at: int = DIED_AT) -> None:
        lab.game.reset()
        lab.saved.clear()
        lab.loaded.clear()
        r.begin()
        steps(r, at - 1)
        made = (list(lab.saved), list(lab.loaded))
        assert not r.step()
        # Nothing was saved or loaded for it, and nothing searched.
        assert (lab.saved, lab.loaded) == made and NOW_SLOT not in lab.saved
        result = r.finish()
        assert (result.ended_by, result.steps, result.ways_sought) == ("reset", at, 0)

    restarts = Restarts(lab)
    # No search allowed, and no moments kept.
    none_allowed, _ = searching(lab, ways_per_go=0)
    falls_in(none_allowed)
    falls_in(lab.runner(watch=lab.watch()))
    # The go's last step: it would end there anyway.
    last, _ = searching(lab, max_steps=DIED_AT)
    falls_in(last)
    assert restarts.told == 3
    # The runner has given up going back: its next go ends at the loss.
    r, _ = searching(lab, way_tries=1, give_up_after=1)
    r.begin()
    steps(r, DIED_AT - 1)
    assert not r.step() and r.ways_sought == 1 and r.gave_up
    assert r.finish().ended_by == "reset"
    falls_in(r)
    # (The go's searches used up: the end of the first test above.)


def test_a_restart_with_no_life_lost_ends_the_go(lab: Lab) -> None:
    r, _ = searching(lab)
    r.begin()
    steps(r, DIED_AT - 6)  # far enough for a moment to go back to, and no life lost
    assert r.ways_sought == 0
    made = (list(lab.saved), list(lab.loaded))
    reset = GameEvent(EventKind.RESET, lab.observation.frame, None, WENT_HOME)
    lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))
    assert not r.step()
    assert (lab.saved, lab.loaded) == made
    result = r.finish()
    assert (result.ended_by, result.ways_sought, result.setbacks) == ("reset", 0, 0)


def test_a_step_somebody_else_drove_is_not_held(lab: Lab) -> None:
    r, _ = searching(lab)
    restarts = Restarts(lab)
    r.begin()
    steps(r, DIED_AT - 4)
    # A person takes over for the last steps into the gap.
    alive = True
    for _ in range(12):
        lab.adapter.step(
            going_back.FRAMES, pressed=[b for b in lab.adapter.buttons() if b.name == "right"]
        )
        lab.observation.advance(going_back.FRAMES)
        alive = r.observe_step(["right"], going_back.FRAMES)
        if not alive:
            break
    assert not alive and restarts.told == 1
    result = r.finish()
    assert (result.ended_by, result.ways_sought, result.setbacks) == ("reset", 0, 1)
    assert NOW_SLOT not in lab.saved


def test_a_restart_a_step_after_the_loss(lab: Lab) -> None:
    # A way found: the game is back before the loss, so the restart the engine would
    # have reported a step later never happens.
    r, _ = searching(lab)
    restarts = Restarts(lab, late=True)
    r.begin()
    steps(r, DIED_AT)
    assert restarts.due and (r.ways_sought, r.ways_found) == (1, 1)
    restarts.due = False  # the game went back: there is no restart to report
    while lab.game.state.x <= GAP_ENDS:
        assert r.step()
    r.finish()
    # No way found: the loss stands, the restart comes with the next step and ends the go.
    lab.game.reset()
    r, _ = searching(lab, way_tries=3)
    r.begin()
    steps(r, DIED_AT)
    assert restarts.due and (r.ways_sought, r.ways_found) == (1, 0)
    restarts.tell()
    assert not r.step()
    result = r.finish()
    assert (result.ended_by, result.steps, result.ways_sought) == ("reset", DIED_AT + 1, 1)
