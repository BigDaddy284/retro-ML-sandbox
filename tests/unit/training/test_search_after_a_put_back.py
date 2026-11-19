"""T-1062: a search after the game has put the player back finds a way past on ground
the go has walked.

The go is on the practice hopper behind the real services. A learner that only walks
right loses its first life in the first gap and is taken across by a search, loses
it again at the first walker and is taken past by a second. A third search is made
to find nothing, so the life goes and the game puts the player back at the start.
The learner is walked over the first gap (as the way remembered would walk it) and
comes to the walker a second time, on ground the go has been 500 pixels beyond.

Before this task every try there got across alive and none was the way: the way past
was ground the go had not been on. It is now ground the life has not been on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import START_LIVES
from retro_ai_lab.observation.ground import Ground
from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.search import BUDGET, Search
from retro_ai_lab.training.trial import PAST
from retro_ai_lab.training.vision import ScreenWatch
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import Lab
from tests.unit.training.test_last_chance_in_a_go import GAP_ENDS, LONG, going
from tests.unit.training.test_looking_ahead import THING, Near

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LOG = "retro_ai_lab.training.episode"
OWN = toy_hopper.DYING_FRAMES  # the practice game's own dying pause
OVER_THE_GAP = 19  # steps walked from the start before the jump that clears the first gap
PAST_THE_WALKER = 490  # level pixels: the first walker paces from 416 to 466
SEES_THE_WALKER = 330  # where the watch is made to say that the walker is near
# The two ways a go is run: the practice game's own short pause with the search as it
# first was, and a pause as long as a console game's with the step the loss was
# settled at asked for, as the app runs it.
RUNS = [(OWN, False), (LONG, True)]


@dataclass
class Told:
    """What a go of the kind above came to."""

    runner: EpisodeRunner
    searches: list[Search]
    put_back_at: int  # the go's step at which the game put the player back
    counted: int  # steps of new ground the go had counted by then
    far: int  # how far the first life got, in level pixels
    near: Near | None = None  # what the watch says is near, where the go looks ahead

    @property
    def watch(self) -> ScreenWatch:
        watch = self.runner._watch
        assert watch is not None
        return watch


def a_second_life(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    frames: int,
    chance: bool,
    *,
    held_off: bool = False,
    by_hand: bool = True,
    searches: int = 4,
    wait: int = 0,
    look_ahead: bool = False,
) -> Told:
    """Play the go until its fourth search is over (or as many as ``searches``). With
    ``wait`` the learner stands that many steps before it walks: the life then goes
    on another step, and a look comes every other step."""
    if held_off:
        monkeypatch.setattr(Ground, "new_life", lambda self: None)
    real = episode.find_way
    made: list[Search] = []

    def find(*a: object, **kw: object) -> Search:
        nothing = len(made) == 2  # the third search: the first life goes for good
        made.append(Search(None, 0) if nothing else real(*a, **kw))  # type: ignore[arg-type]
        return made[-1]

    monkeypatch.setattr(episode, "find_way", find)
    r, _ = going(
        lab, frames, ways_per_go=9, max_steps=1500, last_chance=chance, look_ahead=look_ahead
    )
    near = Near(monkeypatch, r.watch) if look_ahead else None  # nothing near, until told
    lab.agent.script.extend([lab.move()] * wait)
    r.begin()
    far = 0
    while lab.game.state.lives == START_LIVES:
        far = max(far, lab.game.state.x)
        assert r.step()
    assert len(made) == 3 and lab.game.state.lives == START_LIVES - 1
    watch = r._watch
    assert watch is not None
    told = Told(r, made, r.steps, watch.ground.steps, far, near)
    if by_hand:
        walk, jump = lab.move("right"), lab.move("a", "right")
        lab.agent.script.extend([walk] * OVER_THE_GAP + [jump] * 4)
    while len(made) < searches:
        assert r.step()
    return told


def past_the_walker(lab: Lab, r: EpisodeRunner) -> None:
    """Play on until the player is past where the first walker paces."""
    for _ in range(120):
        if lab.game.state.x > PAST_THE_WALKER:
            return
        assert r.step()
    raise AssertionError("it did not get past the walker")


@pytest.mark.parametrize(("frames", "chance"), RUNS)
def test_the_second_life_is_taken_past_what_the_first_was(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, frames: int, chance: bool
) -> None:
    told = a_second_life(lab, monkeypatch, frames, chance)
    gap, walker, nothing, again = told.searches
    assert gap.way is not None and walker.way is not None and nothing.way is None
    # The first life got well past the walker before it went.
    assert told.far > 500
    # At the walker again, on ground the go has been beyond: a way past, of the same
    # kind as the first life's (a jump of the same length), and the life is kept.
    assert again.way is not None and again.tries < BUDGET
    assert again.way.change.move == walker.way.change.move == lab.move("a", "right")
    assert again.way.change.hold == walker.way.change.hold
    assert again.way.ground >= PAST
    r = told.runner
    assert lab.game.state.lives == START_LIVES - 1
    assert (r.ways_sought, r.ways_found) == (4, 3)
    # Played, it gets the player past the walker alive.
    past_the_walker(lab, r)
    assert lab.game.state.lives == START_LIVES - 1 and len(told.searches) == 4
    assert lab.game.state.mode == toy_hopper.PLAYING


@pytest.mark.parametrize(("frames", "chance"), RUNS)
def test_with_the_lifes_count_held_off_every_try_is_made_for_nothing(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    frames: int,
    chance: bool,
) -> None:
    """The fault this task mends: the same go, with the ground counted for the go
    alone. Every change is tried, and the life goes at the walker a second time."""
    caplog.set_level(logging.INFO, logger=LOG)
    told = a_second_life(lab, monkeypatch, frames, chance, held_off=True)
    gap, walker, nothing, again = told.searches
    assert gap.way is not None and walker.way is not None and nothing.way is None
    assert again.way is None and again.tries > 100 and not again.out_of_time
    assert lab.game.state.lives == START_LIVES - 2
    assert (told.runner.ways_sought, told.runner.ways_found) == (4, 2)
    said = [r.getMessage() for r in caplog.records if "none got past" in r.getMessage()]
    assert len(said) == 2 and f"made {again.tries} tries" in said[1]


@pytest.mark.parametrize("wait", [0, 1])
@pytest.mark.parametrize(("frames", "chance"), RUNS)
def test_the_searches_of_the_first_life_are_what_they_were(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, frames: int, chance: bool, wait: int
) -> None:
    """The practice game puts the player back in the step the life goes in, and on
    every other step the watch has looked at that before the search begins. The
    tries of that search are in the life that went, whatever the watch has seen."""
    with_it = a_second_life(lab, monkeypatch, frames, chance, searches=3, wait=wait)
    first = [(s.way, s.tries, s.stopped_at, s.lapped) for s in with_it.searches]
    at = (with_it.put_back_at, with_it.counted, with_it.far)
    assert all(way is not None for way, *_ in first[:2])
    lab.agent.script.clear()
    without = a_second_life(lab, monkeypatch, frames, chance, searches=3, wait=wait, held_off=True)
    assert [(s.way, s.tries, s.stopped_at, s.lapped) for s in without.searches] == first
    assert (without.put_back_at, without.counted, without.far) == at


@pytest.mark.parametrize(("frames", "chance"), RUNS)
def test_ground_walked_again_is_not_paid_again(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, frames: int, chance: bool
) -> None:
    told = a_second_life(lab, monkeypatch, frames, chance)
    watch, r = told.watch, told.runner
    # From the put-back to the walker and past it, on the way the search found: the
    # go has counted no step of new ground it had not counted when the life went.
    assert told.searches[3].way is not None
    past_the_walker(lab, r)
    assert lab.game.state.lives == START_LIVES - 1
    assert GAP_ENDS < PAST_THE_WALKER < lab.game.state.x < told.far
    assert watch.ground.steps == told.counted
    assert watch.ground.place[0] < watch.ground.reach[0]


def test_the_log_says_the_game_put_it_back_and_then_that_a_way_was_found(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    a_second_life(lab, monkeypatch, LONG, True)
    lines = [r.getMessage() for r in caplog.records]
    put_back = [n for n, line in enumerate(lines) if "the game put the AI back" in line]
    found = [n for n, line in enumerate(lines) if "found a way past" in line]
    assert len(put_back) == 1 and len(found) == 3
    assert found[1] < put_back[0] < found[2]
    assert "Lost a life at step" in lines[found[2]]


@pytest.mark.parametrize("held_off", [False, True])
@pytest.mark.parametrize(("frames", "chance"), RUNS)
def test_a_life_seen_coming_on_walked_ground_is_saved(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, frames: int, chance: bool, held_off: bool
) -> None:
    """The go looks ahead when a thing of a kind that has cost a life comes near
    (T-1052). In its second life it comes to the walker again, sees the life going,
    and looks for a way past before it goes: on ground the go has walked."""
    told = a_second_life(
        lab, monkeypatch, frames, chance, held_off=held_off, searches=3, look_ahead=True
    )
    r, near = told.runner, told.near
    assert near is not None and r.looked_ahead == 0
    while lab.game.state.x < SEES_THE_WALKER:
        assert r.step()
    assert lab.game.state.lives == START_LIVES - 1 and len(told.searches) == 3
    near.now = THING
    assert r.step()
    near.now = ()
    assert r.looked_ahead == 1 and len(told.searches) == 4
    assert r.ways_sought == 3  # none of the go's searches after a life lost was used
    ahead = told.searches[3]
    if held_off:
        # Every change tried, none of them the way: it carries on and the life goes.
        assert ahead.way is None and ahead.tries > 100
        assert (r.ways_ahead, r.ways_ahead_found) == (1, 0)
        while lab.game.state.lives == START_LIVES - 1:
            assert r.step()
        assert lab.game.state.x < told.far
    else:
        assert ahead.way is not None and ahead.tries < BUDGET
        assert ahead.way.change.move == lab.move("a", "right")
        assert (r.ways_ahead, r.ways_ahead_found) == (1, 1)
        past_the_walker(lab, r)
        assert lab.game.state.lives == START_LIVES - 1
        assert lab.game.state.mode == toy_hopper.PLAYING
        # None of it was paid for a second time.
        assert told.watch.ground.steps == told.counted
