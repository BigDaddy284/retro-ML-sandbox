"""T-1066: the game's pause over a lost life is not taken for a standstill.

A game takes its time over a lost life: the screen stands, nothing listens, and only
then does the count go down. With a pause as long as a real game's the go took that
for a dead screen: it looked for a way on from the middle of the pause and pressed the
wake button, which forgets every saved moment, and the life then went with no search.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import pytest

from retro_ai_lab.core import EmulatorDisconnectedError
from retro_ai_lab.core.worklog import CallStats, app_work
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT
from retro_ai_lab.learning import ActionSpace
from retro_ai_lab.services import DiscoveryUpdated
from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.episode import AHEAD, DEFAULT_UNSTICK_AFTER, LATE, NOW_SLOT
from retro_ai_lab.training.moments import Moments
from retro_ai_lab.training.trial import Outcome
from retro_ai_lab.understanding import VariableKind
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import SURE, Lab, Scripted, steps
from tests.unit.training.test_last_chance_in_a_go import (
    GAP_ENDS,
    LONG,
    LONG_DIED,
    SIX,
    going,
    said,
)
from tests.unit.training.test_standing_still import Asked

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LOG = "retro_ai_lab.training.episode"
STUCK = DEFAULT_UNSTICK_AFTER  # steps of nothing changing before a go counts as stuck
# Walking right from the start with a real game's pause, what this lab's learner sees
# (the place and the lives) has stood for 25 steps once this many are played: it is in
# the pause by then, and the life goes 36 steps on.
ASKED_AT = 38
TO_GO = LONG_DIED - ASKED_AT
DYING = 53  # steps this game takes over a lost life with that pause, as the runner measures it
WAITS = (
    "{lead} for {stayed} steps. Played on off the record from here, a life is lost {on} "
    "steps on whatever is pressed: this is the game's pause over a life already lost, and "
    "no standstill. The AI waits for the life to be counted and then goes back to look for "
    "the way past."
)
# A go of the AI's own making: a few steps right and then nothing. What the learner
# sees last changes on the fifth step, so it has stood for 25 steps once 30 are played
# (and on main the search where it stands, the wake press and the end as "stuck" come
# in the step after those).
OWN = 6
OWN_AT = 5 + STUCK


def paused(lab: Lab, frames: int = LONG, **kw: object) -> tuple[EpisodeRunner, Moments]:
    """A go about to start in a game with a long dying pause, with the stuck rule on
    as the app has it, that keeps moments and may search."""
    kw.setdefault("unstick_after", STUCK)
    kw.setdefault("ways_per_go", 3)
    kw.setdefault("last_chance", True)
    return going(lab, frames, **kw)


class Plays:
    """Notes the plays the runner makes off the record from where the game is, and
    answers them from a list where it is given one (the real play otherwise)."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, answers: Sequence[Outcome] | None = None
    ) -> None:
        self.made: list[tuple[int, ...]] = []
        self.answers = None if answers is None else list(answers)
        self._real = episode.play
        monkeypatch.setattr(episode, "play", self)

    def __call__(
        self, connection: object, slot: str, moves: Sequence[int], *a: object, **kw: object
    ) -> Outcome:
        assert slot == NOW_SLOT and kw.get("watch") is None
        self.made.append(tuple(moves))
        if self.answers is None:
            return self._real(connection, slot, moves, *a, **kw)  # type: ignore[arg-type]
        return self.answers.pop(0)


def standing_still(lab: Lab, moves: int = 400) -> None:
    """The learner walks a few steps right and then does nothing."""
    lab.agent.script.extend([lab.move("right")] * OWN + [lab.move()] * moves)


# ------------------------------------------------------------------ the pause


@pytest.mark.parametrize("wake", [(), ("start",)], ids=["no wake button", "a wake button"])
@pytest.mark.parametrize("stand", [0, 60], ids=["no search at a standstill", "with one"])
def test_the_pause_is_waited_out_and_the_search_follows_the_loss(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    wake: tuple[str, ...],
    stand: int,
) -> None:
    assert (STUCK, AHEAD, LATE, TO_GO) == (25, 40, 4, 36)
    r, moments = paused(lab, stand_after=stand)
    r.set_unstick_buttons(wake)
    plays = Plays(monkeypatch)
    right, nothing = lab.move("right"), lab.move()
    r.begin()
    steps(r, ASKED_AT)
    assert NOW_SLOT not in lab.saved and not plays.made and lab.loaded == []
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
        # Nothing had changed for 25 steps: it asked. The game was saved, the plain move
        # held for 80 steps lost the life on the 36th, nothing held lost it on the 36th
        # too, and the game was put back.
        assert lab.saved.count(NOW_SLOT) == 1 and lab.loaded == [NOW_SLOT] * 3
        assert plays.made == [(right,) * (2 * AHEAD), (nothing,) * TO_GO]
        assert said(caplog) == [WAITS.format(lead="Nothing has changed", stayed=STUCK, on=TO_GO)]
        # It waits: no search where it stands, no wake press, and the go goes on.
        steps(r, LONG_DIED - 1 - r.steps)
        assert (r.ways_sought, r.unsticks, r.times_back) == (0, 0, 0)
        assert len(plays.made) == 2 and lab.loaded == [NOW_SLOT] * 3  # asked the once
        assert lab.game.state.lives == toy_hopper.START_LIVES and len(said(caplog)) == 1
        # The life goes on the step it would have gone on, and is searched for as any.
        assert r.step()
    assert (r.ways_sought, r.ways_found, r.ways_stood, r.times_back) == (1, 1, 0, 1)
    assert moments.steps == 0  # back at the start, before the edge
    settled, found = said(caplog)[1:]
    assert settled.startswith(f"Lost a life at step {LONG_DIED}. That was settled {DYING} steps")
    assert found.startswith(f"Lost a life at step {LONG_DIED}. Went back {LONG_DIED} steps and")
    asked = lab.agent.asked
    while lab.game.state.x <= GAP_ENDS:
        assert r.step()
    assert lab.agent.asked == asked and lab.game.state.lives == toy_hopper.START_LIVES
    result = r.finish()
    assert (result.setbacks, result.first_setback_step) == (1, LONG_DIED)
    assert (result.ways_sought, result.ways_found, r.unsticks) == (1, 1, 0)
    assert result.ended_by == "stopped"


def test_the_asking_runs_flat_out_and_is_timed_for_the_log(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    save, load = lab.adapter.save_state, lab.adapter.load_state
    monkeypatch.setattr(lab.adapter, "save_state", lambda s: (order.append(f"save {s}"), save(s)))
    monkeypatch.setattr(lab.adapter, "load_state", lambda s: (order.append(f"load {s}"), load(s)))

    @contextmanager
    def flat_out() -> Iterator[None]:
        order.append("fast")
        try:
            yield
        finally:
            order.append("normal")

    r, _ = paused(lab, flat_out=flat_out)
    r.begin()
    steps(r, ASKED_AT)
    assert "fast" not in order
    order.clear()
    timed = app_work.stats.get("step: pause asked", CallStats()).count
    assert r.step()
    # Saved, played twice and put back, all before the speed is what it was again.
    assert order == ["fast", f"save {NOW_SLOT}", *[f"load {NOW_SLOT}"] * 3, "normal"]
    assert app_work.stats["step: pause asked"].count == timed + 1
    steps(r, 5)  # waiting costs no more of either
    assert "fast" not in order[1:] and app_work.stats["step: pause asked"].count == timed + 1


def test_a_pause_in_which_only_the_spot_stands_is_waited_out_too(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """What the learner sees keeps changing in the pause (a clock that never stops), so
    the stuck rule has nothing to say: the go has stayed about one spot for as long as a
    standstill takes."""
    lab.teach()
    clock = lab.knowledge.propose(
        "ram", LAYOUT.timer, kind=VariableKind.POSITION, name="Clock", evidence=SURE
    )
    lab.bus.publish(DiscoveryUpdated((clock,), (), (), ()))
    r, moments = paused(lab, stand_after=30)
    plays = Plays(monkeypatch)
    right, nothing = lab.move("right"), lab.move()
    r.begin()
    with caplog.at_level(logging.INFO, logger=LOG):
        while not plays.made:
            assert r.step()
        at = r.steps - 1  # steps played when it asked
        assert at < LONG_DIED - 1
        assert plays.made == [(right,) * (2 * AHEAD), (nothing,) * (LONG_DIED - at)]
        (line,) = said(caplog)
        stood = re.fullmatch(
            re.escape(
                WAITS.format(lead="Stayed about one spot", stayed="N", on=LONG_DIED - at)
            ).replace("N", r"(\d+)"),
            line,
        )
        assert stood is not None and int(stood[1]) >= 30
        steps(r, LONG_DIED - 1 - r.steps)
        assert (r.ways_sought, r.ways_stood, r.times_back, len(plays.made)) == (0, 0, 0, 2)
        assert r.step()
    assert (r.ways_sought, r.ways_found, r.ways_stood, r.times_back) == (1, 1, 0, 1)
    assert moments.steps == 0 and lab.game.state.lives == toy_hopper.START_LIVES


def test_the_play_is_as_long_as_the_pause_it_remembers_and_says_so_once(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    r, _ = paused(lab)
    plays = Plays(monkeypatch)
    right, nothing = lab.move("right"), lab.move()
    r.begin()
    with caplog.at_level(logging.DEBUG, logger=LOG):
        steps(r, LONG_DIED)
        assert r.dying_steps == DYING and len(plays.made) == 2  # measured at the search
        # The way found is played and the go walks on into the second gap. A list of
        # moves is played to its end before anything is asked: the pause there is
        # asked about when the way is done.
        while len(plays.made) == 2:
            assert r.step()
        waits = [x for x in caplog.records if "the game's pause" in x.getMessage()]
    first, second = plays.made[2:]
    assert first == (right,) * (AHEAD + DYING)  # 40 steps and the pause it remembers
    assert second == (nothing,) * len(second) and 0 < len(second) < TO_GO
    # Said once where a person sees it, and after that in the detailed log.
    assert [x.levelno for x in waits] == [logging.INFO, logging.DEBUG]
    assert waits[1].getMessage().endswith("goes back to look for the way past.")
    assert r.unsticks == 0 and r.ways_stood == 0


def test_a_longer_pause_than_forty_steps_is_seen_with_none_remembered(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, _ = paused(lab, 255)  # the longest pause the practice game can take
    r.set_unstick_buttons(("start",))
    plays = Plays(monkeypatch)
    r.begin()
    steps(r, ASKED_AT + 1)
    assert r.dying_steps is None
    assert [len(made) for made in plays.made] == [2 * AHEAD, 50]  # 50 steps on: more than 40
    while r.ways_sought == 0:
        assert r.step()
    assert (r.ways_found, r.unsticks, r.steps) == (1, 0, ASKED_AT + 50)


@pytest.mark.parametrize(
    ("moves", "second"),
    [
        ([["right"], ["a", "right"], ["b", "up"]], ("b", "up")),
        ([["right"], ["a", "right"]], ("a", "right")),
        ([["right"]], None),
    ],
    ids=[
        "the first move shares a button",
        "no move without the plain one's button",
        "one move in all",
    ],
)
def test_the_second_play_is_of_whatever_other_move_there_is(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    moves: list[list[str]],
    second: tuple[str, ...] | None,
) -> None:
    """The other move is the first that shares no button with the plain one; with none
    such, the first other; and with one move in all, the one play is all there is."""
    lab.space = ActionSpace(moves)
    lab.agent = Scripted(lab.space)
    r, _ = paused(lab)
    r.set_unstick_buttons(("start",))
    plays = Plays(monkeypatch)
    r.begin()
    steps(r, LONG_DIED - 1)
    made = [(lab.move("right"),) * (2 * AHEAD)]
    if second is not None:
        made.append((lab.move(*second),) * TO_GO)
    assert plays.made == made and (r.unsticks, r.ways_sought) == (0, 0)
    assert r.step() and (r.unsticks, r.ways_sought) == (0, 1)  # the life goes, and is searched


def test_the_next_pause_is_asked_about_after_a_search_for_nothing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The life seen going has gone: what was known of it is not held against the next."""
    r, moments = paused(lab)
    r.set_unstick_buttons(("start",))
    plays = Plays(monkeypatch)
    asked = Asked(monkeypatch)  # every search finds nothing
    r.begin()
    steps(r, LONG_DIED)
    assert len(asked.calls) == 1 and len(plays.made) == 2 and len(moments) == 0
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1
    # Put back to the start a life down, it walks off the edge again.
    steps(r, LONG_DIED)
    assert len(plays.made) == 4 and r.unsticks == 0
    assert len(asked.calls) == 2 and r.ways_sought == 2  # and that life is searched for too
    assert lab.game.state.lives == toy_hopper.START_LIVES - 2


def test_a_go_begins_with_nothing_waited_for(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    r, _ = paused(lab)
    r.set_unstick_buttons(("start",))
    plays = Plays(monkeypatch)
    r.begin()
    steps(r, ASKED_AT + 1)
    assert len(plays.made) == 2
    r.finish()  # stopped in the pause, with the life still to go
    lab.game.reset()
    standing_still(lab)
    r.begin()
    steps(r, OWN_AT + 1)
    # A standstill of its own in the next go is asked about and handled.
    assert len(plays.made) == 4 and r.unsticks == 1


# ------------------------------------------------- a standstill of the AI's own


@pytest.mark.parametrize("wake", [(), ("start",)], ids=["no wake button", "a wake button"])
@pytest.mark.parametrize("stand", [0, 60], ids=["no search at a standstill", "with one"])
def test_a_standstill_of_its_own_is_handled_on_the_step_it_was(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    wake: tuple[str, ...],
    stand: int,
) -> None:
    """It stands short of the edge. Walking on would cost a life; standing does not.
    The asking costs a save, two plays and a load, and the search where it stands, the
    wake press and the end as "stuck" come on the step they came on before."""
    r, moments = paused(lab, stand_after=stand)
    r.set_unstick_buttons(wake)
    plays = Plays(monkeypatch)
    asked = Asked(monkeypatch)  # the search where it stands finds nothing
    right, nothing = lab.move("right"), lab.move()
    standing_still(lab)
    r.begin()
    steps(r, OWN_AT)
    assert NOW_SLOT not in lab.saved and not plays.made and not asked.calls
    x = lab.game.state.x
    with caplog.at_level(logging.DEBUG, logger=LOG):
        went_on = r.step()
    # Its plain move is the one that was paid: held, that loses a life some steps on.
    # Nothing held for as long does not, so no life is on its way out.
    walked, stood = plays.made
    assert walked == (right,) * (2 * AHEAD)
    assert stood == (nothing,) * len(stood) and 1 < len(stood) < 2 * AHEAD
    assert not any("the game's pause" in line for line in said(caplog))
    # Saved to ask and put back; the search where it stands saves and puts back again.
    assert lab.saved.count(NOW_SLOT) == (2 if stand else 1)
    assert lab.loaded.count(NOW_SLOT) == (4 if stand else 3)
    assert len(asked.calls) == (1 if stand else 0) and all(call[2] for call in asked.calls)
    assert (r.unsticks, went_on) == ((1, True) if wake else (0, False))
    assert lab.game.state.x == x and lab.game.state.lives == toy_hopper.START_LIVES
    if not wake:
        assert r.finish().ended_by == "stuck"
        return
    assert len(moments) == 0  # the wake press is a step that cannot be played again


def test_where_the_plain_move_loses_no_life_one_play_is_all(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, _ = paused(lab)
    r.set_unstick_buttons(("start",))
    plays = Plays(monkeypatch, [Outcome(2 * AHEAD, False, 0)])
    standing_still(lab)
    r.begin()
    steps(r, OWN_AT + 1)
    assert plays.made == [(lab.move("right"),) * (2 * AHEAD)]
    assert lab.saved.count(NOW_SLOT) == lab.loaded.count(NOW_SLOT) == 1  # and put back
    assert r.unsticks == 1  # handled at once


def test_a_life_lost_on_another_step_with_another_move_is_not_waited_for(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, _ = paused(lab)
    r.set_unstick_buttons(("start",))
    plays = Plays(monkeypatch, [Outcome(12, True, 0), Outcome(11, True, 0)])
    standing_still(lab)
    r.begin()
    steps(r, OWN_AT + 1)
    assert plays.made == [(lab.move("right"),) * (2 * AHEAD), (lab.move(),) * 12]
    assert r.unsticks == 1  # handled at once


def test_a_life_that_does_not_go_is_waited_for_no_longer(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Both plays lost it on their 12th step, and then the game does not."""
    r, _ = paused(lab)
    r.set_unstick_buttons(("start",))
    lost = Outcome(12, True, 0)
    plays = Plays(monkeypatch, [lost, lost, Outcome(2 * AHEAD, False, 0)])
    standing_still(lab)
    r.begin()
    with caplog.at_level(logging.INFO, logger=LOG):
        steps(r, OWN_AT + 1)
    assert said(caplog) == [WAITS.format(lead="Nothing has changed", stayed=STUCK, on=12)]
    assert len(plays.made) == 2 and r.unsticks == 0
    # It would go on the 12th step from where it asked, and is given 4 more.
    steps(r, 11 + LATE)
    assert len(plays.made) == 2 and r.unsticks == 0
    assert r.step()
    assert len(plays.made) == 2 and r.unsticks == 1  # not asked again: the standstill is handled
    # The wake press forgot the saved moments, so for the next 25 steps of standing no
    # search could follow a life lost: nothing is asked, and it is pressed again.
    steps(r, STUCK)
    assert len(plays.made) == 2 and r.unsticks == 2
    # Once it has walked on and stands again, with moments far enough back, it asks as
    # it did the first time: the wait that came to nothing is over and done with.
    lab.agent.script.clear()
    lab.agent.script.extend([lab.move("right")] * 8 + [lab.move()] * 400)
    while r.unsticks == 2:
        assert r.step()
    assert len(plays.made) == 3 and plays.made[2] == (lab.move("right"),) * (2 * AHEAD)


def test_going_back_ends_the_wait(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    """The game is somewhere else: what was seen from where it stood is not of it."""
    r, moments = paused(lab)
    r.set_unstick_buttons(("start",))
    lost = Outcome(70, True, 0)
    plays = Plays(monkeypatch, [lost, lost, Outcome(2 * AHEAD, False, 0)])
    standing_still(lab)
    r.begin()
    steps(r, OWN_AT + 1)
    assert len(plays.made) == 2 and r.unsticks == 0  # it would wait 70 steps and more
    r.go_back(moments.kept[0])
    lab.agent.script.clear()
    standing_still(lab)  # from the start again: the same few steps, and nothing
    steps(r, OWN_AT + 1)
    assert len(plays.made) == 3 and r.unsticks == 1  # asked anew, and handled


def test_a_list_of_moves_is_played_out_before_it_asks(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, moments = paused(lab)
    plays = Plays(monkeypatch)
    standing_still(lab)
    r.begin()
    steps(r, OWN)
    r.go_back(moments.kept[0], then=[lab.move()] * 60)  # a way to play: 60 steps of nothing
    steps(r, 60)
    assert not plays.made  # nothing has changed for far more than 25 steps
    assert not r.step()  # the list is done: it asks, no life is going, and the go ends stuck
    assert len(plays.made) == 2


# -------------------------------------------------- where no search could follow


@pytest.mark.parametrize(
    "kw",
    [{"ways_per_go": 0}, {"moments": None}, {"lives": False}],
    ids=["searches switched off", "no saved moments", "no life count known"],
)
def test_nothing_is_asked_where_no_search_could_follow(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, kw: dict[str, object]
) -> None:
    lab.teach(lives=bool(kw.pop("lives", True)))
    lab.game.dying_frames = LONG
    lab.game.reset()
    watch = lab.watch()
    lab.saved.clear()
    kw.setdefault("moments", Moments(SIX))
    kw.setdefault("ways_per_go", 3)
    r = lab.runner(watch=watch, unstick_after=STUCK, last_chance=True, **kw)
    r.set_unstick_buttons(("start",))
    plays = Plays(monkeypatch)
    r.begin()
    steps(r, LONG_DIED)
    # As before: the wake button is pressed in the pause, and nothing more is played.
    assert not plays.made and NOW_SLOT not in lab.saved and r.unsticks > 0
    assert (r.ways_sought, lab.game.state.lives) == (0, toy_hopper.START_LIVES - 1)


def test_nothing_is_asked_once_the_searches_are_used_up(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, _ = paused(lab, ways_per_go=1)
    r.set_unstick_buttons(("start",))
    plays = Plays(monkeypatch)
    Asked(monkeypatch)  # the one search finds nothing
    r.begin()
    steps(r, LONG_DIED)
    assert len(plays.made) == 2 and (r.ways_sought, r.unsticks) == (1, 0)
    steps(r, LONG_DIED)  # the second life goes the same way, with no search left
    assert len(plays.made) == 2 and r.ways_sought == 1 and r.unsticks > 0
    assert lab.game.state.lives == toy_hopper.START_LIVES - 2


# ------------------------------------------------------------------- the link


@pytest.mark.parametrize("stand", [None, 30], ids=["nothing changes", "about one spot"])
@pytest.mark.parametrize("fails", ["save", "load 1", "load 2", "load 3"])
def test_a_link_that_drops_while_it_asks_ends_the_go(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, stand: int | None, fails: str
) -> None:
    r, _ = paused(lab) if stand is None else paused(lab, unstick_after=None, stand_after=stand)
    r.set_unstick_buttons(("start",))
    r.begin()
    steps(r, ASKED_AT)
    save, load = lab.adapter.save_state, lab.adapter.load_state
    loads: list[str] = []
    dropped: list[int] = []  # the steps the go had played when the link went

    def saving(slot: str) -> None:
        if slot == NOW_SLOT and fails == "save":
            dropped.append(r.steps)
            raise EmulatorDisconnectedError("The emulator went away.")
        save(slot)

    def loading(slot: str) -> None:
        loads.append(slot)
        if fails == f"load {len(loads)}":
            dropped.append(r.steps)
            raise EmulatorDisconnectedError("The emulator went away.")
        load(slot)

    monkeypatch.setattr(lab.adapter, "save_state", saving)
    monkeypatch.setattr(lab.adapter, "load_state", loading)
    while r.step():
        assert r.steps < LONG_DIED - 1
    assert loads == [NOW_SLOT] * (0 if fails == "save" else int(fails[-1]))
    assert dropped == [r.steps]  # nothing was tried or played after it
    assert not r.signals.listening  # the go is over, as wherever the link drops
    result = r.finish()
    assert result.ended_by == "disconnected" and (result.setbacks, r.unsticks) == (0, 0)
    assert (result.ways_sought, result.ways_stood) == (0, 0)  # and no search is counted
