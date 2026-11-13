"""T-1006: trying the small changes to what it did until one gets past."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import search
from retro_ai_lab.training.search import BUDGET, CARRY_ON, Search, Way, find_way
from retro_ai_lab.training.trial import PAST, Outcome
from retro_ai_lab.training.ways import Change, changes
from tests.unit.training.test_trial import FRAMES, LIVES, SPOT, Hopper, a_go_that_fell_in


def look_for(hopper: Hopper, trace: Sequence[int], watch: object, **kw: object) -> Search:
    kw.setdefault("lives", [LIVES])
    kw.setdefault("frames_per_step", FRAMES)
    return find_way(hopper.connection, SPOT, trace, hopper.space, watch=watch, **kw)  # type: ignore[arg-type]


def test_it_finds_the_first_change_that_gets_across(hopper: Hopper) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    right = hopper.move("right")
    trace = [right] * died
    before = (watch.looks, watch.ground.place, watch.ground.reach, watch.ground.steps, watch.me)
    lent = []

    def trial() -> object:
        lent.append(watch.trial(mark))
        return lent[-1]

    found = look_for(hopper, trace, trial)
    way = found.way
    assert isinstance(way, Way) and way.tries == found.tries <= BUDGET
    # One button more than it was holding, somewhere before the edge.
    assert way.change.move in hopper.space.near(right) and 0 <= way.change.at < died
    assert way.moves == way.change.apply(trace, CARRY_ON) and len(way.moves) == died + CARRY_ON
    assert way.ground >= PAST

    # The game itself agrees: played from the save it is alive at the end and beyond
    # the gap, and every change before it in the list loses the life.
    def plays(moves: Sequence[int]) -> tuple[bool, int]:
        out = hopper.play(list(moves))
        return out.lost, hopper.game.state.x

    lost, x = plays(way.moves)
    gap_end = 240  # the far side of the first gap on this level
    assert not lost and x > gap_end and hopper.game.state.lives == toy_hopper.START_LIVES
    earlier = []
    for change in changes(trace, hopper.space):
        if change == way.change:
            break
        earlier.append(change)
    else:
        raise AssertionError("the way's change is not in the list")
    assert all(plays(change.apply(trace, CARRY_ON))[0] for change in earlier)
    # Each of those cost one try; the way cost two: once blind, once with the screen.
    assert found.tries == len(earlier) + 2 and len(lent) == 1
    # Nothing was done to the go's own watch.
    assert before == (
        watch.looks,
        watch.ground.place,
        watch.ground.reach,
        watch.ground.steps,
        watch.me,
    )


def test_the_same_search_twice_is_the_same(hopper: Hopper) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    trace = [hopper.move("right")] * died
    first = look_for(hopper, trace, lambda: watch.trial(mark))
    second = look_for(hopper, trace, lambda: watch.trial(mark))
    assert first == second and first.way is not None


def test_the_budget_is_never_overspent(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    trace = [hopper.move("right")] * died
    needed = look_for(hopper, trace, lambda: watch.trial(mark)).tries
    assert needed > 3
    played = []
    real = search.play

    def counting(*args: object, **kw: object) -> Outcome:
        played.append(kw.get("watch") is not None)
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(search, "play", counting)
    # One try short: the change that would get across is tried blind and stays alive,
    # and there is no try left to look at it with.
    short = look_for(hopper, trace, lambda: watch.trial(mark), budget=needed - 1)
    assert short == Search(way=None, tries=needed - 1) and len(played) == needed - 1
    played.clear()
    assert look_for(hopper, trace, lambda: watch.trial(mark), budget=1) == Search(None, 1)
    assert played == [False]
    played.clear()
    exact = look_for(hopper, trace, lambda: watch.trial(mark), budget=needed)
    assert exact.way is not None and exact.tries == needed == len(played)
    assert played[-1] is True and not any(played[:-1])


class Clock:
    """A clock the test moves: a second passes with every play."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = 100.0
        self.plays: list[bool] = []
        real = search.play

        def timed(*args: object, **kw: object) -> Outcome:
            self.plays.append(kw.get("watch") is not None)
            self.now += 1.0
            return real(*args, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr(search, "play", timed)

    def __call__(self) -> float:
        return self.now


def test_no_play_starts_once_the_time_is_up(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    trace = [hopper.move("right")] * died
    clock = Clock(monkeypatch)
    out = look_for(hopper, trace, lambda: watch.trial(mark), seconds=5, clock=clock)
    assert out == Search(None, 5, out_of_time=True) and out.seconds == 5.0
    assert clock.plays == [False] * 5
    # A play that is started is finished, however little time was left.
    clock.plays.clear()
    out = look_for(hopper, trace, lambda: watch.trial(mark), seconds=0.25, clock=clock)
    assert out == Search(None, 1, out_of_time=True) and clock.plays == [False]


def test_a_way_found_in_time_is_the_same_way(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    trace = [hopper.move("right")] * died
    plain = look_for(hopper, trace, lambda: watch.trial(mark))
    assert plain.way is not None and not plain.out_of_time
    assert 0.0 < plain.seconds < 60.0  # how long it really took, with no limit set
    needed = plain.tries
    clock = Clock(monkeypatch)
    # Its last play, the one with the screen, starts with a second still to go.
    timed = look_for(hopper, trace, lambda: watch.trial(mark), seconds=needed, clock=clock)
    assert timed == plain and timed.seconds == float(needed) and not timed.out_of_time
    # A second less: the change stayed alive without the screen, and that is all.
    clock.plays.clear()
    short = look_for(hopper, trace, lambda: watch.trial(mark), seconds=needed - 1, clock=clock)
    assert short == Search(None, needed - 1, out_of_time=True)
    assert clock.plays == [False] * (needed - 1)
    # How long it took is not part of what it found.
    assert Search(None, 3, seconds=1.0) == Search(None, 3, seconds=2.0)
    assert Search(None, 3) != Search(None, 3, out_of_time=True)


def test_no_way_when_the_fall_has_begun(hopper: Hopper) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    right = hopper.move("right")
    # A save from the last steps before the life goes: the player is already in the gap.
    buttons = {b.name: b for b in hopper.adapter.buttons()}
    hopper.adapter.load_state(SPOT)
    hopper.adapter.step(FRAMES * (died - 3), pressed=[buttons["right"]])
    assert hopper.game.state.lives == toy_hopper.START_LIVES
    hopper.adapter.save_state("late")
    trace = [right] * 3
    every = len(list(changes(trace, hopper.space)))
    found = find_way(
        hopper.connection,
        "late",
        trace,
        hopper.space,
        frames_per_step=FRAMES,
        lives=[LIVES],
        watch=lambda: watch.trial(mark),
    )
    assert found == Search(way=None, tries=every) and 0 < every < BUDGET


# ------------------------------------------------------------ the rule, with scripted tries


class Script:
    """Stands in for ``trial.play``: gives the outcomes it was handed, in order."""

    def __init__(self, outcomes: Sequence[Outcome | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def __call__(
        self, connection: object, slot: str, moves: Sequence[int], space: object, **kw: object
    ) -> Outcome:
        self.calls.append((tuple(moves), kw.get("watch") is not None))
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


DEAD = Outcome(steps=5, lost=True, ground=0)
ALIVE = Outcome(steps=30, lost=False, ground=0)  # what a try without the screen can say
NOWHERE = Outcome(steps=30, lost=False, ground=1)
ACROSS = Outcome(steps=30, lost=False, ground=3)


def scripted(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[Outcome | Exception]
) -> tuple[Script, list[Change], list[int]]:
    script = Script(outcomes)
    monkeypatch.setattr(search, "play", script)
    trace = [hopper.move("right")] * 10
    return script, list(changes(trace, hopper.space)), trace


def test_alive_is_not_enough(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    script, listed, trace = scripted(
        hopper, monkeypatch, [DEAD, ALIVE, NOWHERE, DEAD, ALIVE, ACROSS]
    )
    watches: list[object] = []

    def trial() -> object:
        watches.append(object())
        return watches[-1]

    found = look_for(hopper, trace, trial, carry_on=4)
    # The second change stayed alive and went nowhere new; the fourth got across.
    assert found.way == Way(listed[3].apply(trace, 4), listed[3], tries=6, ground=3)
    assert found.tries == 6 and len(watches) == 2
    assert script.calls == [
        (listed[0].apply(trace, 4), False),
        (listed[1].apply(trace, 4), False),
        (listed[1].apply(trace, 4), True),
        (listed[2].apply(trace, 4), False),
        (listed[3].apply(trace, 4), False),
        (listed[3].apply(trace, 4), True),
    ]


def test_a_try_that_fails_under_the_screen_is_not_the_way(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    lost_when_seen = Outcome(steps=12, lost=True, ground=4)
    _, listed, trace = scripted(hopper, monkeypatch, [ALIVE, lost_when_seen, ALIVE, ACROSS])
    found = look_for(hopper, trace, object)
    assert found.way is not None and found.way.change == listed[1] and found.tries == 4


def test_the_time_is_checked_before_the_second_play_too(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, _, trace = scripted(hopper, monkeypatch, [DEAD, ALIVE, ACROSS])
    now = [0.0]

    def played(*args: object, **kw: object) -> Outcome:
        now[0] += 1.0
        return script(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(search, "play", played)
    out = look_for(hopper, trace, object, seconds=2, clock=lambda: now[0])
    assert out == Search(None, 2, out_of_time=True)
    assert [seen for _, seen in script.calls] == [False, False]  # the look was not made
    # With the time for it, the look is made and the way found.
    script.outcomes[:] = [DEAD, ALIVE, ACROSS]
    script.calls.clear()
    now[0] = 0.0
    out = look_for(hopper, trace, object, seconds=2.5, clock=lambda: now[0])
    assert out.way is not None and out.tries == 3 and not out.out_of_time


def test_every_play_counts_and_the_list_can_run_out(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A budget that ends between the blind try and the look: the look is not made.
    script, listed, trace = scripted(hopper, monkeypatch, [DEAD, DEAD, ALIVE, ACROSS])
    out = look_for(hopper, trace, object, budget=3, seconds=10_000)
    assert out == Search(None, 3) and not out.out_of_time  # the tries ran out, not the time
    assert [seen for _, seen in script.calls] == [False, False, False]
    # Every change loses the life: the search stops when the list does.
    script, listed, trace = scripted(hopper, monkeypatch, [DEAD] * 500)
    assert 0 < len(listed) < 500
    out = look_for(hopper, trace, object, budget=10_000, seconds=10_000)
    assert out == Search(None, len(listed)) and not out.out_of_time
    # An emulator error is the caller's to deal with.
    script, listed, trace = scripted(hopper, monkeypatch, [DEAD, EmulatorError("gone")])
    with pytest.raises(EmulatorError, match="gone"):
        look_for(hopper, trace, object)


def test_nonsense_is_refused(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    script, _, trace = scripted(hopper, monkeypatch, [])
    for bad in (
        {"lives": []},
        {"budget": 0},
        {"budget": 2.5},
        {"budget": True},
        {"carry_on": -1},
        {"frames_per_step": 0},
        {"look_every": 0},
        {"seconds": 0},
        {"seconds": -1.5},
        {"seconds": "soon"},
        {"seconds": True},
        {"clock": None},
    ):
        with pytest.raises(ConfigurationError):
            look_for(hopper, trace, object, **bad)
    with pytest.raises(ConfigurationError):
        look_for(hopper, [], object)
    with pytest.raises(ConfigurationError):
        look_for(hopper, [len(hopper.space)], object)
    with pytest.raises(ConfigurationError):
        look_for(hopper, trace, None)
    with pytest.raises(ConfigurationError):
        find_way(
            hopper.connection,
            "",
            trace,
            hopper.space,
            frames_per_step=4,
            lives=[LIVES],
            watch=object,
        )
    assert script.calls == []  # nothing was played
    assert (BUDGET, CARRY_ON) == (120, 20)
