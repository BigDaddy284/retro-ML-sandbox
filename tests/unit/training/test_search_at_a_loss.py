"""T-1007: a go that loses a life goes back and looks for a way past."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.services import DiscoveryUpdated
from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.episode import NOW_SLOT, EpisodeResult
from retro_ai_lab.training.moments import Moments
from retro_ai_lab.training.search import Search, Way
from retro_ai_lab.training.ways import Change
from retro_ai_lab.understanding import VariableKind
from retro_ai_lab.understanding.events import EventKind
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, SLOTS, Lab, steps

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

GAP_ENDS = 240  # the far edge of the first gap on this level
FOUND_ON = 18  # the play the search finds the way on, its own two included
WAY = "right + a for 4 steps, starting 17 steps before the life went"


def searching(lab: Lab, **kw: object) -> tuple[EpisodeRunner, Moments]:
    """A go about to start that keeps moments, watches the screen and may search."""
    if lab.knowledge.best(VariableKind.LIVES) is None:
        lab.teach()
    lab.game.reset()  # the player is looked for from the start of the level
    watch = lab.watch()
    lab.saved.clear()  # finding the player made saves of its own
    lab.loaded.clear()
    moments = Moments(SLOTS)
    kw.setdefault("ways_per_go", 1)
    return lab.runner(moments=moments, watch=watch, **kw), moments


def test_a_go_that_falls_in_goes_back_and_gets_across(lab: Lab) -> None:
    r, moments = searching(lab)
    r.begin()
    steps(r, DIED_AT - 1)
    assert lab.loaded == [] and r.ways_sought == 0
    start = moments.kept[0]
    assert start.step == 0 and DIED_AT - moments.kept[-1].step < 30  # the newest is too close
    assert r.step()  # the life goes, and inside the same step it looks and goes back
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    assert lab.saved[-1] == NOW_SLOT and lab.loaded[-1] == start.slot
    assert set(lab.loaded) == {start.slot}  # every try and the way are from that moment
    assert lab.game.state.lives == toy_hopper.START_LIVES
    assert moments.steps == 0 and moments.kept == (start,)
    # The way is played without asking the learner, and gets the player across alive.
    asked, learned = lab.agent.asked, len(lab.agent.learned)
    played = []
    while lab.agent.asked == asked:
        assert r.step()
        played.append(r.last_action)
    played.pop()  # the step the learner was asked again
    walk, jump = lab.move("right"), lab.move("a", "right")
    assert played == [walk] * 19 + [jump] * 4 + [walk] * (DIED_AT - 23) + [walk] * 20
    assert lab.game.state.lives == toy_hopper.START_LIVES
    assert lab.game.state.x > GAP_ENDS
    # Its steps are steps like any other: counted, learned from, and looked at.
    assert r.steps == DIED_AT + len(played) + 1
    taught = [action for _, action, *_ in lab.agent.learned[learned:]]
    assert taught[:23] == [walk] * 19 + [jump] * 4
    result = r.finish()
    assert (result.setbacks, result.first_setback_step) == (1, DIED_AT)
    assert (result.ways_sought, result.ways_found) == (1, 1)


def test_the_loss_is_real_and_the_tries_are_nobody_s(lab: Lab) -> None:
    r, _ = searching(lab)
    r.begin()
    steps(r, DIED_AT - 1)
    events: list[object] = []
    lab.bus.subscribe(object, events.append)
    paid = r.reward  # for the new ground on the way to the gap
    assert r.step()
    assert r.ways_found == 1
    # The step the life went on was learned, with its penalty, as the end of that story.
    assert len(lab.agent.learned) == DIED_AT
    *_, (_, action, reward, _, done, spanned) = lab.agent.learned
    assert (action, reward, done, spanned) == (lab.move("right"), -10.0, True, 1)
    assert r.reward == paid - 10.0 and r.steps == DIED_AT
    # The engine saw the life go and nothing else: no life gained, no restart, and not
    # one of the plays of the search.
    kinds = [e.kind for u in events if isinstance(u, DiscoveryUpdated) for e in u.events]
    assert kinds == [EventKind.SETBACK]
    assert len(lab.loaded) == FOUND_ON + 1  # the plays, and going back
    # Playing on, being put back costs nothing and no life is lost again.
    for _ in range(DIED_AT + 20):
        assert r.step()
        assert r.last_reward >= 0.0
    assert r.finish().setbacks == 1


def test_no_way_found_leaves_the_game_where_it_was(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same go without a search, to compare with.
    plain, _ = searching(lab, ways_per_go=0)
    plain.begin()
    states = []
    for _ in range(DIED_AT + 12):
        assert plain.step()
        states.append((lab.game.state, plain.state, plain.reward))
    plain.finish()
    taught = list(lab.agent.learned)

    lab.agent.learned.clear()
    lab.saved.clear()
    r, moments = searching(lab, ways_per_go=2, way_tries=5)
    saved_now = []
    save = lab.adapter.save_state

    def saving(slot: str) -> None:
        if slot == NOW_SLOT:
            saved_now.append(lab.game.state)
        save(slot)

    monkeypatch.setattr(lab.adapter, "save_state", saving)
    r.begin()
    for step in range(DIED_AT):
        assert r.step()
        assert (lab.game.state, r.state, r.reward) == states[step]
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 0, 0)
    assert len(lab.loaded) == 5 + 1 and lab.loaded[-1] == NOW_SLOT
    assert saved_now == [states[DIED_AT - 1][0]]
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1
    # The loss stands: what was saved before it leads nowhere any more.
    assert len(moments) == 0
    # The go carries on as if nobody had looked, and saves afresh with its next step.
    for step in range(DIED_AT, DIED_AT + 12):
        assert r.step()
        assert (lab.game.state, r.state, r.reward) == states[step]
    assert [m.step for m in moments.kept] == [DIED_AT]
    result = r.finish()
    assert lab.agent.learned == taught
    assert (result.setbacks, result.ways_sought, result.ways_found) == (1, 1, 0)


def test_only_so_many_searches_a_go(lab: Lab) -> None:
    r, moments = searching(lab, ways_per_go=1, way_tries=5)
    r.begin()
    steps(r, DIED_AT)
    assert r.ways_sought == 1 and r.ways_found == 0
    saves, loads = len(lab.saved), len(lab.loaded)
    # It walks into the gap again: nothing is left to look with, so nothing is saved either.
    while lab.game.state.lives > toy_hopper.START_LIVES - 2:
        assert r.step()
    assert r.ways_sought == 1
    assert (len(lab.saved), len(lab.loaded)) == (saves, loads) and len(moments) == 0
    assert r.finish().setbacks == 2
    # The next go has its search again.
    lab.game.reset()
    r.begin()
    steps(r, DIED_AT)
    assert r.ways_sought == 1 and len(lab.loaded) == 2 * loads
    r.finish()


def test_no_search_without_what_it_needs(lab: Lab) -> None:
    lab.teach()
    # No screen watch: nothing can tell new ground, so the loss stands.
    moments = Moments(SLOTS)
    blind = lab.runner(moments=moments, ways_per_go=1)
    blind.begin()
    steps(blind, DIED_AT)
    assert blind.ways_sought == 0 and lab.loaded == [] and len(moments) == 0
    assert NOW_SLOT not in lab.saved
    blind.finish()

    # No moment far enough back.
    watch = lab.watch()
    lab.saved.clear()
    lab.loaded.clear()
    late = Moments(SLOTS, lead=DIED_AT + 1)
    r = lab.runner(moments=late, watch=watch, ways_per_go=1)
    r.begin()
    steps(r, DIED_AT)
    assert r.ways_sought == 0 and lab.loaded == [] and len(late) == 0
    r.finish()

    # Not asked to search: the moments are kept for whoever holds them, as before.
    lab.game.reset()
    kept = Moments(SLOTS)
    off = lab.runner(moments=kept, watch=watch)
    off.begin()
    steps(off, DIED_AT)
    assert off.ways_sought == 0 and lab.loaded == [] and len(kept) == 2
    assert off.finish().ways_sought == 0


def test_a_life_lost_on_the_way_drops_the_rest(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    walk = lab.move("right")
    wrong = Way((walk,) * (DIED_AT + 20), Change(0, walk, 4), tries=2, ground=3)
    monkeypatch.setattr(episode, "find_way", lambda *a, **kw: Search(wrong, 2))
    r, moments = searching(lab)
    r.begin()
    steps(r, DIED_AT)
    assert (r.ways_found, r.times_back) == (1, 1)
    asked = lab.agent.asked
    steps(r, DIED_AT)  # the "way" walks into the gap as well
    assert lab.agent.asked == asked and lab.game.state.lives == toy_hopper.START_LIVES - 1
    assert r.ways_sought == 1 and len(moments) == 0
    assert r.step()
    assert lab.agent.asked == asked + 1  # the 20 moves still to play were dropped
    assert r.finish().setbacks == 2


def test_the_search_runs_flat_out(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
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

    r, _ = searching(lab, flat_out=flat_out)
    r.begin()
    steps(r, DIED_AT - 1)
    assert "fast" not in order
    order.clear()
    assert r.step()
    plays = ["load back0"] * FOUND_ON
    assert order == ["fast", f"save {NOW_SLOT}", *plays, "normal", "load back0"]
    r.finish()

    # With no way found the present is loaded again before the speed is put back.
    none, _ = searching(lab, flat_out=flat_out, way_tries=3)
    none.begin()
    steps(none, DIED_AT - 1)
    order.clear()
    assert none.step()
    assert order == [
        "fast",
        f"save {NOW_SLOT}",
        *["load back0"] * 3,
        f"load {NOW_SLOT}",
        "normal",
    ]
    none.finish()


def test_a_dropped_link_in_the_search_ends_the_go(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    left: list[str] = []

    @contextmanager
    def flat_out() -> Iterator[None]:
        try:
            yield
        finally:
            left.append("normal")

    r, _ = searching(lab, flat_out=flat_out)
    r.begin()
    steps(r, DIED_AT - 1)

    def gone(slot: str) -> None:
        raise EmulatorError("the link dropped")

    monkeypatch.setattr(lab.adapter, "load_state", gone)
    assert not r.step()
    assert left == ["normal"] and not r.running
    result = r.finish()
    assert result.ended_by == "disconnected"
    assert (result.setbacks, result.ways_sought, result.ways_found) == (1, 1, 0)


def test_a_search_out_of_time_leaves_the_loss_standing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    now = [0.0]

    def clock() -> float:  # a second passes every time it is read
        now[0] += 1.0
        return now[0]

    real = episode.find_way
    limits: list[object] = []

    def timed(*args: object, **kw: object) -> Search:
        limits.append(kw["seconds"])
        return real(*args, **kw, clock=clock)  # type: ignore[arg-type]

    monkeypatch.setattr(episode, "find_way", timed)
    r, moments = searching(lab, way_seconds=4.5)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        r.begin()
        steps(r, DIED_AT)
    assert limits == [4.5]
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 0, 0)
    assert len(lab.loaded) == 4 + 1 and lab.loaded[-1] == NOW_SLOT
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1 and len(moments) == 0
    said = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("Lost a")]
    assert said == [
        f"Lost a life at step {DIED_AT}. Went back {DIED_AT} steps and made 4 tries in 6.0 s, "
        "all the time a search may take: none got past."
    ]
    assert r.step() and r.finish().setbacks == 1
    # With no limit asked for, none is passed on.
    limits.clear()
    free, _ = searching(lab)
    free.begin()
    steps(free, DIED_AT)
    assert limits == [None] and free.finish().ways_found == 1


def test_the_result_and_the_log_say_what_was_found(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, _ = searching(lab)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        r.begin()
        steps(r, DIED_AT)
        found = r.finish()
        none, _ = searching(lab, way_tries=5)
        none.begin()
        steps(none, DIED_AT)
        missed = none.finish()
    said = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("Lost a")]
    took = r"\d+\.\d s"  # how long the search took, to a tenth of a second
    assert len(said) == 2
    assert re.fullmatch(
        f"Lost a life at step {DIED_AT}\\. Went back {DIED_AT} steps and found a way past "
        f"on try {FOUND_ON}, after {took}: {re.escape(WAY)}\\.",
        said[0],
    )
    assert re.fullmatch(
        f"Lost a life at step {DIED_AT}\\. Went back {DIED_AT} steps and made 5 tries "
        f"in {took}: none got past\\.",
        said[1],
    )
    assert found.describe_ways() == ", went back once and found a way past"
    assert missed.describe_ways() == ", went back once and found no way past"
    for result in (found, missed):
        assert EpisodeResult.from_dict(result.to_dict()) == result

    def ways(sought: int, got: int) -> str:
        return EpisodeResult(
            1, 4, 0.0, "stopped", 1, ways_sought=sought, ways_found=got
        ).describe_ways()

    assert ways(0, 0) == ""
    assert ways(3, 0) == ", went back 3 times and found no way past"
    assert ways(3, 1) == ", went back 3 times and found a way past once"
    assert ways(3, 2) == ", went back 3 times and found a way past 2 times"
    assert ways(3, 3) == ", went back 3 times and found a way past each time"
    # A result saved before this task reads as a go that never looked.
    old = found.to_dict()
    del old["ways_sought"], old["ways_found"]
    read = EpisodeResult.from_dict(old)
    assert (read.ways_sought, read.ways_found) == (0, 0) and read.setbacks == found.setbacks


def test_nonsense_is_refused(lab: Lab) -> None:
    lab.teach()
    for bad in (
        {"ways_per_go": -1},
        {"ways_per_go": 1.5},
        {"ways_per_go": True},
        {"way_tries": 0},
        {"way_tries": "many"},
        {"way_seconds": 0},
        {"way_seconds": "long"},
        {"way_seconds": True},
        {"moments": Moments(("back0", NOW_SLOT))},
    ):
        with pytest.raises(ConfigurationError):
            lab.runner(**bad)
    assert lab.saved == [] and lab.loaded == []
