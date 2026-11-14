"""T-1031: a go that loses a life goes back to before the loss was settled."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.chance import Chance
from retro_ai_lab.training.episode import NOW_SLOT, ROOM
from retro_ai_lab.training.moments import LEAD, Moments
from retro_ai_lab.training.search import Search
from retro_ai_lab.understanding import VariableKind
from tests.unit.training import test_going_back as going_back
from tests.unit.training import test_going_back_in_training as in_training
from tests.unit.training.test_going_back import DIED_AT, SLOTS, Lab, steps
from tests.unit.training.test_standing_still import NEAR, Asked, hop_about

lab = going_back.lab  # the practice game behind the real services (pytest fixture)
training_lab = in_training.lab  # and the services a person trains with (pytest fixture)

SIX = tuple(f"back{i}" for i in range(6))  # as many moments as the app keeps
GAP_ENDS = 240  # the far edge of the first gap on this level
LONG = 200  # frames of dying pause: about what Super Mario Bros. takes
LONG_DIED = 74  # walking right from the start with that pause, the life goes on this step
OFF_THE_EDGE = 21  # steps of that walk before nothing changes the loss any more
LOG = "retro_ai_lab.training.episode"


def going(
    lab: Lab, frames: int, slots: tuple[str, ...] = SIX, **kw: object
) -> tuple[EpisodeRunner, Moments]:
    """A go about to start, with that dying pause, that keeps moments and may search."""
    if lab.knowledge.best(VariableKind.LIVES) is None:
        lab.teach()
    lab.game.dying_frames = frames
    lab.game.reset()  # the player is looked for from the start of the level
    watch = lab.watch()
    lab.saved.clear()  # finding the player made saves of its own
    lab.loaded.clear()
    moments = Moments(slots)
    kw.setdefault("ways_per_go", 1)
    # This lab's learner sees the place and the lives and nothing else, so 50 steps of
    # pause are 50 steps of nothing changing: not what is under test here.
    kw.setdefault("unstick_after", None)
    return lab.runner(moments=moments, watch=watch, **kw), moments


def said(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == LOG]


def in_order(loaded: list[str]) -> list[str]:
    """The slots loaded, each once, in the order they were first loaded."""
    return list(dict.fromkeys(loaded))


def test_a_go_goes_back_to_before_the_last_chance_and_gets_across(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    assert ROOM == LEAD // 2 == 15
    r, moments = going(lab, LONG, last_chance=True)
    r.begin()
    steps(r, LONG_DIED - 1)
    assert lab.loaded == [] and r.ways_sought == 0
    assert [m.step for m in moments.kept] == [0, 20, 40, 60]
    start, second, third, newest = moments.kept
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()  # the life goes, and inside the same step it looks and goes back
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    # The moment 34 steps back is the one it went to before, and it is in the pause:
    # every other move from there was asked, and none changed a thing. From the one
    # before, one step is left before the edge. So it is the start it searches from.
    assert in_order(lab.loaded) == [third.slot, second.slot, start.slot]
    assert lab.loaded.count(third.slot) == len(lab.space) == 15  # what it did, and 14 others
    assert newest.slot not in lab.loaded and NOW_SLOT not in lab.loaded
    assert lab.saved[-1] == NOW_SLOT and lab.loaded[-1] == start.slot
    assert lab.game.state.lives == toy_hopper.START_LIVES
    assert moments.steps == 0 and moments.kept == (start,)
    # It says so, in steps.
    settled, found = said(caplog)
    assert settled.startswith(
        "Lost a life at step 74. That was settled 53 steps before: nothing pressed from "
        "then on changes it ("
    )
    assert settled.endswith(
        " to find out). The AI goes back 74 steps, not 34, and tries its changes before then."
    )
    assert found.startswith("Lost a life at step 74. Went back 74 steps and found a way past on")
    assert found.endswith(
        "holding right instead of what it did, with right + a for 4 steps, starting 54 "
        "steps before that would lose the life."
    )
    # The way is played without asking the learner, and gets the player across alive.
    asked = lab.agent.asked
    played = []
    while lab.agent.asked == asked:
        assert r.step()
        played.append(r.last_action)
    played.pop()  # the step the learner was asked again
    walk, jump = lab.move("right"), lab.move("a", "right")
    assert played == [walk] * 20 + [jump] * 4 + [walk] * (LONG_DIED - 24) + [walk] * 20
    assert lab.game.state.lives == toy_hopper.START_LIVES
    assert lab.game.state.x > GAP_ENDS
    result = r.finish()
    assert (result.setbacks, result.first_setback_step) == (1, LONG_DIED)
    assert (result.ways_sought, result.ways_found) == (1, 1)


def test_an_older_moment_where_the_first_leaves_no_room(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Three moments kept: at the loss they are those of steps 20, 40 and 60, and the
    # one of step 20 has a single step before the edge. It is the oldest there is, so
    # that one step is where the changes are tried, and the jump there gets across.
    r, moments = going(lab, LONG, SLOTS, last_chance=True)
    r.begin()
    steps(r, LONG_DIED - 1)
    assert [m.step for m in moments.kept] == [20, 40, 60]
    oldest, first, _ = moments.kept
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    assert in_order(lab.loaded) == [first.slot, oldest.slot]
    assert moments.steps == 20 and moments.kept == (oldest,)
    settled, found = said(caplog)
    assert "That was settled 53 steps before" in settled
    assert settled.endswith("The AI goes back 54 steps, not 34, and tries its changes before then.")
    assert "Went back 54 steps and found a way past on" in found
    assert found.endswith("starting 54 steps before that would lose the life.")
    while lab.game.state.x <= GAP_ENDS:
        assert r.step()
    assert lab.game.state.lives == toy_hopper.START_LIVES
    r.finish()

    # With the game's own pause the moment it goes to anyway has room: 21 steps before
    # the edge, where 15 are asked for. It stays with that one, and says so.
    caplog.clear()
    r, moments = going(lab, toy_hopper.DYING_FRAMES, last_chance=True)
    r.begin()
    steps(r, DIED_AT - 1)
    assert [m.step for m in moments.kept] == [0, 20]
    start = moments.kept[0]
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    assert in_order(lab.loaded) == [start.slot]
    settled, found = said(caplog)
    assert settled.startswith("Lost a life at step 36. That was settled 15 steps before: ")
    assert settled.endswith("The AI goes back 36 steps and tries its changes before then.")
    assert "Went back 36 steps and found a way past on" in found
    r.finish()

    # What it is told is what it goes by: the first moment said to leave 5 steps, each
    # older one has the 20 steps more that lie between them, and nothing more is asked.
    caplog.clear()
    asked: list[tuple[str, int, object]] = []

    def told(connection: object, slot: str, trace: tuple[int, ...], *a: object, **kw: object):  # type: ignore[no-untyped-def]
        asked.append((slot, len(trace), kw["seconds"]))
        return Chance(5, 9)

    monkeypatch.setattr(episode, "last_chance", told)
    searches = Asked(monkeypatch)
    lives: list[object] = []
    monkeypatch.setattr(searches, "enough", lives)  # (the stand-in notes "enough" there)
    r, moments = going(lab, LONG, last_chance=True)
    r.begin()
    steps(r, LONG_DIED - 1)
    start, second, third, _ = moments.kept
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert asked == [(third.slot, 34, None)]
    assert searches.calls == [(second.slot, (lab.move("right"),) * 54, False)]
    settled, _ = said(caplog)
    assert settled == (
        "Lost a life at step 74. That was settled 29 steps before: nothing pressed from "
        "then on changes it (9 plays, 0.0 s to find out). The AI goes back 54 steps, not "
        "34, and tries its changes before then."
    )
    r.finish()


def test_the_search_is_told_and_time_is_kept(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    given: list[dict[str, object]] = []
    real = episode.find_way

    def finding(*args: object, **kw: object) -> Search:
        given.append(kw)
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(episode, "find_way", finding)
    # What the search is given is the steps before the edge from the moment it starts at.
    r, _ = going(lab, LONG, last_chance=True)
    r.begin()
    steps(r, LONG_DIED)
    assert [kw["live"] for kw in given] == [OFF_THE_EDGE]
    r.finish()
    r, _ = going(lab, LONG, SLOTS, last_chance=True)
    r.begin()
    steps(r, LONG_DIED)
    assert [kw["live"] for kw in given] == [OFF_THE_EDGE, 1]
    r.finish()
    # Finding it has the time one search may take, and no more: a play in flight is
    # finished, so with next to no time the first play is all there is.
    given.clear()
    caplog.clear()
    r, _ = going(lab, LONG, last_chance=True, way_seconds=1e-9)
    r.begin()
    steps(r, LONG_DIED - 1)
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert [kw["live"] for kw in given] == [None]
    assert lab.loaded == [NOW_SLOT]  # no play was started; the game is where it was
    first, second = said(caplog)
    assert first == (
        "Lost a life at step 74. Finding from when on that was settled took longer than "
        "the time a search may take, so the AI looks for a way past as it did before."
    )
    assert "Went back 34 steps and made 0 tries" in second and "none got past" in second
    r.finish()
    # Played again, what it did does not lose the life on the same step: nothing can be
    # told from it, and the search is the one it was.
    given.clear()
    caplog.clear()
    monkeypatch.setattr(episode, "last_chance", lambda *a, **kw: None)
    r, _ = going(lab, LONG, last_chance=True, way_tries=3)
    r.begin()
    steps(r, LONG_DIED - 1)
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert [kw["live"] for kw in given] == [None]
    first, second = said(caplog)
    assert first == (
        "Lost a life at step 74. What it did in the 34 steps before, played again, did "
        "not lose the life on the same step, so the AI cannot tell from when on that "
        "was settled, and looks for a way past as it did before."
    )
    assert "Went back 34 steps and made 3 tries" in second
    r.finish()


def test_the_game_is_put_back_when_nothing_is_found(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    two = ("back0", "back1")
    # Two moments kept: at the loss those of steps 40 and 60, both in the pause already.
    r, moments = going(lab, LONG, two, ways_per_go=2, last_chance=True)
    saved_now = []
    save = lab.adapter.save_state

    def saving(slot: str) -> None:
        if slot == NOW_SLOT:
            saved_now.append(lab.game.state)
        save(slot)

    monkeypatch.setattr(lab.adapter, "save_state", saving)
    r.begin()
    steps(r, LONG_DIED - 1)
    assert [m.step for m in moments.kept] == [40, 60] and lab.loaded == []
    learned = len(lab.agent.learned)
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 0, 0)
    # Fifteen plays to know it is too late, the best-paid button held once, and back.
    assert len(lab.loaded) == 15 + 1 + 1 and lab.loaded[-1] == NOW_SLOT
    assert set(lab.loaded[:-1]) == {"back0"}  # the moment of step 40; there is none older
    # The game is exactly where it was when the life went, a life down.
    assert saved_now == [lab.game.state]
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1
    assert (r.steps, len(lab.agent.learned)) == (LONG_DIED, learned + 1)  # no play was learned
    settled, missed = said(caplog)
    assert settled.startswith(
        "Lost a life at step 74. That was settled before the earliest moment the AI can "
        "go back to, 34 steps before: nothing pressed since changes it (15 plays, "
    )
    assert settled.endswith(" to find out).")
    assert "Went back 34 steps and made 1 tries" in missed and "none got past" in missed
    # The loss stands: what was saved before it leads nowhere any more. The go carries
    # on, and saves afresh with its next step.
    assert len(moments) == 0
    steps(r, 12)
    assert [m.step for m in moments.kept] == [LONG_DIED]
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1
    result = r.finish()
    assert (result.setbacks, result.ways_sought, result.ways_found) == (1, 1, 0)


def test_off_unless_asked_for(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The same go as the first test's, without the setting: it goes back 34 steps, into
    # the pause, and every try is lost. That is what a long pause did until now.
    r, moments = going(lab, LONG)
    r.begin()
    steps(r, LONG_DIED - 1)
    third = moments.kept[2]
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 0, 0)
    (missed,) = said(caplog)
    assert missed.startswith("Lost a life at step 74. Went back 34 steps and made ")
    assert missed.endswith("none got past.")
    assert in_order(lab.loaded) == [third.slot, NOW_SLOT]
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1
    r.finish()
    # It is on or off, and nothing else.
    for bad in (1, 0, "yes", None):
        with pytest.raises(ConfigurationError, match="last_chance must be true or false"):
            lab.runner(last_chance=bad)

    # At a standstill no life is lost, so there is nothing to find: it is not asked.
    def never(*args: object, **kw: object) -> None:
        raise AssertionError("no life was lost")

    monkeypatch.setattr(episode, "last_chance", never)
    searches = Asked(monkeypatch)
    r, _ = going(
        lab,
        toy_hopper.DYING_FRAMES,
        last_chance=True,
        stand_after=episode.STAND_AFTER,
        ways_per_go=9,
    )
    r.begin()
    hop_about(lab, r, NEAR)
    assert len(searches.calls) == 1 and searches.calls[0][2] is True
    r.finish()


def test_the_service_asks_for_it(
    training_lab: in_training.Lab,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    made: list[dict[str, object]] = []
    real = in_training.module.EpisodeRunner

    def making(*args: object, **kw: object) -> EpisodeRunner:
        made.append(kw)
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(in_training.module, "EpisodeRunner", making)
    t = training_lab.training()
    game = training_lab.connect()
    game.dying_frames = LONG
    in_training.walking_right(t, monkeypatch)
    with caplog.at_level(logging.INFO, logger=LOG):
        in_training.until_it_looks(t, most=LONG_DIED + 10)
    assert [kw["last_chance"] for kw in made] == [True]
    go = t.episode
    assert go is not None and (go.ways_sought, go.ways_found, go.times_back) == (1, 1, 1)
    assert game.state.lives == toy_hopper.START_LIVES
    assert any("That was settled 53 steps before" in line for line in said(caplog))
    # With a pause like that, training gets across the first gap.
    while game.state.x <= GAP_ENDS:
        assert t.step() and t.episode is go
        assert game.state.lives == toy_hopper.START_LIVES
    t.stop()
