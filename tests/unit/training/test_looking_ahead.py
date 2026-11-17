"""T-1052: when a thing of a kind that has cost a life comes near, the go plays ahead
off the record what it is about to do, and a life seen going is saved before it goes."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.episode import AHEAD, AHEAD_EVERY, NOW_SLOT, EpisodeResult
from retro_ai_lab.training.moments import LEAD, Moments
from retro_ai_lab.training.search import Search
from retro_ai_lab.training.trial import Outcome
from retro_ai_lab.training.vision import ScreenWatch
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_last_chance_in_a_go import (
    GAP_ENDS,
    LOG,
    LONG,
    LONG_DIED,
    SIX,
    going,
    said,
)

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

OWN = toy_hopper.DYING_FRAMES  # the practice game's own dying pause
THING = ((1, "thing-9"),)  # what the watch says is near: a track and its kind
NOTHING = Search(None, 5)


class Near:
    """Stands in for what the go's watch says is near of a kind that has cost a life:
    what the test sets, and how often it was asked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, watch: ScreenWatch | None) -> None:
        assert watch is not None
        self.now: tuple[tuple[int, str], ...] = ()
        self.asked = 0
        monkeypatch.setattr(watch, "threats", self)

    def __call__(self) -> tuple[tuple[int, str], ...]:
        self.asked += 1
        return self.now


class Played:
    """Stands in for the play ahead: notes what it was given and says the life stays,
    or goes at the step the test names."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, lost_at: int | None = None) -> None:
        self.moves: list[tuple[int, ...]] = []
        self.slots: list[str] = []
        self.lives: list[list[str]] = []
        self.lost_at = lost_at
        monkeypatch.setattr(episode, "play", self)

    def __call__(
        self, connection: object, slot: str, moves: tuple[int, ...], space: object, **kw: object
    ) -> Outcome:
        self.moves.append(tuple(moves))
        self.slots.append(slot)
        self.lives.append([v.name for v in kw["lives"]])  # type: ignore[attr-defined]
        assert kw["frames_per_step"] == going_back.FRAMES and "watch" not in kw
        if self.lost_at is None:
            return Outcome(len(moves), False, 0)
        return Outcome(self.lost_at, True, 0)


class Sought:
    """Stands in for ``find_way`` in the runner: notes the save and the moves each
    search was given, and answers from a list (the real search where it says so)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, answers: list[object]) -> None:
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.live: list[object] = []
        self.answers = list(answers)
        self.real = episode.find_way
        monkeypatch.setattr(episode, "find_way", self)

    def __call__(
        self, connection: object, slot: str, trace: tuple[int, ...], *a: object, **kw: object
    ) -> Search:
        self.calls.append((slot, tuple(trace)))
        self.live.append(kw["live"])
        out = self.answers.pop(0)
        if not isinstance(out, Search):
            out = self.real(connection, slot, trace, *a, **kw)  # type: ignore[arg-type]
        return out


def looking(lab: Lab, frames: int = OWN, **kw: object) -> tuple[EpisodeRunner, Moments]:
    """A go about to start that keeps moments, may search, and looks ahead."""
    kw.setdefault("look_ahead", True)
    return going(lab, frames, **kw)


def walked(lab: Lab, r: EpisodeRunner, count: int) -> list[int]:
    """Play ``count`` steps and say which were played without asking the learner."""
    unasked = []
    for n in range(count):
        asked = lab.agent.asked
        assert r.step()
        if lab.agent.asked == asked:
            unasked.append(n)
    return unasked


# ------------------------------------------------------------ a life seen going


def test_a_life_seen_going_is_saved_before_it_goes(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert (AHEAD, AHEAD_EVERY) == (40, 10)
    r, moments = looking(lab)
    near = Near(monkeypatch, r.watch)
    r.begin()
    steps(r, 5)
    assert near.asked == 5 and r.looked_ahead == 0 and lab.loaded == []
    (start,) = moments.kept
    walk, jump = lab.move("right"), lab.move("a", "right")
    # Something of a kind that has cost a life comes near. Walking on, the life goes at
    # step 36 of the go: 31 steps on from here, inside the 40 it plays ahead.
    near.now = THING
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert r.looked_ahead == 1
    assert (r.ways_ahead, r.ways_ahead_found, r.times_back) == (1, 1, 1)
    assert (r.ways_sought, r.ways_found) == (0, 0)  # none of the go's searches was used
    # The play ahead was from a save of the present, which was then put back; the
    # search was from the moment before, and its way is what the game was put back for.
    assert lab.saved[-2:] == [NOW_SLOT, NOW_SLOT]  # the look ahead's, and the search's
    assert lab.loaded[:2] == [NOW_SLOT, NOW_SLOT]  # played ahead from, and put back to
    assert set(lab.loaded[2:]) == {start.slot} and lab.loaded[-1] == start.slot
    (found,) = said(caplog)
    assert found.startswith(
        "Something of a kind that has cost a life (thing-9) came near at step 5, and "
        f"carrying on would cost one {DIED_AT - 5} steps on. Went back 5 steps and found a "
        "way past on try "
    )
    assert found.endswith(
        ": holding right instead of what it did, with right + a for 4 steps, starting 17 "
        "steps before that would lose the life."
    )
    # No life was lost, and the way is played without asking the learner: over the gap.
    near.now = ()
    asked = lab.agent.asked
    played = [r.last_action]  # the step that looked ahead played the way's first move
    while lab.agent.asked == asked:
        assert r.step()
        played.append(r.last_action)
    played.pop()  # the step the learner was asked again
    assert played == [walk] * 19 + [jump] * 4 + [walk] * (DIED_AT - 23) + [walk] * 20
    assert lab.game.state.lives == toy_hopper.START_LIVES and lab.game.state.x > GAP_ENDS
    result = r.finish()
    assert (result.setbacks, result.first_setback_step) == (0, None)
    assert (result.ways_sought, result.ways_found) == (0, 0)
    assert (result.ways_ahead, result.ways_ahead_found) == (1, 1)
    assert result.describe_ways() == (", saw a life going once before it went and found a way past")


def test_the_search_is_over_what_it_did_and_what_it_was_about_to_do(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, moments = looking(lab)
    near = Near(monkeypatch, r.watch)
    sought = Sought(monkeypatch, [NOTHING])
    walk, jump = lab.move("right"), lab.move("a", "right")
    lab.agent.script.extend([jump] * 3)
    r.begin()
    steps(r, 26)
    first, second = moments.kept
    assert (first.step, second.step) == (0, 20)
    # The life would go 23 steps on: the moment 6 steps back is 29 steps before that
    # step, too near it (a moment lies 30 steps before a loss, T-1003), so it is the
    # one before.
    assert LEAD == 30
    played = Played(monkeypatch, lost_at=23)
    near.now = THING
    assert r.step()
    assert played.slots == [NOW_SLOT] and played.lives == [["Lives"]]
    ((slot, trace),) = sought.calls
    assert slot == first.slot
    assert trace == (jump,) * 3 + (walk,) * 23 + (walk,) * 23
    assert sought.live == [None]  # nobody asked from when on it was settled
    # One step further on it is the newer moment.
    r, moments = looking(lab)
    near = Near(monkeypatch, r.watch)
    sought = Sought(monkeypatch, [NOTHING])
    r.begin()
    steps(r, 26)
    first, second = moments.kept
    played.lost_at = 24
    near.now = THING
    assert r.step()
    assert sought.calls == [(second.slot, (walk,) * 6 + (walk,) * 24)]


def test_the_dying_pause_it_remembers_is_played_on_top_and_asked_about(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A first go loses a life the long way (200 frames of pause) and learns from its
    # search how long this game takes over one: 53 steps.
    r, moments = looking(lab, LONG, last_chance=True)
    near = Near(monkeypatch, r.watch)
    r.begin()
    steps(r, LONG_DIED)
    assert (r.ways_sought, r.ways_found, r.ways_ahead, r.dying_steps) == (1, 1, 0, 53)
    r.finish()
    # The next go looks 40 steps and that pause ahead: without it, the 40 steps from
    # step 5 end before the life has gone, and nothing would be seen.
    lab.game.reset()
    r.begin()
    steps(r, 5)
    near.now = THING
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert (r.looked_ahead, r.ways_sought, r.ways_ahead, r.ways_ahead_found) == (1, 0, 1, 1)
    settled, found = said(caplog)
    assert settled.startswith(
        f"Carrying on from step 5 would cost a life {LONG_DIED - 5} steps on. That was "
        "settled 53 steps before: nothing pressed from then on changes it ("
    )
    assert settled.endswith(
        " to find out). The AI goes back 5 steps and tries its changes before then."
    )
    assert found.startswith(
        "Something of a kind that has cost a life (thing-9) came near at step 5, and "
        f"carrying on would cost one {LONG_DIED - 5} steps on. Went back 5 steps and found a "
        "way past on try "
    )
    # On the way found, with the thing still near, it looks ahead no more until the
    # story is past the step the life would have gone at: a way that only put the loss
    # off would else be found again and again from the same moment.
    looked = []
    while moments.steps <= LONG_DIED:
        assert r.step()
        looked.append(r.looked_ahead)
    assert looked == [1] * LONG_DIED
    assert lab.game.state.lives == toy_hopper.START_LIVES and lab.game.state.x > GAP_ENDS
    # Past it, it does: the next place a life would go at is seen from here.
    assert r.step()
    assert r.looked_ahead == 2
    assert r.finish().setbacks == 0


# ------------------------------------------------------------ when it looks


def test_it_looks_ahead_for_a_thing_once_and_not_too_often(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, moments = looking(lab)
    near = Near(monkeypatch, r.watch)
    played = Played(monkeypatch)
    r.begin()
    steps(r, 3)
    assert played.moves == []  # nothing near
    near.now = THING
    steps(r, 1)
    assert len(played.moves) == 1 and r.looked_ahead == 1
    # The same thing, look after look: it has been looked ahead for.
    steps(r, AHEAD_EVERY + 1)
    assert len(played.moves) == 1
    # Another thing beside it, as soon as it is seen: enough steps have gone by.
    near.now = (*THING, (2, "thing-9"))
    steps(r, 1)
    assert len(played.moves) == 2
    # And one more at the next step: not within ten steps of the last look ahead...
    near.now = (*near.now, (3, "thing-4"))
    steps(r, AHEAD_EVERY - 1)
    assert len(played.moves) == 2
    # ...but then.
    steps(r, 1)
    assert len(played.moves) == 3 and r.looked_ahead == 3
    # Put back to an earlier moment, the things in sight are that moment's again: all
    # three are looked ahead for anew, in one play, once ten steps of the go have gone.
    r.go_back(moments.kept[0])
    steps(r, AHEAD_EVERY - 1)
    assert len(played.moves) == 3
    steps(r, 1)
    assert len(played.moves) == 4
    # And then no more, in whatever order the watch names them.
    near.now = near.now[::-1]
    steps(r, AHEAD_EVERY + 2)
    assert len(played.moves) == 4
    r.go_back(moments.kept[0])
    steps(r, 1)
    assert len(played.moves) == 5 and r.looked_ahead == 5
    assert r.finish().ways_sought == 0  # no life was seen going: nothing was sought


@pytest.mark.parametrize(
    "why",
    ["not asked for", "no searches", "no moments", "no lives", "nothing to go by"],
)
def test_where_no_search_could_follow_it_does_not_look(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, why: str
) -> None:
    played = Played(monkeypatch)
    if why == "not asked for":
        r, _ = going(lab, OWN)  # as every go before this task: the runner is not told to
        assert not r.looks_ahead
    elif why == "no searches":
        r, _ = looking(lab, ways_per_go=0)
    elif why == "no moments":
        lab.teach()
        watch = lab.watch()
        r = lab.runner(watch=watch, look_ahead=True, ways_per_go=1)
    elif why == "no lives":
        lab.teach(lives=False)
        watch = lab.watch()
        r = lab.runner(moments=Moments(SIX), watch=watch, look_ahead=True, ways_per_go=1)
    else:
        r, _ = looking(lab)
    assert r.looks_ahead is (why != "not asked for")
    near = Near(monkeypatch, r.watch)
    near.now = THING
    r.begin()
    if why == "nothing to go by":
        # Before its first step nothing has been paid and nothing done: there is no
        # move to play ahead with. One step later there is.
        assert r.step()
        assert played.moves == [] and r.looked_ahead == 0
        assert r.step()
        assert len(played.moves) == 1
        return
    steps(r, 20)
    assert played.moves == [] and r.looked_ahead == 0 and NOW_SLOT not in lab.saved


def test_look_ahead_is_true_or_false(lab: Lab) -> None:
    lab.teach()
    for wrong in (1, 0, "yes", None):
        with pytest.raises(ConfigurationError, match="look_ahead must be true or false"):
            lab.runner(look_ahead=wrong)


@pytest.mark.parametrize("at", ["the save", "the play", "the load"])
def test_the_link_dropping_while_it_looks_ahead_ends_the_go(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, at: str
) -> None:
    r, _ = looking(lab)
    near = Near(monkeypatch, r.watch)
    r.begin()
    steps(r, 5)

    def gone(*a: object, **kw: object) -> None:
        raise EmulatorError("the link dropped")

    if at == "the save":
        save = lab.adapter.save_state
        monkeypatch.setattr(
            lab.adapter, "save_state", lambda slot: gone() if slot == NOW_SLOT else save(slot)
        )
    elif at == "the play":
        monkeypatch.setattr(episode, "play", gone)
    else:
        Played(monkeypatch)
        monkeypatch.setattr(lab.adapter, "load_state", gone)
    near.now = THING
    asked = lab.agent.asked
    assert not r.step()
    assert lab.agent.asked == asked and r.looked_ahead == 0  # no step was played
    assert r.finish().ended_by == "disconnected"


# ------------------------------------------------------------ what is played ahead


def test_what_is_played_ahead(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    r, moments = looking(lab)
    near = Near(monkeypatch, r.watch)
    played = Played(monkeypatch)
    walk, stand, hop = lab.move("right"), lab.move(), lab.move("a")
    r.begin()
    steps(r, 6)
    assert r.paid_buttons == ("right",)
    # With nothing queued: its plain move, the one that has paid best, 40 steps of it.
    near.now = THING
    steps(r, 1)
    assert played.moves == [(walk,) * AHEAD] and played.slots == [NOW_SLOT]
    assert played.lives == [["Lives"]]
    # With moves queued (here those a going back was given): what is left of them comes
    # first. Nine are played before it may look ahead again.
    r.go_back(moments.kept[0], then=[stand] * 11 + [hop])
    steps(r, AHEAD_EVERY)
    assert len(played.moves) == 2
    assert played.moves[-1] == (stand, stand, hop) + (walk,) * (AHEAD - 3)
    # More queued than it plays ahead: the first 40 of them.
    r.go_back(moments.kept[0], then=[hop] * (AHEAD + 19))
    steps(r, AHEAD_EVERY)
    assert played.moves[-1] == (hop,) * AHEAD and len(played.moves) == 3


def test_with_nothing_paid_it_plays_ahead_the_move_it_mostly_made(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, _ = looking(lab)
    near = Near(monkeypatch, r.watch)
    played = Played(monkeypatch)
    hop, stand = lab.move("a"), lab.move()
    lab.agent.script.extend([hop, stand, hop, hop, stand, stand, stand])
    r.begin()
    steps(r, 4)
    assert r.paid_buttons == ()  # jumping on the spot is paid nothing
    near.now = THING
    steps(r, 1)
    assert played.moves == [(hop,) * AHEAD]  # three hops in the four steps before


def test_looking_ahead_is_off_the_record(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ten steps ahead here, so that the walk stays alive: the game's own play, not a
    # stand-in's. The same go without the look ahead is the same go.
    monkeypatch.setattr(episode, "AHEAD", 10)
    plain, _ = going(lab, OWN)
    plain.begin()
    steps(plain, 12)
    where, learned = lab.game.state, list(lab.agent.learned)
    without = plain.finish()

    lab.agent.learned.clear()
    r, moments = looking(lab)
    near = Near(monkeypatch, r.watch)
    r.begin()
    steps(r, 5)
    near.now = THING
    before = (lab.game.state, r.steps, r.reward, list(lab.agent.learned), moments.kept)
    assert r.step()
    assert r.looked_ahead == 1 and lab.saved[-1] == NOW_SLOT and lab.loaded == [NOW_SLOT] * 2
    # One step on, and nothing else: no step, pay, lesson or moment more than that.
    assert r.steps == before[1] + 1 and moments.kept == before[4]
    assert len(lab.agent.learned) <= len(before[3]) + 1
    steps(r, 6)
    assert lab.game.state == where and lab.agent.learned == learned
    assert r.finish() == without


# ------------------------------------------------------------ alive


def test_alive_nothing_is_changed(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    r, moments = looking(lab, LONG)  # the long pause: the walk goes on for 74 steps
    near = Near(monkeypatch, r.watch)
    played = Played(monkeypatch)
    walk, stand = lab.move("right"), lab.move()
    r.begin()
    assert walked(lab, r, 3) == []  # the learner is asked each step, and walks
    lab.agent.script.extend([stand] * 4)  # what the learner does next
    # A thing comes near and carrying on stays alive: nothing is queued. The learner is
    # asked at that very step and after, and what it chooses is played, not the move
    # that was checked. (A queue filled with that move kept a go from looking for a
    # way on where it stood at a wall: no level was finished in twelve goes.)
    near.now = THING
    moves = []
    for _ in range(4):
        asked = lab.agent.asked
        assert r.step()
        assert lab.agent.asked == asked + 1
        moves.append(r.last_action)
    assert played.moves == [(walk,) * AHEAD] and moves == [stand] * 4
    # Moves queued are played as they were and no more: after the last of them the
    # learner is asked again, though it looked ahead while they were played.
    r.go_back(moments.kept[0], then=[stand] * (AHEAD_EVERY + 2))
    assert walked(lab, r, AHEAD_EVERY + 5) == list(range(AHEAD_EVERY + 2))
    assert r.looked_ahead == 2 and (r.ways_sought, r.ways_ahead) == (0, 0)


# ------------------------------------------------------------ no way found


def test_no_way_found_and_the_go_carries_on(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    r, moments = looking(lab, ways_per_go=1)
    near = Near(monkeypatch, r.watch)
    sought = Sought(monkeypatch, [NOTHING, "the real search"])
    r.begin()
    steps(r, 5)
    kept, where = moments.kept, lab.game.state
    near.now = (*THING, (5, "thing-4"))  # two things: the nearer is the one it names
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert (r.looked_ahead, r.ways_sought, r.ways_ahead, r.ways_ahead_found) == (1, 0, 1, 0)
    assert r.times_back == 0 and lab.loaded[-1] == NOW_SLOT
    # The game is where one step from there puts it, and the moments are kept: the
    # loss has not happened.
    assert moments.kept == kept and moments.steps == 6
    assert lab.game.state.x > where.x and lab.game.state.lives == toy_hopper.START_LIVES
    assert said(caplog) == [
        "Something of a kind that has cost a life (thing-9) came near at step 5, and "
        f"carrying on would cost one {DIED_AT - 5} steps on. Went back 5 steps and made 5 "
        "tries in 0.0 s: none got past, so it carries on. It has now looked for nothing "
        "as often as a go may before a life is lost, and looks ahead no more in this go."
    ]
    # It looks ahead no more in this go: one search for nothing was all it had.
    near.now = ((2, "thing-9"),)
    steps(r, DIED_AT - 7)
    assert r.looked_ahead == 1 and (r.ways_sought, r.ways_ahead) == (0, 1)
    # The life goes, and the go's search after a life lost is still there to be made.
    assert r.step()
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    assert (r.ways_ahead, r.ways_ahead_found) == (1, 0)
    assert len(sought.calls) == 2
    result = r.finish()
    assert result.setbacks == 1
    assert result.describe_ways() == (
        ", went back once and found a way past, saw a life going once before it went and "
        "found no way past"
    )
    # A new go looks ahead again.
    sought.answers.append(NOTHING)
    lab.game.reset()
    r.begin()
    steps(r, 6)
    assert r.looked_ahead == 1 and r.ways_ahead == 1


def test_it_stops_looking_after_as_many_for_nothing_as_a_go_has_searches(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    r, _ = looking(lab, ways_per_go=2)
    near = Near(monkeypatch, r.watch)
    Sought(monkeypatch, [NOTHING, NOTHING])
    played = Played(monkeypatch, lost_at=35)
    r.begin()
    steps(r, 2)
    with caplog.at_level(logging.INFO, logger=LOG):
        for track in (1, 2, 3):
            near.now = ((track, "thing-9"),)
            steps(r, AHEAD_EVERY)
    assert len(played.moves) == 2 and r.looked_ahead == 2
    assert (r.ways_sought, r.ways_ahead, r.ways_ahead_found) == (0, 2, 0)
    first, second = said(caplog)
    assert first.endswith("none got past, so it carries on.")
    assert second.endswith(
        "none got past, so it carries on. It has now looked for nothing as often as a go "
        "may before a life is lost, and looks ahead no more in this go."
    )
    assert r.finish().describe_ways() == (
        ", saw a life going 2 times before it went and found no way past"
    )


@pytest.mark.parametrize("a_go_may", [1, 3])
def test_it_stops_looking_ahead_in_a_game_where_that_keeps_finding_nothing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, a_go_may: int
) -> None:
    """As the searches after a life lost and those at a standstill do, each counted
    apart: as many in a row for nothing as ``give_up_after``, from go to go."""
    r, _ = looking(lab, give_up_after=2, ways_per_go=a_go_may)
    near = Near(monkeypatch, r.watch)
    sought = Sought(monkeypatch, [NOTHING, "the real search", NOTHING, NOTHING, "real"])

    def a_go() -> None:
        # The same thing at the same place in every go: what a go looked ahead for,
        # and the step a way found took it past, are that go's alone.
        lab.game.reset()
        near.now = ()
        r.begin()
        steps(r, 5)
        near.now = THING
        assert r.step()

    # For nothing, then a way found (which starts the count again), then for nothing
    # twice in a row.
    with caplog.at_level(logging.INFO, logger=LOG):
        for found in (0, 1, 0):
            a_go()
            assert (r.looked_ahead, r.ways_ahead, r.ways_ahead_found) == (1, 1, found)
            assert not r.ahead_gave_up
            r.finish()
        assert not said(caplog)[-1].startswith("Looking for a way past")
        a_go()
    assert r.ahead_gave_up and (r.ways_ahead, r.ways_ahead_found) == (1, 0)
    last, gave_up = said(caplog)[-2:]
    assert last.endswith("none got past, so it carries on.")  # one sentence says it stops
    assert gave_up == (
        "Looking for a way past a life it saw going found none 2 times in a row, so the AI "
        "stops looking ahead in this game: here it only costs time."
    )
    # Not in this go either, whatever the go had left: another thing comes near.
    near.now = ((9, "thing-9"),)
    steps(r, AHEAD_EVERY + 1)
    assert r.looked_ahead == 1 and len(sought.calls) == 4
    r.finish()
    # From then on it does not look ahead, in this go or a later one...
    a_go()
    assert (r.looked_ahead, r.ways_ahead) == (0, 0) and len(sought.calls) == 4
    # ...and still goes back after a life lost.
    assert not r.gave_up
    near.now = ()
    steps(r, DIED_AT - 6)
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    # A runner told never to give up does not.
    r, _ = looking(lab, give_up_after=None)
    near = Near(monkeypatch, r.watch)
    sought = Sought(monkeypatch, [NOTHING] * 9)
    for _ in range(9):
        a_go()
        assert r.looked_ahead == 1 and not r.ahead_gave_up
        r.finish()


def test_a_way_found_ahead_costs_none_of_the_gos_searches(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, _ = looking(lab, ways_per_go=1)
    near = Near(monkeypatch, r.watch)
    r.begin()
    steps(r, 5)
    near.now = THING
    assert r.step()
    assert (r.ways_sought, r.ways_ahead, r.ways_ahead_found, r.times_back) == (0, 1, 1, 1)
    # Over the gap, the learner walks on into the next place a life goes at: the go's
    # one search after a life lost is made there.
    near.now = ()
    while r.ways_sought == 0:
        assert r.step()
    assert (r.ways_sought, r.ways_ahead, r.times_back) == (1, 1, 2)
    # And it still looks ahead: the way found took none of those either.
    played = Played(monkeypatch)
    near.now = ((7, "thing-9"),)
    while not played.moves:
        assert r.step()
    assert r.looked_ahead == 2


def test_moments_are_kept_while_it_may_still_look_ahead(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The go's one search after a life lost is used up at the first gap (it finds
    # nothing). Moments were no longer saved from then on (T-1007: no search is left
    # to go back for); a go that looks ahead still has a use for them.
    def a_go(look_ahead: bool) -> tuple[EpisodeRunner, Moments]:
        r, moments = going(lab, OWN, look_ahead=look_ahead)
        Sought(monkeypatch, [NOTHING])
        r.begin()
        steps(r, DIED_AT)
        assert (r.ways_sought, r.ways_found) == (1, 0) and len(moments) == 0
        steps(r, 3)
        return r, moments

    _, moments = a_go(False)
    assert len(moments) == 0
    r, moments = a_go(True)
    assert len(moments) == 1 and r.ways_ahead == 0


# ------------------------------------------------------------ the result


def test_the_result_counts_them() -> None:
    def result(sought: int, found: int, stood: int, ahead: int, ahead_found: int) -> EpisodeResult:
        return EpisodeResult(
            10, 40, 1.0, "max_steps", 3, ways_sought=sought, ways_found=found,
            ways_stood=stood, ways_ahead=ahead, ways_ahead_found=ahead_found,
        )  # fmt: skip

    assert result(0, 0, 0, 0, 0).describe_ways() == ""
    saw = ", saw a life going "
    says = {
        (0, 0, 0, 1, 1): saw + "once before it went and found a way past",
        (0, 0, 0, 1, 0): saw + "once before it went and found no way past",
        (0, 0, 0, 3, 3): saw + "3 times before it went and found a way past each time",
        (0, 0, 0, 3, 1): saw + "3 times before it went and found a way past once",
        (0, 0, 0, 4, 2): saw + "4 times before it went and found a way past 2 times",
        (0, 0, 0, 2, 0): saw + "2 times before it went and found no way past",
        # After what it says of the go's own searches, which reads as it did.
        (2, 1, 0, 1, 1): ", went back 2 times and found a way past once"
        + saw
        + "once before it went and found a way past",
        (3, 1, 2, 5, 4): ", went back 3 times (2 times where it stood still) and found a "
        "way past once" + saw + "5 times before it went and found a way past 4 times",
        (1, 1, 1, 0, 0): ", went back once (where it stood still) and found a way past",
        (2, 0, 2, 0, 0): ", went back 2 times (each time where it stood still) and found "
        "no way past",
        (2, 2, 1, 0, 0): ", went back 2 times (once where it stood still) and found a way "
        "past each time",
        (2, 1, 0, 0, 0): ", went back 2 times and found a way past once",
    }
    for counts, words in says.items():
        assert result(*counts).describe_ways() == words
    # Saved and read back; a go saved before this task has none.
    kept = result(4, 1, 1, 3, 2)
    data = kept.to_dict()
    assert (data["ways_ahead"], data["ways_ahead_found"]) == (3, 2)
    assert EpisodeResult.from_dict(data) == kept
    del data["ways_ahead"], data["ways_ahead_found"]
    old = EpisodeResult.from_dict(data)
    assert (old.ways_ahead, old.ways_ahead_found) == (0, 0) and old.ways_sought == 4
    plain = EpisodeResult(10, 40, 1.0, "max_steps", 3)
    assert (plain.ways_ahead, plain.ways_ahead_found) == (0, 0)
