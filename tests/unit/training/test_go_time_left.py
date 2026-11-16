"""T-1046: what a go in progress has left, on its timer and with nothing paid."""

from __future__ import annotations

from retro_ai_lab.training import EpisodeRunner, Mode
from retro_ai_lab.training.service import GoLeft
from tests.unit.training import test_going_back as going_back
from tests.unit.training import test_service as the_service
from tests.unit.training.test_go_until_over import NO_TIMER, Paying
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_remembered_way import planted, remembering
from tests.unit.training.test_standing_still import recalling

lab = going_back.lab  # the practice hopper behind the real services (pytest fixture)
racer = the_service.lab  # the practice racer behind the training service (pytest fixture)

TIME = 50  # the racer lab's "Each go lasts", in steps
MOST = 27_000  # the most a go until the game is over may last there


def left(r: EpisodeRunner) -> tuple[int, int | None]:
    return r.steps_left, r.unpaid_steps_left


# ------------------------------------------------------------ the runner


def test_a_go_counts_its_timer_down(lab: Lab) -> None:
    lab.teach()
    Paying(lab, {3: (5.0,)})  # what it is paid is nothing to a timer
    r = lab.runner(max_steps=30)
    assert left(r) == (30, None)  # a go not begun has all of it
    r.begin()
    assert left(r) == (30, None)
    seen = []
    while r.step():
        seen.append(r.steps_left)
    assert seen == list(range(29, 0, -1))
    assert left(r) == (0, None) and r.finish().ended_by == "max_steps"
    # The next go has all of it again.
    lab.game.reset()
    r.begin()
    assert left(r) == (30, None)
    r.step()
    assert left(r) == (29, None)


def test_the_wait_with_nothing_paid_goes_back_up_when_it_is_paid(lab: Lab) -> None:
    lab.teach()
    paying = Paying(lab, {3: (5.0,), 7: (5.0,)})
    r = lab.runner(max_steps=NO_TIMER, gain_within=10)
    assert left(r) == (NO_TIMER, 10)
    r.begin()
    seen = [r.unpaid_steps_left]
    while r.step():
        seen.append(r.unpaid_steps_left)
    assert seen == [10, 9, 8, 10, 9, 8, 7, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert left(r) == (NO_TIMER - 17, 0) and r.finish().ended_by == "no_gain"
    # Points taken away are no pay: they leave the count where it was.
    lab.game.reset()
    paying.new_go({2: (-5.0,), 4: (2.0,)})
    r.begin()
    steps(r, 3)
    assert left(r) == (NO_TIMER - 3, 7)
    steps(r, 2)
    assert left(r) == (NO_TIMER - 5, 9)
    # A new go starts from the whole count.
    r.finish()
    lab.game.reset()
    paying.new_go()
    r.begin()
    assert left(r) == (NO_TIMER, 10)


def test_the_way_it_remembers_does_not_count_either_down(lab: Lab) -> None:
    paying = Paying(lab)
    walk, stand = lab.move("right"), lab.move()
    r, route, harness = remembering(lab, max_steps=30, gain_within=12)
    planted(route, harness, [walk] * 20)
    lab.agent.script.extend([stand] * 50)
    paying.new_go({5: (3.0,)})
    r.begin()
    for _ in range(20):
        assert r.step()
        assert left(r) == (30, 12)
    assert (r.steps, r.own_steps) == (20, 0)
    steps(r, 4)
    assert left(r) == (26, 8)


def test_a_way_just_found_played_past_the_end_reads_nought(lab: Lab) -> None:
    # The go walks into the first gap on its 36th step, goes back and finds the way
    # across: 56 steps, played to their end although the timer is up on the 50th.
    r, _, _ = recalling(lab, max_steps=50)
    r.begin()
    steps(r, DIED_AT)
    assert r.ways_found == 1 and left(r) == (50 - DIED_AT, None)
    steps(r, 13)
    assert r.own_steps == 49 and r.steps_left == 1
    seen = set()
    for _ in range(42):
        assert r.step()
        seen.add(r.steps_left)
    assert seen == {0} and r.own_steps == 91  # never less than nought
    assert not r.step() and r.finish().ended_by == "max_steps"


def test_so_does_the_wait_while_such_a_way_is_played(lab: Lab) -> None:
    # The same go with no timer. It is last paid for new ground on its 23rd step, so
    # twenty steps with nothing paid are up on the 43rd, part of the way through the
    # way it found, and it is paid again on the 61st.
    r, _, _ = recalling(lab, max_steps=NO_TIMER, gain_within=20)
    r.begin()
    steps(r, 42)
    assert left(r) == (NO_TIMER - 42, 1)
    waits = []
    for _ in range(19):
        assert r.step()
        waits.append(r.unpaid_steps_left)
    assert waits == [0] * 18 + [20] and r.steps == 61


# ------------------------------------------------------------ the service


def test_nothing_is_left_of_a_go_while_training_is_stopped(racer: the_service.Lab) -> None:
    t = racer.training
    assert t.go_left is None  # no game
    racer.connect()
    assert t.go_left is None  # not started
    t.start(Mode.INDEPENDENT)
    assert t.go_left == GoLeft(TIME, False)  # before its first step, all of it
    t.step()
    assert t.go_left == GoLeft(TIME - 1, False)
    t.pause()
    assert t.go_left == GoLeft(TIME - 1, False)  # paused, what it had
    t.resume()
    t.stop()
    assert t.go_left is None
    # And none when the game goes away in the middle of a go.
    t.start(Mode.INDEPENDENT)
    t.step()
    assert t.go_left is not None
    racer.connection.disconnect()
    assert t.go_left is None


def test_a_go_of_a_set_time_has_its_timer_left(racer: the_service.Lab) -> None:
    racer.connect()
    t = racer.training
    t.start(Mode.INDEPENDENT)
    seen = []
    for _ in range(TIME + 2):
        t.step()
        seen.append(t.go_left)
    # Down to one, all of it again as the go ends, and down again in the next.
    assert seen[: TIME - 1] == [GoLeft(n, False) for n in range(TIME - 1, 0, -1)]
    assert seen[TIME - 1 :] == [
        GoLeft(TIME, False),
        GoLeft(TIME - 1, False),
        GoLeft(TIME - 2, False),
    ]
    assert [f.record.result.ended_by for f in racer.finished] == ["max_steps"]
    episode = t.episode
    assert episode is not None and episode.steps_left == TIME - 2


def test_a_go_until_the_game_is_over_has_its_wait_left(racer: the_service.Lab) -> None:
    racer.connect()
    t = racer.training
    t.set_go_until_over(True)
    assert t.go_left is None
    t.start(Mode.INDEPENDENT)
    assert t.go_left == GoLeft(TIME, True)
    paid_back = 0
    for _ in range(60):
        t.step()
        episode, left = t.episode, t.go_left
        assert episode is not None and left is not None and left.unpaid
        assert left.steps == episode.unpaid_steps_left <= TIME
        assert episode.steps_left == MOST - episode.steps > left.steps
        paid_back += left.steps == TIME
    assert paid_back >= 2  # the racer paid it, and each time the wait was whole again
    t.stop()
    # The timer when that is the nearer: the most such a go may last.
    t.set_go_length(MOST + 3000)
    t.start(Mode.INDEPENDENT)
    assert t.go_left == GoLeft(MOST, False)
    t.step()
    assert t.go_left == GoLeft(MOST - 1, False)
    t.stop()
    # Both up on the same step: the timer ends the go, so that is what is left.
    t.set_go_length(MOST)
    t.start(Mode.INDEPENDENT)
    assert t.go_left == GoLeft(MOST, False)
    t.stop()
    t.set_go_length(MOST - 1)
    t.start(Mode.INDEPENDENT)
    assert t.go_left == GoLeft(MOST - 1, True)


def test_a_go_the_child_plays_has_it_too(racer: the_service.Lab) -> None:
    racer.connect()
    t = racer.training
    t.start(Mode.MANUAL)
    assert t.go_left == GoLeft(TIME, False)
    for held in (["right"], ["left"], ["right"]):
        racer.play(held, 4)
    assert t.go_left == GoLeft(TIME - 3, False)
