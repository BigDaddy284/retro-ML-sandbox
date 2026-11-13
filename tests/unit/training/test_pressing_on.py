"""T-1011: a go that has been paid nothing for a while presses on, where it can go back."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import EpisodeRunner
from retro_ai_lab.training.episode import DEFAULT_UNSTICK_AFTER, PRESS_FOR, EpisodeResult
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_search_at_a_loss import GAP_ENDS, searching

lab = going_back.lab  # the practice game behind the real services (pytest fixture)


def until_pressed(r: EpisodeRunner, times: int, most: int = 120) -> list[float]:
    """Step until the go has pressed on ``times`` times. Returns what each step was paid."""
    paid: list[float] = []
    while r.pressed_on < times:
        assert len(paid) < most, f"No press in {most} steps."
        assert r.step()
        paid.append(r.last_reward)
    return paid


def test_a_go_paid_nothing_presses_on_with_the_best_paid_button(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, _ = searching(lab, press_after=8, press_for=6)
    walk, stand = lab.move("right"), lab.move()
    lab.agent.script.extend([walk] * 8 + [stand] * 60)
    assert (r.press_after, r.pressed_on, r.paid_buttons) == (8, 0, ())
    r.begin()
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        paid = until_pressed(r, 1)
    assert r.paid_buttons == ("right",)  # the walk was paid for its new ground
    # The press comes on the step after eight that were paid nothing, and not sooner.
    *before, _ = paid
    assert all(p <= 0 for p in before[-8:]) and before[-9] > 0
    # Six steps of the best-paid button, without asking the learner.
    asked = lab.agent.asked
    played = [r.last_action]
    for _ in range(5):
        assert r.step()
        played.append(r.last_action)
    assert played == [walk] * 6 and lab.agent.asked == asked
    assert r.step() and r.last_action == stand and lab.agent.asked == asked + 1
    # They are steps like any other: counted, and learned from.
    count = r.steps
    result = r.finish()
    assert (result.steps, result.pressed_on) == (count, 1)
    taught = [action for _, action, *_ in lab.agent.learned]
    assert len(taught) == count and taught[len(before) : len(before) + 6] == [walk] * 6
    said = [rec.getMessage() for rec in caplog.records if "presses on" in rec.getMessage()]
    assert said == [
        "The AI has been paid nothing for 8 steps, so it presses on with right for 6 steps. "
        "It does this where it can go back: if that costs a life, it looks for the way past."
    ]


def test_pressed_on_into_the_gap_it_goes_back_and_gets_across(lab: Lab) -> None:
    r, _ = searching(lab, press_after=8)
    walk, stand = lab.move("right"), lab.move()
    lab.agent.script.extend([walk] * 8 + [stand] * 400)
    r.begin()
    assert PRESS_FOR == 16
    until_pressed(r, 1)
    assert lab.game.state.x < toy_hopper.TILE * lab.game.gaps[0][0]  # short of the gap still
    while r.ways_sought == 0:
        assert r.step()
    # The press walked it in; the life went, and going back found the way.
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    while lab.game.state.x <= GAP_ENDS:
        assert r.step()
        assert lab.game.state.lives == toy_hopper.START_LIVES
    # Its one search is used: beyond the gap it cannot go back, and does not press on.
    pressed = r.pressed_on
    while r.step():
        pass
    result = r.finish()
    assert result.ended_by == "stuck" and result.pressed_on == pressed
    assert (result.ways_sought, result.ways_found) == (1, 1)


def test_it_does_not_press_on_where_it_cannot_go_back(lab: Lab) -> None:
    walk, stand = lab.move("right"), lab.move()

    def stands_still(r: EpisodeRunner) -> EpisodeResult:
        lab.game.reset()
        lab.agent.script.clear()
        lab.agent.script.extend([walk] * 8 + [stand] * 400)
        r.begin()
        while r.step():
            assert r.last_action in (walk, stand)
        assert lab.game.state.x < toy_hopper.TILE * lab.game.gaps[0][0]
        return r.finish()

    # Switched off (the default), with no search allowed, and with no moments kept.
    off, _ = searching(lab)
    none_allowed, _ = searching(lab, press_after=8, ways_per_go=0)
    assert off.press_after == 0
    for r in (off, none_allowed, lab.runner(watch=lab.watch(), press_after=8)):
        result = stands_still(r)
        assert (result.ended_by, result.pressed_on, result.ways_sought) == ("stuck", 0, 0)
    # Once the runner has given up going back, it no longer presses on either.
    r, _ = searching(lab, press_after=8, way_tries=1, give_up_after=1)
    first = stands_still(r)
    assert first.pressed_on >= 1 and (first.ways_sought, first.ways_found) == (1, 0)
    assert r.gave_up
    again = stands_still(r)
    assert (again.ended_by, again.pressed_on, again.setbacks) == ("stuck", 0, 0)


def test_a_go_about_to_end_stuck_presses_on_first_once(lab: Lab) -> None:
    # The count of unpaid steps is set out of reach: only standing still can press.
    r, _ = searching(lab, press_after=300)
    walk, stand, back = lab.move("right"), lab.move(), lab.move("left")
    lab.agent.script.extend([walk] * 8 + [stand] * 400)
    r.begin()
    paid = until_pressed(r, 1)
    # Nothing had changed for as long as ends a go as stuck (this lab's brain is told
    # its place in wide steps, so that began while it still walked); the go did not end.
    assert len(paid) > DEFAULT_UNSTICK_AFTER and paid[-1] <= 0
    assert r.last_action == walk and lab.game.state.x < 150
    for _ in range(PRESS_FOR - 1):
        assert r.step() and r.last_action == walk
    assert lab.game.state.x > 190  # walked on, into the gap
    r.finish()
    # A press that changes nothing either leaves the go stuck, as it was before:
    # pushing against the left end of the level, where "left" is all that was paid.
    lab.game.reset()
    r, _ = searching(lab, press_after=300)
    lab.agent.script.clear()
    lab.agent.script.extend([back] * 400)
    r.begin()
    until_pressed(r, 1)
    assert r.paid_buttons == ("left",) and lab.game.state.x == 0
    count = 0
    while r.step():
        count += 1
        assert r.last_action == back
    result = r.finish()
    assert count == PRESS_FOR - 1 and (result.ended_by, result.pressed_on) == ("stuck", 1)


def test_pay_starts_the_count_again_and_a_way_is_not_interrupted(lab: Lab) -> None:
    # Walking right is paid every step or two, with at most five steps between; the
    # twelve steps of falling and dying pay nothing, so the press is set past those.
    r, _ = searching(lab, press_after=13, ways_per_go=2)
    walk, jump = lab.move("right"), lab.move("a", "right")
    r.begin()
    steps(r, DIED_AT - 1)
    assert r.pressed_on == 0 and lab.agent.asked == DIED_AT - 1
    assert r.step() and r.ways_found == 1
    # The way begins with 19 steps over ground walked before, which pay nothing: with
    # a search left it could press on, and it leaves the way alone.
    asked, played = lab.agent.asked, []
    while lab.agent.asked == asked:
        assert r.step()
        played.append(r.last_action)
    played.pop()
    assert played == [walk] * 19 + [jump] * 4 + [walk] * (DIED_AT - 23) + [walk] * 20
    assert r.pressed_on == 0


def test_a_press_that_got_nothing_is_followed_by_the_next_best_button(lab: Lab) -> None:
    walk, jump = lab.move("right"), lab.move("a", "right")
    back, stand = lab.move("left"), lab.move()

    def go(then: int) -> EpisodeRunner:
        # (This lab's brain is told its place in wide steps only, so pottering about
        # one spot would count as nothing changing and end the go: not what is tested.)
        r, _ = searching(lab, press_after=8, press_for=4, unstick_after=None)
        lab.agent.script.clear()
        lab.agent.script.extend([jump] * 4 + [walk] * 10 + [then] * 200)
        r.begin()
        steps(r, 14)
        assert r.paid_buttons == ("right", "a")  # both were paid, right the more
        return r

    def pressed(r: EpisodeRunner, times: int) -> tuple[int, float]:
        """The move of the press, and what its four steps were paid."""
        paid = until_pressed(r, times)
        assert all(p <= 0 for p in paid[-9:-1])
        move, got = r.last_action, r.last_reward
        for _ in range(3):
            assert r.step() and r.last_action == move
            got += r.last_reward
        assert move is not None
        return move, got

    # Walking back over ground it has been on pays nothing, and so does being pressed
    # on over it: the next press is the next-best-paid button's, then round again.
    r = go(back)
    assert pressed(r, 1) == (walk, 0.0)
    assert pressed(r, 2) == (lab.move("a"), 0.0)
    assert pressed(r, 3) == (walk, 0.0)
    assert lab.game.state.lives == toy_hopper.START_LIVES and r.ways_sought == 0
    assert r.finish().pressed_on == 3
    # Standing where it got to, the press walks on to new ground and is paid: the
    # press after it is the best-paid button's again, not the next one's turn.
    r = go(stand)
    move, got = pressed(r, 1)
    assert move == walk and got > 0
    assert pressed(r, 2)[0] == walk


def test_nonsense_is_refused_and_the_record_counts_the_presses(lab: Lab) -> None:
    for wrong in (-1, True, 1.5, "5"):
        with pytest.raises(ConfigurationError, match="press_after"):
            lab.runner(press_after=wrong)
    for wrong in (0, -1, True, 2.5):
        with pytest.raises(ConfigurationError, match="press_for"):
            lab.runner(press_for=wrong)
    result = EpisodeResult(10, 40, 1.0, "max_steps", 3, pressed_on=2)
    assert EpisodeResult.from_dict(result.to_dict()) == result
    older = result.to_dict()
    del older["pressed_on"]
    assert EpisodeResult.from_dict(older).pressed_on == 0
    assert EpisodeResult(10, 40, 1.0, "max_steps", 3).describe_pressing() == ""
    assert EpisodeResult(1, 4, 0.0, "stuck", 1, pressed_on=1).describe_pressing() == (
        ", pressed on once"
    )
    assert result.describe_pressing() == ", pressed on 2 times"
