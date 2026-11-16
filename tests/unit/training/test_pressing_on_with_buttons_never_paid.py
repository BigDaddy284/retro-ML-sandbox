"""T-1048: pressing on with the button paid most often, then the buttons never paid."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.search import Search
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_go_until_over import Paying
from tests.unit.training.test_going_back import Lab, steps
from tests.unit.training.test_pressing_on import until_pressed
from tests.unit.training.test_search_at_a_loss import searching

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LOGGER = "retro_ai_lab.training.episode"
PRESSES = (
    "The AI has been paid nothing for 8 steps, so it presses on with {} for 4 steps. "
    "It does this where it can go back: if that costs a life, it looks for the way past."
)
TRIES = (
    "The AI has been paid nothing for 8 steps, and the buttons it has been paid for got "
    "it nothing either, so it tries {}, which it has not been paid for yet, for 4 steps."
)


def paid_go(lab: Lab, paying: Paying, r: EpisodeRunner, moves: list[int]) -> None:
    """A go of these moves, paid by the test alone."""
    lab.game.reset()
    lab.agent.script.clear()
    lab.agent.script.extend(moves)
    r.begin()
    steps(r, len(moves))
    r.finish()
    paying.new_go()


def dithering(lab: Lab) -> EpisodeRunner:
    """A go that jumps on, walks on and then walks back over its own ground for good:
    "right" and "a" have been paid, right on the more steps, and nothing is paid now.
    (This lab's brain is told its place in wide steps only, so pottering about one spot
    would count as nothing changing and end the go: not what is tested.)"""
    r, _ = searching(lab, press_after=8, press_for=4, unstick_after=None)
    lab.agent.script.clear()
    lab.agent.script.extend(
        [lab.move("a", "right")] * 4 + [lab.move("right")] * 10 + [lab.move("left")] * 300
    )
    r.begin()
    steps(r, 14)
    assert r.paid_buttons == ("right", "a")
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


# ------------------------------------------------------------ which button is first


def test_the_button_paid_most_often_is_first(lab: Lab) -> None:
    lab.teach()
    paying = Paying(lab, {1: (50.0,), 2: (1.0,), 3: (1.0,)})
    r = lab.runner()
    assert r.paid_buttons == ()
    # One large sum for a jump on the spot, then two small ones for walking on.
    paid_go(lab, paying, r, [lab.move("a"), lab.move("right"), lab.move("right")])
    assert r.paid_buttons == ("right", "a")
    # A move that holds two buttons counts for both of them.
    paying.new_go({1: (1.0,)})
    paid_go(lab, paying, r, [lab.move("a", "right")])
    assert r.paid_buttons == ("right", "a")  # three steps against two
    paying.new_go({1: (1.0,), 2: (1.0,)})
    paid_go(lab, paying, r, [lab.move("a"), lab.move("a")])
    assert r.paid_buttons == ("a", "right")  # four steps against three


def test_equal_counts_go_by_the_sum_and_then_the_name(lab: Lab) -> None:
    lab.teach()
    paying = Paying(lab, {1: (5.0,), 2: (7.0,), 3: (5.0,)})
    r = lab.runner()
    paid_go(lab, paying, r, [lab.move("b"), lab.move("right"), lab.move("a")])
    # One step each: the larger sum first, and the name where the sums are the same.
    assert r.paid_buttons == ("right", "a", "b")
    # Points taken away are no pay, and a step that is paid nothing counts for nothing.
    paying.new_go({1: (-9.0,), 3: (2.0,)})
    paid_go(lab, paying, r, [lab.move("left"), lab.move("up"), lab.move("b")])
    assert r.paid_buttons == ("b", "right", "a")  # b on two steps now


def test_the_count_is_kept_over_the_goes(lab: Lab) -> None:
    lab.teach()
    paying = Paying(lab, {1: (1.0,), 2: (1.0,)})
    r = lab.runner()
    paid_go(lab, paying, r, [lab.move("a"), lab.move("a")])
    assert r.paid_buttons == ("a",)
    paying.new_go({1: (100.0,)})
    paid_go(lab, paying, r, [lab.move("right")])
    assert r.paid_buttons == ("a", "right")  # two steps against one, whatever the sums
    paying.new_go({1: (1.0,), 2: (1.0,)})
    paid_go(lab, paying, r, [lab.move("right"), lab.move("right")])
    assert r.paid_buttons == ("right", "a")  # three against two: the goes before count


# ------------------------------------------------------------ the round


def test_after_the_paid_buttons_the_ones_never_paid_take_their_turn(lab: Lab) -> None:
    r = dithering(lab)
    walk = lab.move("right")
    # The paid buttons first, as before; then each button never paid that has a move
    # of its own, in the controller's order; then round again from the first.
    round_ = [walk, lab.move("a"), lab.move("up"), lab.move("down"), lab.move("left")]
    round_ += [lab.move("b")]
    assert [b.name for b in toy_hopper.BUTTONS] == [
        "up",
        "down",
        "left",
        "right",
        "a",
        "b",
        "start",
    ]
    held = [pressed(r, times) for times in range(1, 9)]
    assert held == [(move, 0.0) for move in round_ + round_[:2]]
    # "start" has no move of its own, and is never pressed on with.
    assert all("start" not in lab.space.buttons_for(move) for move, _ in held)
    assert r.paid_buttons == ("right", "a")  # none of it was paid: they are who they were
    assert lab.game.state.lives == toy_hopper.START_LIVES and r.ways_sought == 0
    assert r.finish().pressed_on == 8


def test_any_pay_starts_the_round_again(lab: Lab) -> None:
    r = dithering(lab)
    paying = Paying(lab)
    walk, back = lab.move("right"), lab.move("left")
    assert pressed(r, 1) == (walk, 0.0)
    assert pressed(r, 2) == (lab.move("a"), 0.0)
    # Paid on a step of its own choosing (walking back): the next press is the first
    # button's again, not the turn of the first button never paid.
    assert r.step() and (r.last_action, r.last_reward) == (back, 0.0)
    paying.pay[paying.step + 1] = (0.5,)
    assert r.step() and (r.last_action, r.last_reward) == (back, 0.5)
    assert r.paid_buttons == ("right", "a", "left")
    # And "left", paid now, has its turn among the paid buttons: the ones never paid
    # come after it, and it is not held twice in a round.
    round_ = [walk, lab.move("a"), back, lab.move("up"), lab.move("down"), lab.move("b"), walk]
    assert [pressed(r, times) for times in range(3, 10)] == [(move, 0.0) for move in round_]


def test_a_button_never_paid_that_is_paid_is_a_paid_button(lab: Lab) -> None:
    r = dithering(lab)
    paying = Paying(lab)
    walk, up = lab.move("right"), lab.move("up")
    assert pressed(r, 1) == (walk, 0.0)
    assert pressed(r, 2) == (lab.move("a"), 0.0)
    # The third press holds "up", never paid; its second step is paid.
    until_pressed(r, 3)
    assert r.last_action == up and r.paid_buttons == ("right", "a")
    paying.pay[paying.step + 1] = (0.5,)
    assert r.step() and (r.last_action, r.last_reward) == (up, 0.5)
    assert r.paid_buttons == ("right", "a", "up")
    steps(r, 2)  # the rest of the press
    assert r.last_action == up
    # The round starts again from the first; "up" is third in it now, held once.
    round_ = [walk, lab.move("a"), up, lab.move("down"), lab.move("left"), lab.move("b"), walk]
    assert [pressed(r, times) for times in range(4, 11)] == [(move, 0.0) for move in round_]


def test_nothing_paid_yet_no_press(lab: Lab) -> None:
    # Standing at the start nothing is ever paid: there is no button to begin with,
    # and the buttons never paid are not pressed on with either.
    r, _ = searching(lab, press_after=8, press_for=4, unstick_after=None, max_steps=80)
    stand = lab.move()
    lab.agent.script.extend([stand] * 80)
    r.begin()
    while r.step():
        assert r.last_action == stand and r.last_reward <= 0
    result = r.finish()
    assert r.paid_buttons == ()
    assert (result.ended_by, result.steps, result.pressed_on) == ("max_steps", 80, 0)


# ------------------------------------------------------------ what a person reads


def test_the_log_says_when_it_tries_a_button_never_paid(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r = dithering(lab)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        for times in range(1, 9):
            pressed(r, times)
    said = [
        (rec.levelno, rec.getMessage())
        for rec in caplog.records
        if "paid nothing for" in rec.getMessage()
    ]
    info, detail = logging.INFO, logging.DEBUG
    assert said == [
        (info, PRESSES.format("right")),  # the first press of all: the sentence it was
        (detail, PRESSES.format("a")),
        (info, TRIES.format("up")),  # the first with a button never paid, said once
        (detail, TRIES.format("down")),
        (detail, TRIES.format("left")),
        (detail, TRIES.format("b")),
        (detail, PRESSES.format("right")),
        (detail, PRESSES.format("a")),
    ]


# ------------------------------------------------------------ the search's plain way


def test_the_plain_way_is_the_button_paid_most_often(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = episode.find_way
    given: list[object] = []

    def spy(*args: object, **kw: object) -> Search:
        given.append(kw.get("plain"))
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(episode, "find_way", spy)
    r, _ = searching(lab)
    # One jump on the spot is paid a fortune; walking on is paid a little, step after
    # step, until the gap takes the life.
    Paying(lab, {1: (5000.0,)})
    lab.agent.script.append(lab.move("a"))
    r.begin()
    while r.ways_sought == 0:
        assert r.step()
    assert r.paid_buttons == ("right", "a")
    assert given == [lab.move("right")]
