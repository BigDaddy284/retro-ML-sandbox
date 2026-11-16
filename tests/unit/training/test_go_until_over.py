"""T-1044: a go with no timer ends once it has been paid nothing for a set time."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.services import ButtonsHeld, DiscoveryUpdated
from retro_ai_lab.services.rewards import PaidAgain, RewardEarned
from retro_ai_lab.training import EpisodeRunner
from retro_ai_lab.training.episode import EpisodeResult
from retro_ai_lab.training.search import CARRY_ON
from retro_ai_lab.understanding import EventKind, GameEvent
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_remembered_way import planted, remembering
from tests.unit.training.test_search_at_a_loss import GAP_ENDS
from tests.unit.training.test_standing_still import recalling

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

NO_TIMER = 400  # a timer out of reach of these goes


class Paying:
    """Pays the go from the test: what each step of the go is paid, by its number
    (several amounts are several payments on that step). A go here is paid nothing
    else. ``again`` is what a go played straight would have been paid again (T-1028),
    which nobody is paid."""

    def __init__(
        self,
        lab: Lab,
        pay: dict[int, tuple[float, ...]] | None = None,
        again: dict[int, float] | None = None,
    ) -> None:
        self.lab = lab
        self.pay = dict(pay or {})
        self.again = dict(again or {})
        self.step = 0
        lab.bus.subscribe(ButtonsHeld, self)

    def new_go(
        self,
        pay: dict[int, tuple[float, ...]] | None = None,
        again: dict[int, float] | None = None,
    ) -> None:
        self.pay, self.again, self.step = dict(pay or {}), dict(again or {}), 0

    def __call__(self, event: ButtonsHeld) -> None:
        self.step += 1
        cause = GameEvent(EventKind.SCORE_GAIN, event.frame, None, "")
        for amount in self.pay.get(self.step, ()):
            self.lab.bus.publish(
                RewardEarned(event.frame, amount, ("row",), cause, ("Row",), (amount,))
            )
        if self.step in self.again:
            amount = self.again[self.step]
            self.lab.bus.publish(PaidAgain(event.frame, amount, ("row",), (amount,)))


def play(r: EpisodeRunner) -> tuple[int, EpisodeResult]:
    """Play a go to its end: the steps that went on, and the result."""
    r.begin()
    went_on = 0
    while r.step():
        went_on += 1
        assert went_on < 1000
    return went_on, r.finish()


def how(result: EpisodeResult) -> tuple[str, int, float]:
    return result.ended_by, result.steps, result.reward


# ------------------------------------------------------------ the count


def test_without_it_a_go_is_the_go_it_was(lab: Lab) -> None:
    lab.teach()
    Paying(lab, {3: (5.0,), 7: (5.0,)})
    went_on, result = play(lab.runner(max_steps=30))
    assert went_on == 29 and how(result) == ("max_steps", 30, 10.0)
    # Asked for by name, none is none.
    lab.game.reset()
    went_on, result = play(lab.runner(max_steps=30, gain_within=None))
    assert went_on == 29 and how(result) == ("max_steps", 30, 0.0)


def test_a_go_ends_when_it_has_been_paid_nothing_for_that_long(lab: Lab) -> None:
    lab.teach()
    paying = Paying(lab, {3: (5.0,), 7: (5.0,)})
    r = lab.runner(max_steps=NO_TIMER, gain_within=10)
    r.begin()
    steps(r, 16)  # ten steps after the last pay it is still going
    assert r.running and r.steps == 16
    assert not r.step()  # and the seventeenth is its last
    assert how(r.finish()) == ("no_gain", 17, 10.0)
    # A go that is never paid lasts that many steps.
    lab.game.reset()
    paying.new_go()
    went_on, result = play(r)
    assert went_on == 9 and how(result) == ("no_gain", 10, 0.0)
    # Paid on the very step the count would run out, it has as many again.
    lab.game.reset()
    paying.new_go({10: (1.0,)})
    assert how(play(r)[1]) == ("no_gain", 20, 1.0)
    # One step is the least: a go paid on every step but one ends on that one.
    lab.game.reset()
    paying.new_go({step: (2.0,) for step in (1, 2, 3, 5)})
    assert how(play(lab.runner(max_steps=NO_TIMER, gain_within=1))[1]) == ("no_gain", 4, 6.0)


def test_only_what_it_is_paid_starts_the_count_again(lab: Lab) -> None:
    lab.teach()
    # A penalty is no gain.
    paying = Paying(lab, {3: (5.0,), 8: (-3.0,)})
    r = lab.runner(max_steps=NO_TIMER, gain_within=10)
    assert how(play(r)[1]) == ("no_gain", 13, 2.0)
    # A step that costs more than it pays is none either; one that pays more is.
    lab.game.reset()
    paying.new_go({3: (5.0,), 8: (4.0, -6.0)})
    assert how(play(r)[1]) == ("no_gain", 13, 3.0)
    lab.game.reset()
    paying.new_go({3: (5.0,), 8: (4.0, -3.0)})
    assert how(play(r)[1]) == ("no_gain", 18, 6.0)
    # A step that pays what it costs has gained nothing.
    lab.game.reset()
    paying.new_go({3: (5.0,), 8: (4.0, -4.0)})
    assert how(play(r)[1]) == ("no_gain", 13, 5.0)
    # What a go played straight would have been paid again is nobody's pay (T-1028).
    lab.game.reset()
    paying.new_go({3: (5.0,)}, again={8: 5.0})
    assert how(play(r)[1]) == ("no_gain", 13, 5.0)


def test_every_go_counts_from_its_own_start(lab: Lab) -> None:
    lab.teach()
    paying = Paying(lab, {9: (5.0,)})
    r = lab.runner(max_steps=NO_TIMER, gain_within=10)
    assert how(play(r)[1]) == ("no_gain", 19, 5.0)
    # The next go of the same runner is not nine steps to the good.
    lab.game.reset()
    paying.new_go()
    assert how(play(r)[1]) == ("no_gain", 10, 0.0)


# ------------------------------------------------------------ what comes before it


def test_the_game_starting_over_and_the_timer_come_first(lab: Lab) -> None:
    lab.teach()
    r = lab.runner(max_steps=NO_TIMER, gain_within=10)
    r.begin()
    steps(r, 9)
    # The game starts over on the step the count runs out: that is what ended the go.
    reset = GameEvent(EventKind.RESET, 1, None, "Everything went back to where it started.")
    lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))
    assert not r.step()
    result = r.finish()
    assert (result.ended_by, result.steps) == ("reset", 10)
    # A timer that is up sooner ends the go as it did, and one up on the same step too.
    for timer, end in ((8, ("max_steps", 8)), (10, ("max_steps", 10)), (11, ("no_gain", 10))):
        lab.game.reset()
        result = play(lab.runner(max_steps=timer, gain_within=10))[1]
        assert (result.ended_by, result.steps) == end


def test_the_steps_from_memory_do_not_count(lab: Lab) -> None:
    paying = Paying(lab)
    walk, stand = lab.move("right"), lab.move()

    def a_go(pay: dict[int, tuple[float, ...]]) -> tuple[EpisodeRunner, int, EpisodeResult]:
        """A go that plays a remembered way of 20 steps and then stands about."""
        r, route, harness = remembering(lab, max_steps=NO_TIMER, gain_within=12)
        planted(route, harness, [walk] * 20)
        lab.agent.script.extend([stand] * 50)
        paying.new_go(pay)
        asked = lab.agent.asked
        r.begin()
        steps(r, 20)  # the whole way, past where the count would have run out
        assert (r.remembered, r.steps, r.own_steps, lab.agent.asked) == (20, 20, 0, asked)
        went_on = 0
        while r.step():
            went_on += 1
        assert lab.agent.asked - asked == r.own_steps  # the learner chose its own steps
        return r, went_on, r.finish()

    r, went_on, result = a_go({})
    assert (went_on, r.own_steps) == (11, 12)
    assert (result.ended_by, result.steps, result.remembered) == ("no_gain", 32, 20)
    # Paid on the way it remembers, it still has all of its own steps after it, and
    # no more than those.
    r, went_on, result = a_go({5: (3.0,), 20: (3.0,)})
    assert (went_on, result.ended_by, result.steps, r.own_steps) == (11, "no_gain", 32, 12)
    # And paid on its own fourth step it has twelve more.
    r, went_on, result = a_go({24: (3.0,)})
    assert (went_on, result.ended_by, result.steps, r.own_steps) == (15, "no_gain", 36, 16)


def test_a_way_just_found_is_played_to_its_end_first(lab: Lab) -> None:
    # A go on the practice game with the screen watched is paid for new ground: here
    # last on its 23rd step, before it walks into the first gap on its 36th. It goes
    # back and finds the way across, 56 steps of it. Twenty steps with no pay are up
    # on the 43rd step, while the way is still going back over old ground.
    r, route, _ = recalling(lab, max_steps=NO_TIMER, gain_within=20)
    r.begin()
    steps(r, DIED_AT)
    assert (r.ways_sought, r.ways_found, r.own_steps) == (1, 1, DIED_AT)
    steps(r, 56)  # every move of the way is played
    assert r.steps == DIED_AT + 56 == 2 * DIED_AT + CARRY_ON
    assert lab.game.state.x > GAP_ENDS and lab.game.state.lives == toy_hopper.START_LIVES
    # Beyond the gap it was paid again, the last time on its 85th step, and it ends
    # twenty steps after that.
    played = r.steps
    while r.step():
        played += 1
        assert played < 200
    result = r.finish()
    assert (result.ended_by, result.steps) == ("no_gain", 105) and played == 104
    assert len(route.moves) > 23  # and the way is remembered


# ------------------------------------------------------------ nonsense


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, False, "ten", [10]])
def test_a_count_that_is_no_whole_number_of_steps_is_refused(lab: Lab, bad: object) -> None:
    lab.teach()
    with pytest.raises(ConfigurationError, match="gain_within must be a whole number of steps"):
        lab.runner(gain_within=bad)
