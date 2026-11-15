"""T-1033: asking for the last chance with a guess at where it lies."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training import chance
from retro_ai_lab.training.chance import WALK as WALKED
from retro_ai_lab.training.chance import Chance
from tests.unit.training.test_chance import BACK, JUMP, LENGTH, PAD, WALK, Clock, Game, ask, until
from tests.unit.training.test_chance_shut import OTHERS, asked_at


def asking(monkeypatch: pytest.MonkeyPatch, step: int, move: int = JUMP, **kw: object):  # type: ignore[no-untyped-def]
    game = Game(until(step, move))
    monkeypatch.setattr(chance, "play", game)
    return ask(**kw), game


def test_a_guess_never_changes_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    assert WALKED == 8 and PAD.near(WALK)[0] == JUMP
    without = within = fewer = more = 0
    for step in range(LENGTH + 1):
        for shut in (None, *sorted({max(step, 1), min(step + 3, LENGTH), LENGTH})):
            was, plain = asking(monkeypatch, step, shut=shut)
            assert was == Chance(step, len(plain.calls))
            # No guess is the asking as it was, play for play.
            assert asking(monkeypatch, step, shut=shut, guess=None)[1].calls == plain.calls
            for guess in range(1, LENGTH if shut is None else shut):
                got, game = asking(monkeypatch, step, shut=shut, guess=guess)
                # The same step, surely, wherever the guess lay.
                assert got == Chance(step, len(game.calls)), (step, shut, guess)
                assert game.calls[0] == plain.calls[0]
                assert asked_at(game)[0] == guess
                if shut is not None:
                    assert all(at < shut for at in asked_at(game)), (step, shut, guess)
                if shut is None:
                    without += len(plain.calls)
                    within += len(game.calls)
                    fewer += len(game.calls) < len(plain.calls)
                    more += len(game.calls) > len(plain.calls)
    # A guess from anywhere is no saving: most guesses are far off, and one that is
    # too late costs a play for every other move before anything is known.
    assert (without, within, fewer, more) == (62829, 64549, 575, 888)
    # The saving is in a guess that is right or a little early.
    plain = good = 0
    for step in range(2, LENGTH + 1):
        plain += len(asking(monkeypatch, step)[1].calls)
        good += len(asking(monkeypatch, step, guess=step - 1)[1].calls)
    # 39 plays a step without, 16 with (and 2 where the last step still saves).
    assert (plain, good) == (1524, 38 * (2 + OTHERS) + 2)


def test_a_good_guess_costs_a_walk_and_one_step_too_late(monkeypatch: pytest.MonkeyPatch) -> None:
    # The guess on the last step that still changes it: what it did (1 play), that
    # step (1: the first other move asked gets out), the next (every other move).
    got, game = asking(monkeypatch, 30, guess=29)
    assert got == Chance(30, 2 + OTHERS) and len(game.calls) == 2 + OTHERS
    assert asked_at(game) == [29, *[30] * OTHERS]
    assert len(asking(monkeypatch, 30)[1].calls) == 7 + OTHERS  # without a guess: 21
    # Where a move far from the one made is the only one that gets out, the step that
    # still changes it costs the moves asked before that one; the next, all of them.
    order = [
        *PAD.near(WALK),
        *(m for m in range(len(PAD)) if m != WALK and m not in PAD.near(WALK)),
    ]
    got, game = asking(monkeypatch, 30, BACK, guess=29)
    assert got == Chance(30, len(game.calls))
    assert asked_at(game) == [*[29] * (order.index(BACK) + 1), *[30] * OTHERS]
    # A guess that is early: one step at a time from it, nothing halved.
    for early in range(1, WALKED + 1):
        got, game = asking(monkeypatch, 30, guess=30 - early)
        assert got == Chance(30, 1 + early + OTHERS), early
        assert asked_at(game) == [*range(30 - early, 30), *[30] * OTHERS], early
    # Earlier than it walks: the walk ends with nothing found too late, and the
    # halving goes on between its last step and the end.
    got, game = asking(monkeypatch, 30, guess=29 - WALKED)
    assert got == Chance(30, len(game.calls))
    at = asked_at(game)
    assert at[: WALKED + 1] == list(range(29 - WALKED, 30))
    assert at[WALKED + 1 :] == [*[34] * OTHERS, *[31] * OTHERS, *[30] * OTHERS]
    # The guessed step is too late itself: every other move is asked there, and the
    # halving goes on before it, the first step first as ever.
    got, game = asking(monkeypatch, 30, guess=35)
    assert got == Chance(30, len(game.calls))
    assert asked_at(game) == [*[35] * OTHERS, 0, 17, 26, *[30] * OTHERS, 28, 29]
    got, game = asking(monkeypatch, 30, guess=30)
    assert got == Chance(30, len(game.calls))
    assert asked_at(game) == [*[30] * OTHERS, 0, 15, 22, 26, 28, 29]
    # Too late from the first step on, with a guess: the guess and the first step.
    got, game = asking(monkeypatch, 0, guess=5)
    assert got == Chance(0, 1 + 2 * OTHERS)
    assert asked_at(game) == [*[5] * OTHERS, *[0] * OTHERS]
    # The last step could still save the life: the guess on it is all there is to ask.
    got, game = asking(monkeypatch, LENGTH, guess=LENGTH - 1)
    assert got == Chance(LENGTH, 2) and asked_at(game) == [LENGTH - 1]
    # No step known to be too late is asked on the walk, and the one before it found
    # in time is the answer with nothing more asked.
    got, game = asking(monkeypatch, 31, guess=29, shut=31)
    assert got == Chance(31, 3) and asked_at(game) == [29, 30]
    got, game = asking(monkeypatch, 31, guess=29, shut=33)
    assert got == Chance(31, 3 + OTHERS) and asked_at(game) == [29, 30, *[31] * OTHERS]
    got, game = asking(monkeypatch, 33, guess=29, shut=33)
    assert got == Chance(33, 5) and asked_at(game) == [29, 30, 31, 32]


def test_out_of_time_with_a_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed(**kw: object) -> tuple[Chance | None, int]:
        game = Game(until(13))
        clock = Clock(game)  # a second passes with every play
        monkeypatch.setattr(chance, "play", clock.play)
        kw.setdefault("partial", True)
        return ask(clock=clock, **kw), len(game.calls)

    # Guessed three steps early: what it did (1), steps 10, 11 and 12 (1 each), step
    # 13 (14, too late). Until that one is done nothing is known.
    whole = 4 + OTHERS
    assert timed(guess=10) == (Chance(13, whole), whole)
    for seconds in range(1, whole + 3):
        got, plays = timed(seconds=seconds, guess=10)
        assert plays == min(seconds, whole)
        assert got == (None if seconds < whole else Chance(13, whole)), seconds
        # With a step given as too late, that one is known from the first play on.
        got, plays = timed(seconds=seconds, guess=10, shut=20)
        assert got == Chance(*((20, seconds, False) if seconds < whole else (13, whole))), seconds
        assert timed(seconds=seconds, guess=10, partial=False)[0] == (
            None if seconds < whole else Chance(13, whole)
        )
    # Guessed too late, at step 20: what it did (1), step 20 (14, too late), the first
    # step (1), step 10 (1), step 15 (14, too late), step 12 (1), step 13 (14, too late).
    whole = 4 + 3 * OTHERS
    assert timed(guess=20) == (Chance(13, whole), whole)
    for seconds in range(1, whole + 3):
        got, plays = timed(seconds=seconds, guess=20)
        assert plays == min(seconds, whole)
        if seconds < 1 + OTHERS:
            assert got is None, seconds
        elif seconds < 3 + 2 * OTHERS:
            assert got == Chance(20, seconds, sure=False), seconds
        elif seconds < whole:
            assert got == Chance(15, seconds, sure=False), seconds
        else:
            assert got == Chance(13, whole), seconds
    # The walk ended with nothing found too late (guess 1, step 13 further on than it
    # walks): nothing is known until the halving finds a step.
    got, plays = timed(seconds=2 + WALKED, guess=1)
    assert (got, plays) == (None, 2 + WALKED)


def test_misuse_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    game = Game(lambda moves: None)
    monkeypatch.setattr(chance, "play", game)
    for bad in (0, -1, True, 2.5, "3", LENGTH, LENGTH + 1):
        with pytest.raises(ConfigurationError, match="guess must be a whole number from 1 to"):
            ask(guess=bad)
    for bad in (20, 21):  # at or past the step known to be too late
        with pytest.raises(ConfigurationError, match="guess must be a whole number from 1 to"):
            ask(guess=bad, shut=20)
    with pytest.raises(ConfigurationError, match="guess must be a whole number from 1 to"):
        ask([WALK], guess=1)  # one step: there is none to guess
    assert game.calls == []  # nothing was played
    assert ask(guess=19, shut=20) == Chance(0, 1 + 2 * OTHERS)  # a game where nothing saves
