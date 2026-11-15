"""T-1032: asking for the last chance from a step known to be too late, and what the
asking gives when its time runs out."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training import chance
from retro_ai_lab.training.chance import Chance
from retro_ai_lab.training.trial import Outcome
from tests.unit.training.test_chance import JUMP, LENGTH, PAD, TRACE, WALK, Clock, Game, ask, until

OTHERS = len(PAD) - 1  # the moves asked before a step is called too late


def asked_at(game: Game) -> list[int]:
    """The step each play after the first made its other move from."""
    return [next(i for i, move in enumerate(moves) if move != WALK) for moves in game.calls[1:]]


def test_the_halving_starts_from_the_step_given(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Chance(3, 9) == Chance(3, 9, sure=True) != Chance(3, 9, sure=False)
    without = within = fewer = more = 0
    for step in range(LENGTH + 1):
        plain = Game(until(step))
        monkeypatch.setattr(chance, "play", plain)
        was = ask()
        assert was == Chance(step, len(plain.calls))
        for shut in range(max(step, 1), LENGTH + 1):
            game = Game(until(step))
            monkeypatch.setattr(chance, "play", game)
            got = ask(shut=shut)
            # The same step, surely, with no step from there on asked.
            assert got == Chance(step, len(game.calls)), (step, shut)
            assert all(at < shut for at in asked_at(game)), (step, shut)
            if shut == LENGTH:  # nothing is known: the asking as it was, play for play
                assert game.calls == plain.calls
            without += len(plain.calls)
            within += len(game.calls)
            fewer += len(game.calls) < len(plain.calls)
            more += len(game.calls) > len(plain.calls)
    # Over every step and everything that can be known of it, a quarter of the plays
    # are saved. Not in every case: halving from another end can land worse.
    assert (without, within, fewer, more) == (38260, 29315, 588, 99)
    # It is taken at its word: a step given too early is the answer, with nothing from
    # it on asked.
    game = Game(until(30))
    monkeypatch.setattr(chance, "play", game)
    assert ask(shut=20) == Chance(20, len(game.calls))
    assert max(asked_at(game)) == 19
    # Too late from the second step on: the first step is all there is to ask.
    game = Game(until(1))
    monkeypatch.setattr(chance, "play", game)
    assert ask(shut=1) == Chance(1, 2) and asked_at(game) == [0]
    game = Game(until(0))
    monkeypatch.setattr(chance, "play", game)
    assert ask(shut=1) == Chance(0, 1 + OTHERS) and asked_at(game) == [0] * OTHERS
    # Far fewer plays where the step lies just before it: 13 of 40 with 14 given.
    game = Game(until(13))
    monkeypatch.setattr(chance, "play", game)
    assert ask(shut=14) == Chance(13, len(game.calls)) and len(game.calls) < 46 / 2


def test_out_of_time_it_gives_the_latest_step_it_knows(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed(**kw: object) -> tuple[Chance | None, int]:
        game = Game(until(13))
        clock = Clock(game)  # a second passes with every play
        monkeypatch.setattr(chance, "play", clock.play)
        kw.setdefault("partial", True)
        return ask(clock=clock, **kw), len(game.calls)

    assert PAD.near(WALK)[0] == JUMP  # so a step that is not too late costs one play
    # With nothing given: what it did (1 play), the first step (1), step 20 (14, too
    # late), step 10 (1), step 15 (14, too late), step 12 (1), step 13 (14, too late).
    assert timed() == (Chance(13, 46), 46)
    for seconds in range(1, 50):
        got, plays = timed(seconds=seconds)
        assert plays == min(seconds, 46)
        if seconds < 16:
            # Nothing is known yet. A step whose asking the time cut short counts for
            # nothing, with 13 of its 14 moves asked too.
            assert got is None, seconds
        elif seconds < 31:
            assert got == Chance(20, seconds, sure=False), seconds
        elif seconds < 46:
            assert got == Chance(15, seconds, sure=False), seconds
        else:
            assert got == Chance(13, 46, sure=True), seconds
    # With step 20 given as too late: what it did (1), the first step (1), step 10 (1),
    # step 15 (14), step 12 (1), step 13 (14). From the first play on something is known.
    assert timed(shut=20) == (Chance(13, 32), 32)
    for seconds in range(1, 36):
        got, plays = timed(seconds=seconds, shut=20)
        assert plays == min(seconds, 32)
        if seconds < 17:
            assert got == Chance(20, seconds, sure=False), seconds
        elif seconds < 32:
            assert got == Chance(15, seconds, sure=False), seconds
        else:
            assert got == Chance(13, 32, sure=True), seconds
    # The length given is nothing known.
    assert timed(seconds=5, shut=LENGTH) == (None, 5)
    # The first step found too late in time is sure, however little time was left.
    game = Game(until(0))
    clock = Clock(game)
    monkeypatch.setattr(chance, "play", clock.play)
    assert ask(clock=clock, seconds=1 + OTHERS, shut=20) == Chance(0, 1 + OTHERS)
    game = Game(until(0))
    clock = Clock(game)
    monkeypatch.setattr(chance, "play", clock.play)
    assert ask(clock=clock, seconds=OTHERS, shut=20, partial=True) == Chance(20, OTHERS, False)
    # What it did was not played again: nothing is known of this save at all, whatever
    # was given. (This clock moves a second each time it is read.)
    game = Game(until(13))
    monkeypatch.setattr(chance, "play", game)
    reads = iter(range(100, 200))
    assert ask(clock=lambda: float(next(reads)), seconds=0.5, shut=20, partial=True) is None
    assert game.calls == []
    # Unless it is asked for, out of time is None whatever is known, as it was (T-1031).
    for seconds in (1, 20, 31):
        assert timed(seconds=seconds, shut=20, partial=False) == (None, seconds)
        assert timed(seconds=seconds, partial=False) == (None, seconds)
    assert timed(seconds=32, shut=20, partial=False) == (Chance(13, 32), 32)
    # Played again it does not lose the life on that step: None, as ever, at once.
    game = Game(lambda moves: Outcome(steps=LENGTH, lost=False, ground=0))
    monkeypatch.setattr(chance, "play", game)
    assert ask(shut=20, partial=True) is None and len(game.calls) == 1


def test_misuse_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    game = Game(lambda moves: None)
    monkeypatch.setattr(chance, "play", game)
    for bad in (0, -1, True, 2.5, "3", LENGTH + 1):
        with pytest.raises(ConfigurationError, match="shut must be a whole number from 1 to"):
            ask(shut=bad)
    for bad in (1, 0, "yes", None):
        with pytest.raises(ConfigurationError, match="partial must be true or false"):
            ask(partial=bad)
    assert game.calls == []  # nothing was played
    assert ask(TRACE, shut=LENGTH) == Chance(0, 1 + OTHERS)
