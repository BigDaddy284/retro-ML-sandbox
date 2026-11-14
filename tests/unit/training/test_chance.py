"""T-1031: from when on a life lost could no longer be saved."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.training import chance
from retro_ai_lab.training.chance import Chance, last_chance
from retro_ai_lab.training.trial import Outcome
from tests.unit.training.test_trial import FRAMES, LIVES, SPOT, Hopper

PAD = ActionSpace.from_buttons(("up", "down", "left", "right", "a", "b"))
WALK = PAD.index_of(["right"])
JUMP = PAD.index_of(["a", "right"])
BACK = PAD.index_of(["left"])
LENGTH = 40
TRACE = [WALK] * LENGTH


class Game:
    """Stands in for ``trial.play`` with a game of one rule: the life goes on step
    ``LENGTH`` unless ``saves`` says of the moves that they get out of it."""

    def __init__(self, saves: Callable[[Sequence[int]], Outcome | None]) -> None:
        self.saves = saves
        self.calls: list[tuple[int, ...]] = []

    def __call__(
        self, connection: object, slot: str, moves: Sequence[int], space: object, **kw: object
    ) -> Outcome:
        assert slot == SPOT and kw == {"frames_per_step": FRAMES, "lives": [LIVES]}
        self.calls.append(tuple(moves))
        got = self.saves(moves)
        return Outcome(steps=LENGTH, lost=True, ground=0) if got is None else got


def ask(trace: Sequence[int] = TRACE, **kw: object) -> Chance | None:
    kw.setdefault("lives", [LIVES])
    kw.setdefault("frames_per_step", FRAMES)
    return last_chance(object(), SPOT, trace, PAD, **kw)  # type: ignore[arg-type]


def until(step: int, move: int = JUMP) -> Callable[[Sequence[int]], Outcome | None]:
    """The rule of a game where ``move`` made at any step before ``step`` saves the
    life, and nothing from that step on does."""

    def saves(moves: Sequence[int]) -> Outcome | None:
        if move in moves[:step]:
            return Outcome(steps=len(moves), lost=False, ground=0)
        return None

    return saves


def test_it_finds_the_step_wherever_it_lies(monkeypatch: pytest.MonkeyPatch) -> None:
    for step in range(LENGTH + 1):  # 0: even the first step is too late; 40: the last saves
        game = Game(until(step))
        monkeypatch.setattr(chance, "play", game)
        got = ask()
        assert got == Chance(live=step, plays=len(game.calls)), step
        # The first play is what it did, as it did it; every other is that up to some
        # step and one other move held from there to the end.
        assert game.calls[0] == tuple(TRACE)
        for moves in game.calls[1:]:
            assert len(moves) == LENGTH
            at = next(i for i, move in enumerate(moves) if move != WALK)
            assert moves[:at] == (WALK,) * at and len(set(moves[at:])) == 1
        # Halving: seven steps at most for forty, and never the whole list of them.
        assert len({moves.index(moves[-1]) for moves in game.calls[1:]}) <= 7
    # A move that loses the life sooner has changed something too (one that loses it
    # later has not lost it yet when the moves are done: that is the rule above).
    for other in (LENGTH - 1, 26):

        def sooner(moves: Sequence[int], other: int = other) -> Outcome | None:
            if BACK in moves[:25]:
                return Outcome(steps=other, lost=True, ground=0)
            return None

        monkeypatch.setattr(chance, "play", Game(sooner))
        found = ask()
        assert found is not None and found.live == 25


def test_every_other_move_is_asked_the_near_ones_first(monkeypatch: pytest.MonkeyPatch) -> None:
    near = PAD.near(WALK)
    others = [move for move in range(len(PAD)) if move != WALK]
    assert JUMP in near and BACK not in near and len(others) == 14
    # The one move that saves the life is far from the one made: asking the near ones
    # only would put the last chance at the start.
    game = Game(until(30, BACK))
    monkeypatch.setattr(chance, "play", game)
    got = ask()
    assert got is not None and got.live == 30
    by_step: dict[int, list[int]] = {}
    for moves in game.calls[1:]:
        at = next(i for i, move in enumerate(moves) if move != WALK)
        by_step.setdefault(at, []).append(moves[at])
    assert set(by_step) == {0, 20, 30, 25, 27, 28, 29}
    for at, asked in by_step.items():
        # The near ones first, then the rest in their order, each once.
        rest = [move for move in others if move not in near]
        if at >= 30:  # too late: every other move was asked before that was said
            assert asked == [*near, *rest]
        else:  # the first that changes it is enough
            assert asked == [*near, *rest][: len(asked)] and asked[-1] == BACK
            assert asked.count(BACK) == 1
    assert got.plays == len(game.calls) == 1 + sum(len(asked) for asked in by_step.values())
    # Whichever the one move is, the first of the pad, the last or any between.
    for move in others:
        monkeypatch.setattr(chance, "play", Game(until(30, move)))
        found = ask()
        assert found is not None and found.live == 30, move
    # And whatever move it made itself: here the pad's last is not near it.
    last = len(PAD) - 1
    assert last not in PAD.near(BACK) and last in near
    for move in (0, last):
        monkeypatch.setattr(chance, "play", Game(until(30, move)))
        found = ask([BACK] * LENGTH)
        assert found is not None and found.live == 30, move
    # With the near move saving it, a step that is not too late costs one play.
    game = Game(until(30, near[0]))
    monkeypatch.setattr(chance, "play", game)
    assert ask() == Chance(30, 1 + 14 + 6)  # what it did, step 30 in full, six steps open
    # The moves near the one made go by that step's own move.
    mixed = [WALK] * 20 + [BACK] * 20
    game = Game(lambda moves: None)
    monkeypatch.setattr(chance, "play", game)
    assert ask(mixed) == Chance(0, 15)
    assert [moves[0] for moves in game.calls[1:]] == [
        *PAD.near(WALK),
        *(m for m in others if m not in PAD.near(WALK)),
    ]
    # One move in all: there is no other to make, so nothing could be changed.
    lone = ActionSpace.from_buttons(())
    assert len(lone) == 1
    assert last_chance(
        object(), SPOT, [0] * LENGTH, lone, frames_per_step=FRAMES, lives=[LIVES]
    ) == (  # type: ignore[arg-type]
        Chance(0, 1)
    )


def test_a_trace_that_does_not_lose_the_life_again_says_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alive = Outcome(steps=LENGTH, lost=False, ground=0)
    sooner = Outcome(steps=LENGTH - 1, lost=True, ground=0)
    for outcome in (alive, sooner):
        game = Game(lambda moves, outcome=outcome: outcome)  # type: ignore[misc]
        monkeypatch.setattr(chance, "play", game)
        assert ask() is None
        assert game.calls == [tuple(TRACE)]  # one play, and no more

    # An emulator error is the caller's.
    def gone(*args: object, **kw: object) -> Outcome:
        raise EmulatorError("gone")

    monkeypatch.setattr(chance, "play", gone)
    with pytest.raises(EmulatorError, match="gone"):
        ask()


class Clock:
    """A clock the test moves: a second passes with every play."""

    def __init__(self, game: Game) -> None:
        self.now = 100.0
        self.game = game

    def play(self, *args: object, **kw: object) -> Outcome:
        self.now += 1.0
        return self.game(*args, **kw)  # type: ignore[arg-type]

    def __call__(self) -> float:
        return self.now


def test_no_play_is_started_once_the_time_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed() -> tuple[Game, Clock]:
        game = Game(until(30))
        clock = Clock(game)
        monkeypatch.setattr(chance, "play", clock.play)
        return game, clock

    game, clock = timed()
    whole = ask(clock=clock)
    assert whole is not None and whole.live == 30
    needs = whole.plays
    assert needs == len(game.calls) > 15
    # A play short: nothing is said, and not one play is started after the time.
    for seconds in (1, 5, needs - 1):
        game, clock = timed()
        assert ask(seconds=seconds, clock=clock) is None
        assert len(game.calls) == seconds
    # The time is looked at before a play, not after: the last play may end on it.
    for seconds in (needs, needs + 0.5, 1000):
        game, clock = timed()
        assert ask(seconds=seconds, clock=clock) == whole
        assert len(game.calls) == needs
    # Half a second is up before the second play.
    game, clock = timed()
    assert ask(seconds=0.5, clock=clock) is None and len(game.calls) == 1


def test_misuse_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    game = Game(lambda moves: None)
    monkeypatch.setattr(chance, "play", game)
    for bad in (
        {"lives": []},
        {"frames_per_step": 0},
        {"frames_per_step": 2.5},
        {"frames_per_step": True},
        {"seconds": 0},
        {"seconds": -1.5},
        {"seconds": "soon"},
        {"seconds": True},
        {"clock": None},
    ):
        with pytest.raises(ConfigurationError):
            ask(**bad)
    with pytest.raises(ConfigurationError, match="nothing it did"):
        ask([])
    with pytest.raises(ConfigurationError):
        ask([len(PAD)])
    with pytest.raises(ConfigurationError, match="name of the save"):
        last_chance(object(), "", TRACE, PAD, frames_per_step=FRAMES, lives=[LIVES])  # type: ignore[arg-type]
    assert game.calls == []  # nothing was played


# ------------------------------------------------------------ on the practice game itself


def walked_in(hopper: Hopper) -> int:
    """The step a walk right from the save loses its life at."""
    out = hopper.play([hopper.move("right")] * 300)
    assert out.lost
    return out.steps


def test_the_pause_and_the_fall_come_after_the_last_chance(hopper: Hopper) -> None:
    right = hopper.move("right")
    buttons = {b.name: b for b in hopper.adapter.buttons()}
    found = {}
    for frames in (toy_hopper.DYING_FRAMES, 200):
        hopper.game.dying_frames = frames
        died = walked_in(hopper)
        trace = [right] * died
        got = last_chance(
            hopper.connection, SPOT, trace, hopper.space, frames_per_step=FRAMES, lives=[LIVES]
        )
        assert got is not None
        found[frames] = (got.live, died)
        # After that many steps it has just walked off the edge: on the ground one step
        # sooner, in the air and still alive now.
        hopper.adapter.load_state(SPOT)
        hopper.adapter.step(FRAMES * (got.live - 1), pressed=[buttons["right"]])
        assert hopper.game.state.on_ground and hopper.game.state.mode == toy_hopper.PLAYING
        hopper.adapter.step(FRAMES, pressed=[buttons["right"]])
        assert not hopper.game.state.on_ground and hopper.game.state.mode == toy_hopper.PLAYING
        # What is left is the fall, three steps, and the pause, every frame of it.
        fall = 3
        assert died - got.live == fall + frames // FRAMES

        # The game itself agrees, step by step: from there on no move changes a thing,
        # and at every step before it some move does.
        def settled(step: int, died: int = died, trace: list[int] = trace) -> bool:
            return all(
                hopper.play(trace[:step] + [move] * (died - step))
                == Outcome(steps=died, lost=True, ground=0)
                for move in range(len(hopper.space))
                if move != right
            )

        assert [settled(step) for step in range(died)] == (
            [False] * got.live + [True] * (died - got.live)
        )
        assert got.plays < 5 * len(hopper.space)  # 48 and 62 of them: halving, not a sweep
    # The same number of steps before the fall, whatever the pause; the counter is later.
    (short, short_died), (long, long_died) = found[toy_hopper.DYING_FRAMES], found[200]
    assert short == long == 21
    assert long_died - short_died == (200 - toy_hopper.DYING_FRAMES) // FRAMES == 38
    assert hopper.game.state.lives == toy_hopper.START_LIVES - 1  # where a play ended: its own
