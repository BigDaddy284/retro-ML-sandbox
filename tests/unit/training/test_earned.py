"""T-1034: asking the game whether a value went up whatever was pressed."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.training import earned
from retro_ai_lab.training.earned import Rise, rose_anyway
from retro_ai_lab.understanding.knowledge import Variable, VariableKind
from tests.unit.training import test_trial as trial_tests
from tests.unit.training.test_chance import BACK, JUMP, PAD, WALK
from tests.unit.training.test_trial import FRAMES, SPOT, Hopper

hopper = trial_tests.hopper  # the practice game behind a real connection (pytest fixture)

PAGE = Variable("page", "ram", LAYOUT.x_page, kind=VariableKind.PROGRESS)
CLOCK = Variable("clock", "ram", LAYOUT.clock + 1, kind=VariableKind.PROGRESS)
LENGTH = 12
TRACE = [WALK] * LENGTH
OTHERS = len(PAD) - 1
WAS = 3  # what the value reads at the save


class Game:
    """Stands in for ``earned.ends`` with a game of one rule: what the value reads
    after the moves."""

    def __init__(self, after: Callable[[Sequence[int]], int]) -> None:
        self.after = after
        self.calls: list[tuple[int, ...]] = []

    def __call__(
        self, connection: object, slot: str, moves: Sequence[int], space: object, **kw: object
    ) -> tuple[int, int]:
        assert slot == SPOT and kw == {"frames_per_step": FRAMES, "value": PAGE}
        self.calls.append(tuple(moves))
        return WAS, self.after(moves)


def ask(trace: Sequence[int] = TRACE, **kw: object) -> Rise | None:
    kw.setdefault("value", PAGE)
    kw.setdefault("frames_per_step", FRAMES)
    return rose_anyway(object(), SPOT, trace, PAD, **kw)  # type: ignore[arg-type]


def asking(
    monkeypatch: pytest.MonkeyPatch, after: Callable[[Sequence[int]], int], **kw: object
) -> tuple[Rise | None, Game]:
    game = Game(after)
    monkeypatch.setattr(earned, "ends", game)
    return ask(**kw), game


def test_a_rise_whatever_is_pressed(monkeypatch: pytest.MonkeyPatch) -> None:
    near = PAD.near(WALK)
    order = [*near, *(m for m in range(len(PAD)) if m != WALK and m not in near)]
    assert len(order) == OTHERS == 14 and JUMP in near and BACK not in near
    # A clock: one more whatever is held.
    got, game = asking(monkeypatch, lambda moves: WAS + 1)
    assert got == Rise(anyway=True, plays=1 + OTHERS)
    # What it did, as it did it; then every other move held for as many steps, the
    # ones a button away from the first move made first, each once.
    assert game.calls[0] == tuple(TRACE)
    assert game.calls[1:] == [(move,) * LENGTH for move in order]
    # A move that leaves it higher still has not kept it from rising.
    got, game = asking(monkeypatch, lambda moves: WAS + (5 if BACK in moves else 1))
    assert got == Rise(True, 1 + OTHERS)
    # The other moves are those other than the first move made, whatever came after it.
    mixed = [JUMP, *[WALK] * (LENGTH - 1)]
    game = Game(lambda moves: WAS + 1)
    monkeypatch.setattr(earned, "ends", game)
    assert ask(mixed) == Rise(True, 1 + OTHERS)
    assert game.calls[0] == tuple(mixed)
    held = [moves[0] for moves in game.calls[1:]]
    assert JUMP not in held and WALK in held and sorted(held) == sorted(set(held))
    assert held[: len(PAD.near(JUMP))] == list(PAD.near(JUMP))
    # One step is enough to ask about.
    game = Game(lambda moves: WAS + 1)
    monkeypatch.setattr(earned, "ends", game)
    assert ask([WALK]) == Rise(True, 1 + OTHERS) and game.calls[1] == (order[0],)


def test_a_rise_one_move_prevents(monkeypatch: pytest.MonkeyPatch) -> None:
    near = PAD.near(WALK)
    order = [*near, *(m for m in range(len(PAD)) if m != WALK and m not in near)]
    # A page crossed: only going on to the right gets there.
    right = {WALK, *near}
    got, game = asking(monkeypatch, lambda moves: WAS + (1 if set(moves) <= right else 0))
    # Known at the first move that leaves it lower: nothing is asked after that.
    first = next(i for i, move in enumerate(order) if move not in right)
    assert got == Rise(anyway=False, plays=2 + first)
    assert game.calls[1:] == [(move,) * LENGTH for move in order[: first + 1]]
    # The very first other move asked can be the one.
    got, game = asking(monkeypatch, lambda moves: WAS + (0 if order[0] in moves else 1))
    assert got == Rise(False, 2) and len(game.calls) == 2
    # And the very last: every move was asked, and one says no.
    got, game = asking(monkeypatch, lambda moves: WAS + (0 if order[-1] in moves else 1))
    assert got == Rise(False, 1 + OTHERS)
    # Lower than the trace left it is lower, though it is higher than at the save.
    got, game = asking(monkeypatch, lambda moves: WAS + (1 if BACK in moves else 2))
    assert got is not None and not got.anyway
    # As high is not lower.
    got, game = asking(monkeypatch, lambda moves: WAS + 2)
    assert got == Rise(True, 1 + OTHERS)


def test_nothing_rose_when_played_again(monkeypatch: pytest.MonkeyPatch) -> None:
    for after in (WAS, WAS - 1, 0):
        got, game = asking(monkeypatch, lambda moves, after=after: after)
        assert got is None and game.calls == [tuple(TRACE)]


class Clock:
    """A clock the test moves: a second passes with every play."""

    def __init__(self, game: Game) -> None:
        self.now = 100.0
        self.game = game

    def play(self, *args: object, **kw: object) -> tuple[int, int]:
        self.now += 1.0
        return self.game(*args, **kw)  # type: ignore[arg-type]

    def __call__(self) -> float:
        return self.now


def test_no_play_is_started_once_the_time_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed(after: Callable[[Sequence[int]], int], **kw: object) -> tuple[Rise | None, int]:
        game = Game(after)
        clock = Clock(game)
        monkeypatch.setattr(earned, "ends", clock.play)
        return ask(clock=clock, **kw), len(game.calls)

    whole = 1 + OTHERS
    assert timed(lambda moves: WAS + 1) == (Rise(True, whole), whole)
    for seconds in range(1, whole + 3):
        got, plays = timed(lambda moves: WAS + 1, seconds=seconds)
        # Out of time nothing is known: a move not asked yet may be the one that says no.
        assert plays == min(seconds, whole), seconds
        assert got == (Rise(True, whole) if seconds >= whole else None), seconds
    # A no found in time is an answer, however little time was left after it.
    assert timed(lambda moves: WAS + (0 if BACK in moves else 1), seconds=1000.0)[0] is not None
    order = [*PAD.near(WALK), *(m for m in range(len(PAD)) if m not in (WALK, *PAD.near(WALK)))]
    at = 2 + order.index(BACK)
    assert timed(lambda moves: WAS + (0 if BACK in moves else 1), seconds=at) == (
        Rise(False, at),
        at,
    )
    assert timed(lambda moves: WAS + (0 if BACK in moves else 1), seconds=at - 1) == (None, at - 1)
    # The time is up before anything is played: nothing is.
    game = Game(lambda moves: WAS + 1)
    monkeypatch.setattr(earned, "ends", game)
    reads = iter(range(100, 200))
    assert ask(clock=lambda: float(next(reads)), seconds=0.5) is None and game.calls == []


def test_misuse_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    game = Game(lambda moves: WAS + 1)
    monkeypatch.setattr(earned, "ends", game)
    for slot in ("", None, 3):
        with pytest.raises(ConfigurationError, match="needs the name of the save"):
            rose_anyway(object(), slot, TRACE, PAD, frames_per_step=FRAMES, value=PAGE)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="nothing it did to play again"):
        ask([])
    with pytest.raises(ConfigurationError):
        ask([WALK, len(PAD)])
    for value in ("page", None, LAYOUT.x_page):
        with pytest.raises(ConfigurationError, match="value must be a value of the game's memory"):
            ask(value=value)
    for frames in (0, -4, 2.0, True, "4"):
        with pytest.raises(ConfigurationError, match="frames_per_step must be a whole number"):
            ask(frames_per_step=frames)
    for seconds in (0, -1.0, True, "2"):
        with pytest.raises(ConfigurationError, match="may take must be above 0"):
            ask(seconds=seconds)
    with pytest.raises(ConfigurationError, match="clock must give the time"):
        ask(clock=12.0)
    assert game.calls == []  # nothing was played


def walk_on(hopper: Hopper, moves: Sequence[int]) -> None:
    buttons = {b.name: b for b in hopper.adapter.buttons()}
    for move in moves:
        pressed = [buttons[name] for name in sorted(hopper.space.buttons_for(move))]
        hopper.adapter.step(FRAMES, pressed=pressed)


@pytest.mark.parametrize("frames", [200, toy_hopper.DYING_FRAMES])
def test_a_page_crossed_is_earned_and_the_clock_is_not(
    hopper: Hopper, frames: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The practice game, with a long dying pause and with its own. Standing for 35
    # steps and then walking right with a jump over the gap, step 64 is where the
    # player crosses into the second page of the level, and also where the game's
    # clock, 256 frames after the level began, ticks its high byte over.
    game, space = hopper.game, hopper.space
    game.dying_frames = frames
    game.reset()
    still, walk, jump = hopper.move(), hopper.move("right"), hopper.move("a", "right")
    way = [still] * 35 + [walk] * 20 + [jump] * 4 + [walk] * 5
    walk_on(hopper, way[:50])
    hopper.adapter.save_state("before")
    trace = way[50:]
    walk_on(hopper, trace)
    ram = game.ram
    assert (ram[LAYOUT.x_page], ram[LAYOUT.clock + 1], game.state.lives) == (1, 1, 3)
    assert 256 <= game.state.x < 264
    others = len(space) - 1

    def ask_of(value: Variable) -> Rise | None:
        return rose_anyway(
            hopper.connection, "before", trace, space, frames_per_step=FRAMES, value=value
        )

    # The page was earned: known as soon as a move is held that does not get there.
    assert ask_of(PAGE) == Rise(anyway=False, plays=2)
    # Here the clock looks earned too. The first other move held, a jump kept up,
    # drops the player into the gap, and this game's clock stops while it dies. One
    # answer is one piece of evidence: what follows from several is the caller's.
    assert ask_of(CLOCK) == Rise(anyway=False, plays=2)
    # Where no move gets the player killed, the clock is seen for what it is: standing
    # at the start, it ticked over whatever was held for the fourteen steps before.
    game.reset()
    walk_on(hopper, [still] * 50)
    hopper.adapter.save_state("standing")
    walk_on(hopper, [still] * 14)
    assert (game.ram[LAYOUT.clock + 1], game.ram[LAYOUT.x_page]) == (1, 0)
    assert rose_anyway(
        hopper.connection, "standing", [still] * 14, space, frames_per_step=FRAMES, value=CLOCK
    ) == Rise(anyway=True, plays=1 + others)
    # And the same whatever it did itself in those steps.
    assert rose_anyway(
        hopper.connection, "standing", [walk] * 14, space, frames_per_step=FRAMES, value=CLOCK
    ) == Rise(anyway=True, plays=1 + others)
    # From a save after the rise there is no rise to ask about.
    hopper.adapter.load_state("before")
    walk_on(hopper, trace)
    hopper.adapter.save_state("after")
    for value in (CLOCK, PAGE):
        assert (
            rose_anyway(
                hopper.connection, "after", [walk] * 5, space, frames_per_step=FRAMES, value=value
            )
            is None
        )
    # The pad is let go at the end.
    assert hopper.adapter.held == frozenset()
    # A save that was never made is the emulator's to refuse.
    with pytest.raises((ConfigurationError, EmulatorError)):
        rose_anyway(hopper.connection, "nowhere", trace, space, frames_per_step=FRAMES, value=PAGE)
    # A move that needs a button this controller does not have.
    odd = ActionSpace([frozenset({"turbo"})])
    with pytest.raises(ConfigurationError):
        rose_anyway(hopper.connection, "before", [0], odd, frames_per_step=FRAMES, value=PAGE)
    # The emulator fails part-way: the error comes out and the pad is let go.
    real, calls = hopper.adapter.step, [0]

    def failing(*args: object, **kw: object) -> object:
        calls[0] += 1
        if calls[0] == 5:
            raise EmulatorError("The link dropped.")
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(hopper.adapter, "step", failing)
    with pytest.raises(EmulatorError):
        ask_of(PAGE)
    assert hopper.adapter.held == frozenset()


def test_nothing_of_it_is_on_the_record(hopper: Hopper) -> None:
    observation = hopper.observation
    assert observation.enabled and observation.observe() is not None
    observation.advance(4)
    frame, seen = observation.frame, len(hopper.events)
    assert frame > 0 and seen > 0  # it is listening, and events do reach the list
    still = hopper.move()
    hopper.game.dying_frames = 8  # a life lost in a play is gone before the play is over
    lives = hopper.game.state.lives
    hopper.adapter.save_state("here")
    # Seventy steps of standing: the clock ticks over. Up, down and left held change
    # nothing; right, the fourth other move, walks the player into the gap, a life
    # goes, and the level and its clock start again.
    got = rose_anyway(
        hopper.connection, "here", [still] * 70, hopper.space, frames_per_step=FRAMES, value=CLOCK
    )
    assert got == Rise(anyway=False, plays=5)
    assert hopper.game.state.lives < lives  # the game is left where that play ended
    # None of it reached anything that listens.
    assert observation.frame == frame and len(hopper.events) == seen
    # The game is the caller's to put back: it can.
    hopper.adapter.load_state("here")
    assert hopper.game.state.lives == lives
