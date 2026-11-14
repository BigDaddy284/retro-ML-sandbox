"""T-1022: the practice games starting over, against their own account of it.

Each game is played by a script through many game overs. The detector is given a
hand-made understanding (the life count, the score, a number of progress and the
"game over" switch or mode: what a good run of the AI's own guessing holds) and
begins to watch
in the middle of a game, after the first life has gone, so that no watch begins
where the game starts. The game says when it really started over; the detector's
restarts are held against that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from retro_ai_lab.emulator.fake import ToyRacer, toy_hopper, toy_racer
from retro_ai_lab.emulator.fake.toy_hopper import ToyHopper
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.understanding import (
    Encoding,
    EventDetector,
    EventKind,
    Evidence,
    Knowledge,
    VariableKind,
)

RAM = "ram"
SURE = [Evidence("test", "", 3.0)]  # Medium
Held = dict[int, tuple[VariableKind, Encoding]]


@dataclass
class Account:
    """What happened, by the game, and what the detector made of it."""

    real: list[bool] = field(default_factory=list)  # each real restart: was it called?
    lives_lost: int = 0
    called_at_a_life_lost: int = 0
    called_elsewhere: int = 0


def play(
    game: Any,
    held: Held,
    buttons: Callable[[int], tuple[str, ...]],
    lives_at: int,
    frames: int,
) -> Account:
    knowledge = Knowledge("practice")
    for offset, (kind, encoding) in held.items():
        knowledge.propose(RAM, offset, kind=kind, encoding=encoding, name=kind.label, evidence=SURE)
    detector = EventDetector(knowledge)
    restarted = [0]
    really_restart = game._restart_after_game_over

    def restart() -> None:
        restarted[0] += 1
        really_restart()

    game._restart_after_game_over = restart
    account = Account()
    watching = False
    for frame in range(frames):
        restarts_before, lives_before = restarted[0], game.ram[lives_at]
        game.step(buttons(frame))
        lost = game.ram[lives_at] < lives_before
        # The detector's first look is the one after the first life has gone.
        if not watching and not lost:
            continue
        events = detector.observe(MemorySnapshot(frame, {RAM: bytes(game.ram)}))
        called = any(e.kind is EventKind.RESET for e in events)
        if not watching:
            watching = True
            assert events == ()
        elif restarted[0] != restarts_before:
            account.real.append(called)
        elif lost:
            account.lives_lost += 1
            account.called_at_a_life_lost += called
        else:
            account.called_elsewhere += called
    return account


def test_the_racing_game_starting_over_is_called() -> None:
    at = toy_racer.LAYOUT
    held: Held = {
        at.checkpoint: (VariableKind.PROGRESS, Encoding.U8),
        at.lives: (VariableKind.LIVES, Encoding.U8),
        at.score: (VariableKind.SCORE, Encoding.U16LE),
        at.game_over: (VariableKind.FLAG, Encoding.U8),
    }
    for seed in (0, 1, 2, 3):
        game = ToyRacer(seed=seed)

        def buttons(frame: int, game: ToyRacer = game) -> tuple[str, ...]:
            return ("start",) if game.ram[at.game_over] else ("up",)

        account = play(game, held, buttons, at.lives, 4000)
        assert len(account.real) >= 20 and account.lives_lost >= 60, seed
        # The first time the game starts over the score and the checkpoint go where
        # the AI has never seen them, and it may go uncalled. Every one after it is
        # called (not one was before this task: no watch began at a start).
        assert all(account.real[1:]), (seed, account.real)
        assert account.called_at_a_life_lost == 0 and account.called_elsewhere == 0, seed


def test_the_jumping_game_starting_over_is_called_and_a_life_lost_is_not() -> None:
    at = toy_hopper.LAYOUT
    held: Held = {
        at.x_page: (VariableKind.PROGRESS, Encoding.U8),
        at.lives: (VariableKind.LIVES, Encoding.U8),
        at.score: (VariableKind.SCORE, Encoding.U16LE),
        at.mode: (VariableKind.FLAG, Encoding.U8),
    }
    # A level on which the script gets off the first screenful before it dies, so
    # that the place in the level is one of the things the game starting over moves.
    game = ToyHopper(seed=13)

    def buttons(frame: int) -> tuple[str, ...]:
        if game.ram[at.mode] == toy_hopper.GAME_OVER:
            return ("start",)
        return ("right", "a") if (frame // 20) % 3 == 0 else ("right",)

    account = play(game, held, buttons, at.lives, 12000)
    assert len(account.real) >= 8 and account.lives_lost >= 25
    assert all(account.real), account.real
    # A life lost puts the player back at the level's start: the place in the level
    # and the mode are where a restart puts them. Before this task 19 of the 28
    # lives lost here were called a restart.
    assert account.called_at_a_life_lost == 0 and account.called_elsewhere == 0
