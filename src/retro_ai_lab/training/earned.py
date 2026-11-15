"""Whether a value went up whatever was pressed (T-1034).

Some of what the AI reads as progress is no doing of the player's. The high byte
of a game's clock goes up every so many frames, and looks like a counter of
rounds; paid as progress, it pays best for standing still and staying alive. Its
own statistics do not tell it from a checkpoint. What it rises on does.

:func:`rose_anyway` asks the game. It plays what the AI did up to a rise again
from a save taken before it, and then, from the same save and for as many steps,
every other move of the controls held instead. Where the value ends as high with
every one of them, nothing pressed in those steps had a part in the rise. Where
one of them leaves it lower, the rise was earned, and that is known at once. No
game, no value and no button is named here, and a play is a try like any other
(:mod:`retro_ai_lab.training.trial`): off the record.

Another move can also get the player killed, or a level restarted, and a clock
with it: then a rise that was no doing of the player's looks earned. So one
answer is one piece of evidence, and what follows from several is the caller's.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from retro_ai_lab.core import Button, ConfigurationError, EmulatorError, StepResult
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.services.connection import ConnectionService
from retro_ai_lab.understanding.knowledge import Variable, decode


@dataclass(frozen=True, slots=True)
class Rise:
    """What the game said of a rise."""

    anyway: bool  # the value ended as high whatever was held: nothing pressed earned it
    plays: int  # plays it took to find out


class _TimeUpError(Exception):
    """The time it may take has passed."""


def ends(
    connection: ConnectionService,
    slot: str,
    moves: Sequence[int],
    space: ActionSpace,
    *,
    frames_per_step: int,
    value: Variable,
) -> tuple[int, int]:
    """Load the save in ``slot`` and play ``moves``, one step of ``frames_per_step``
    frames each: what ``value`` read at the save, and after the last move.

    The pad is let go at the end. The game is left where the moves ended: putting it
    back is the caller's, and so is an emulator error.
    """
    known = {b.name: b for b in connection.call(lambda a: a.buttons())}
    held = []
    for move in moves:
        names = space.buttons_for(move)  # refuses a move these controls don't have
        unknown = sorted(names - known.keys())
        if unknown:
            raise ConfigurationError(f"Unknown buttons for this game: {unknown}.")
        held.append(tuple(known[n] for n in sorted(names)))
    reads = ((value.region, value.offset, value.width),)

    def step(pressed: tuple[Button, ...]) -> StepResult:
        return connection.call(lambda a: a.step(frames_per_step, pressed=pressed, reads=reads))

    try:
        connection.call(lambda a: a.load_state(slot))
        began = decode(connection.call(lambda a: a.read_many(reads))[0], value.encoding)
        ended = began
        for pressed in held:
            ended = decode(step(pressed).reads[0], value.encoding)
    finally:
        with contextlib.suppress(EmulatorError):
            connection.call(lambda a: a.set_input(()))
    return began, ended


def rose_anyway(
    connection: ConnectionService,
    slot: str,
    trace: Sequence[int],
    space: ActionSpace,
    *,
    frames_per_step: int,
    value: Variable,
    seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Rise | None:
    """Whether ``value`` went up over ``trace``, played from the save in ``slot``,
    whatever was pressed.

    ``trace`` is what the AI did from the save to the step the value rose at. None
    where, played again, it leaves the value no higher than the save has it (then
    there is no rise to ask about). Otherwise every other move of the controls is
    held from the save for as many steps, those a button away from the first move
    made first: ``Rise(False, plays)`` at the first that leaves the value lower
    than the trace left it, ``Rise(True, plays)`` where none does.

    With ``seconds``, no play is started once that long has passed on ``clock``; a
    play in flight is finished, and the answer is None. The game is left wherever
    the last play ended.
    """
    if not slot or not isinstance(slot, str):
        raise ConfigurationError("Asking about a rise needs the name of the save to start from.")
    if not trace:
        raise ConfigurationError("There is nothing it did to play again.")
    if not isinstance(value, Variable):
        raise ConfigurationError("value must be a value of the game's memory the AI reads.")
    if isinstance(frames_per_step, bool) or not isinstance(frames_per_step, int):
        raise ConfigurationError("frames_per_step must be a whole number, 1 or more.")
    if frames_per_step < 1:
        raise ConfigurationError("frames_per_step must be a whole number, 1 or more.")
    if seconds is not None and (
        isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds > 0
    ):
        raise ConfigurationError("The time asking about a rise may take must be above 0.")
    if not callable(clock):
        raise ConfigurationError("clock must give the time when called.")
    for move in trace:
        space[move]  # refuses a move the controls don't have, before anything is played
    length = len(trace)
    began = clock()
    plays = 0

    def end_of(moves: Sequence[int]) -> tuple[int, int]:
        nonlocal plays
        if seconds is not None and clock() - began >= seconds:
            raise _TimeUpError
        plays += 1
        return ends(connection, slot, moves, space, frames_per_step=frames_per_step, value=value)

    try:
        before, top = end_of(trace)
        if top <= before:
            return None
        made = trace[0]
        near = space.near(made)
        others = (*near, *(m for m in range(len(space)) if m != made and m not in near))
        for other in others:
            if end_of([other] * length)[1] < top:
                return Rise(False, plays)
        return Rise(True, plays)
    except _TimeUpError:
        return None


__all__ = ["Rise", "ends", "rose_anyway"]
