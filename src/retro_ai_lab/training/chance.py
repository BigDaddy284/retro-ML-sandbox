"""From when on a life lost could no longer be saved (T-1031).

A game does not take the life the moment the mistake is made. The player falls
for a while, or the game plays the dying out, and only then does the counter
drop: on the practice game 12 to 18 steps later, on Super Mario Bros. about 50.
Nothing pressed in that while changes anything. A search for a way past that
goes back a fixed number of steps from the counter, and tries its changes from
the last step backwards, spends its tries there, and where the while is longer
than the steps it goes back, all of them.

:func:`last_chance` finds where that while begins, by asking the game. It plays
what the AI did again from the save, and then the same with another move held
from some step to the end. Where every other move, held from a step on, loses
the life on the very same step as before, nothing from that step on mattered:
the loss was settled. It halves its way to the first such step. No game, no
button and no number of frames is named here, and a play is a try like any
other (:mod:`retro_ai_lab.training.trial`): off the record.

The halving takes it that once a loss is settled it stays settled. Every other
move is asked before a step is called too late, the ones a button away from the
move made first: asking only a few put the step too early in the try-outs, and
the steps just before it are where most ways past begin.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.services.connection import ConnectionService
from retro_ai_lab.training.trial import play
from retro_ai_lab.understanding.knowledge import Variable


@dataclass(frozen=True, slots=True)
class Chance:
    """Where the loss was settled."""

    # Steps of the trace before the first one from which nothing changes the loss:
    # the last chance is the step before it. 0 where even the first step is too
    # late; the whole length where the last step could still have saved the life.
    live: int
    plays: int  # plays it took to find out
    # False where the time ran out first (T-1032): the loss was settled from that step
    # on for certain, and maybe from an earlier one.
    sure: bool = True


class _TimeUpError(Exception):
    """The time it may take has passed."""


def last_chance(
    connection: ConnectionService,
    slot: str,
    trace: Sequence[int],
    space: ActionSpace,
    *,
    frames_per_step: int,
    lives: Sequence[Variable],
    seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    shut: int | None = None,
    partial: bool = False,
) -> Chance | None:
    """How many steps of ``trace``, played from the save in ``slot``, come before the
    loss of the life at its end was settled.

    ``trace`` is what the AI did from the save to the step the life went, and
    ``lives`` the values whose going down is a life lost. None where the trace
    played again does not lose the life on its last step (then nothing can be told
    by playing it). The game is left wherever the last play ended: putting it back
    is the caller's, and so is an emulator error.

    With ``seconds``, no play is started once that long has passed on ``clock``; a
    play in flight is finished, and the answer is None. With ``partial`` as well
    (T-1032), the answer where the time runs out is the latest step known by then
    from which on the loss was settled, with ``sure`` False: the step asked for is
    that one or an earlier one, and changes tried before it are tried at some steps
    too many, which costs tries and nothing else. Still None where nothing is known:
    before the trace was played again, or before any step was found too late.

    ``shut`` is what the caller knows already: the loss was settled from that step
    of the trace on at the latest (a later save was too late from its first step,
    and lies that many steps on). The halving then starts from there, not from the
    end, and no step from there on is asked.
    """
    if not slot or not isinstance(slot, str):
        raise ConfigurationError("The last chance needs the name of the save to start from.")
    if not trace:
        raise ConfigurationError("There is nothing it did to play again.")
    if not lives:
        raise ConfigurationError(
            "The last chance needs a life or health value: without one nothing is seen to be lost."
        )
    if isinstance(frames_per_step, bool) or not isinstance(frames_per_step, int):
        raise ConfigurationError("frames_per_step must be a whole number, 1 or more.")
    if frames_per_step < 1:
        raise ConfigurationError("frames_per_step must be a whole number, 1 or more.")
    if seconds is not None and (
        isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds > 0
    ):
        raise ConfigurationError("The time finding the last chance may take must be above 0.")
    if not callable(clock):
        raise ConfigurationError("clock must give the time when called.")
    for move in trace:
        space[move]  # refuses a move the controls don't have, before anything is played
    length = len(trace)
    if not isinstance(partial, bool):
        raise ConfigurationError("partial must be true or false.")
    if shut is not None and (
        isinstance(shut, bool) or not isinstance(shut, int) or not 1 <= shut <= length
    ):
        raise ConfigurationError(
            "shut must be a whole number from 1 to the number of steps it did."
        )
    began = clock()
    plays = 0

    def same(moves: Sequence[int]) -> bool:
        """Whether these moves lose the life on the step the trace lost it."""
        nonlocal plays
        if seconds is not None and clock() - began >= seconds:
            raise _TimeUpError
        plays += 1
        out = play(connection, slot, moves, space, frames_per_step=frames_per_step, lives=lives)
        return out.lost and out.steps == length

    def settled(step: int) -> bool:
        """Whether every other move, held from ``step`` to the end, changes nothing."""
        made = trace[step]
        near = space.near(made)
        others = (*near, *(m for m in range(len(space)) if m != made and m not in near))
        return all(same((*trace[:step], *([other] * (length - step)))) for other in others)

    # Settled from here on, as far as is known: the end says nothing (past the last
    # step there is nothing to change).
    shut_at = length if shut is None else shut
    again = False  # what it did was played again, and lost the life on its last step
    try:
        if not same(trace):
            return None
        again = True
        if settled(0):
            return Chance(0, plays)
        open_at = 0  # not settled at this step
        while shut_at - open_at > 1:
            middle = (open_at + shut_at) // 2
            if settled(middle):
                shut_at = middle
            else:
                open_at = middle
        return Chance(shut_at, plays)
    except _TimeUpError:
        if partial and again and shut_at < length:
            return Chance(shut_at, plays, sure=False)
        return None


__all__ = ["Chance", "last_chance"]
