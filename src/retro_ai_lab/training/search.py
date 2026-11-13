"""Trying the small changes to what it did until one gets past (T-1006).

When the AI loses a life, the go has a save from a little earlier and the list
of moves made since (:mod:`retro_ai_lab.training.moments`). :func:`find_way`
plays that list again from the save with one small change at a time
(:mod:`retro_ai_lab.training.ways`), off the record
(:mod:`retro_ai_lab.training.trial`), until one stays alive and reaches ground
the go had not been on.

A try is played blind first: no screenshot, one read of the life values a step.
Most tries lose the life again, and that is all there is to know about them.
One that stays alive is played once more with a screen watch, because alive is
not enough (standing still stays alive): it is the way only if the watch counts
new ground. Every play, blind or looked at, counts against the budget, so a
search always ends.

The prototype found a way in 9 to 142 tries where there was one, all of them a
move one button from the one made, held 4 to 16 steps.

What it did is not always worth changing (T-1012). A learner that wandered
about before the life went (back a little, a jump too early, a button that
does nothing) leaves a list no single small change gets across with: a third
of the searches on the practice game found nothing, nearly all of them that
way. So a search that is given a plain move, the one that has paid best,
tries the plain way first: that move held from the save, and where that loses
the life, the same small changes to it. Measured on 179 searches it finds a
way in 86% against 70% and sooner (median try 19 and 43 against 42 and 54);
the two together find one in 97%. The plain way has half the tries at most,
and what it did has the rest: it is what gets past where an earlier jump has
to stay in, and where the plain way only walks into a wall.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.services.connection import ConnectionService
from retro_ai_lab.training.trial import play
from retro_ai_lab.training.vision import DEFAULT_LOOK_EVERY, ScreenWatch
from retro_ai_lab.training.ways import Change, changes
from retro_ai_lab.understanding.knowledge import Variable

BUDGET = 120  # plays a search may make
CARRY_ON = 20  # steps a try runs on past where the life went
PLAIN_SHARE = 0.5  # of the budget: the most the plain way may use (T-1012)


@dataclass(frozen=True, slots=True)
class Way:
    """A change that got past."""

    moves: tuple[int, ...]  # what to play from the save: the changed trace and on
    change: Change
    tries: int  # plays the search had made when it was found, its own two included
    ground: int  # steps of new ground the watch counted along it
    # The plain way (T-1012): the move held all the way, when the change is to that and
    # not to what it did, and the step holding it lost the life at. With no life lost
    # there is nothing to change: the change is then that move, held from the start.
    plain: int | None = None
    lost_at: int | None = None


@dataclass(frozen=True, slots=True)
class Search:
    """What a search came to."""

    way: Way | None
    tries: int  # plays made, blind and looked at
    # How long it took (T-1008). Not part of what was found: two searches that
    # find the same in the same tries are the same search.
    seconds: float = field(default=0.0, compare=False)
    out_of_time: bool = False  # it stopped because its time was up, not its tries


def _whole(value: object, what: str, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ConfigurationError(f"{what} must be a whole number, {least} or more.")
    return value


def find_way(
    connection: ConnectionService,
    slot: str,
    trace: Sequence[int],
    space: ActionSpace,
    *,
    frames_per_step: int,
    lives: Sequence[Variable],
    watch: Callable[[], ScreenWatch],
    budget: int = BUDGET,
    carry_on: int = CARRY_ON,
    look_every: int = DEFAULT_LOOK_EVERY,
    seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    plain: int | None = None,
) -> Search:
    """Look for a small change to ``trace``, played from the save in ``slot``, that
    stays alive and reaches new ground.

    ``trace`` is what the AI did from the save to the step the life went.
    ``lives`` are the values whose going down is a life lost; ``watch`` gives a
    fresh screen watch for one try each time it is called (the go's
    :meth:`ScreenWatch.trial`). The game is left wherever the last try ended:
    putting it back is the caller's, and so is an emulator error.

    On a real emulator a try costs real time. With ``seconds``, no play is started
    once that long has passed on ``clock`` since the search began (T-1008); a play
    in flight is finished.

    With ``plain``, a move, the plain way is tried first (T-1012): that move held
    for as long as the trace and the carry-on. Alive and on new ground, it is the
    way. Where it loses a life, the small changes are tried on the plain moves up
    to that step, with at most ``PLAIN_SHARE`` of the budget; after that, and at
    once where the plain way stays alive and gets nowhere new, the changes to
    ``trace``. Moves played once are not played again and cost no try.
    """
    if not slot or not isinstance(slot, str):
        raise ConfigurationError("A search needs the name of the save to start from.")
    if not trace:
        raise ConfigurationError("There is nothing it did to change.")
    if not lives:
        raise ConfigurationError(
            "A search needs a life or health value: without one nothing is seen to be lost."
        )
    if not callable(watch):
        raise ConfigurationError("watch must give a screen watch for a try when called.")
    _whole(budget, "The budget of tries", 1)
    _whole(carry_on, "carry_on", 0)
    _whole(frames_per_step, "frames_per_step", 1)
    _whole(look_every, "look_every", 1)
    if seconds is not None and (
        isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds > 0
    ):
        raise ConfigurationError("The time a search may take must be a number above 0.")
    if not callable(clock):
        raise ConfigurationError("clock must give the time when called.")
    for move in trace:
        space[move]  # refuses a move the controls don't have, before anything is played
    if plain is not None:
        space[plain]
    began = clock()
    tries, late = 0, False

    def taken() -> float:
        return max(0.0, clock() - began)

    def may_play() -> bool:
        """Whether another play may start; says so when it is the time that is up."""
        nonlocal late
        if tries >= budget:
            return False
        late = seconds is not None and taken() >= seconds
        return not late

    played: set[tuple[int, ...]] = set()

    def seek(base: Sequence[int], limit: int) -> tuple[Change, tuple[int, ...], int] | None:
        """The first small change to ``base`` that gets past, with its moves and the
        new ground counted, while the tries stay under ``limit``."""
        nonlocal tries
        for change in changes(base, space):
            moves = change.apply(base, carry_on)
            if moves in played:
                continue  # the other part of the search has been here
            if tries >= limit or not may_play():
                break
            tries += 1
            blind = play(
                connection, slot, moves, space, frames_per_step=frames_per_step, lives=lives
            )
            if blind.lost:
                played.add(moves)
                continue
            if tries >= limit or not may_play():
                break  # alive and not looked at yet: not settled, so not marked as played
            played.add(moves)
            tries += 1
            seen = play(
                connection,
                slot,
                moves,
                space,
                frames_per_step=frames_per_step,
                lives=lives,
                watch=watch(),
                look_every=look_every,
            )
            if seen.past:
                return change, moves, seen.ground
        return None

    share = int(budget * PLAIN_SHARE)
    if plain is not None and share >= 1 and may_play():
        held = (plain,) * (len(trace) + carry_on)
        tries += 1
        first = play(connection, slot, held, space, frames_per_step=frames_per_step, lives=lives)
        if first.lost:
            base = (plain,) * first.steps
            got = seek(base, share)
            if got is not None:
                change, moves, ground = got
                way = Way(moves, change, tries, ground, plain=plain, lost_at=first.steps)
                return Search(way, tries, taken())
        elif tries < share and may_play():
            tries += 1
            seen = play(
                connection,
                slot,
                held,
                space,
                frames_per_step=frames_per_step,
                lives=lives,
                watch=watch(),
                look_every=look_every,
            )
            if seen.past:
                way = Way(held, Change(0, plain, len(trace)), tries, seen.ground, plain=plain)
                return Search(way, tries, taken())
    got = None if late else seek(trace, budget)
    if got is not None:
        change, moves, ground = got
        return Search(Way(moves, change, tries, ground), tries, taken())
    return Search(None, tries, taken(), late)


__all__ = ["BUDGET", "CARRY_ON", "PLAIN_SHARE", "Search", "Way", "find_way"]
