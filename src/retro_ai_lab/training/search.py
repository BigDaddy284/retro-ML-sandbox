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
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.services.connection import ConnectionService
from retro_ai_lab.training.trial import play
from retro_ai_lab.training.vision import DEFAULT_LOOK_EVERY, ScreenWatch
from retro_ai_lab.training.ways import Change, changes
from retro_ai_lab.understanding.knowledge import Variable

BUDGET = 120  # plays a search may make
CARRY_ON = 20  # steps a try runs on past where the life went


@dataclass(frozen=True, slots=True)
class Way:
    """A change that got past."""

    moves: tuple[int, ...]  # what to play from the save: the changed trace and on
    change: Change
    tries: int  # plays the search had made when it was found, its own two included
    ground: int  # steps of new ground the watch counted along it


@dataclass(frozen=True, slots=True)
class Search:
    """What a search came to."""

    way: Way | None
    tries: int  # plays made, blind and looked at


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
) -> Search:
    """Look for a small change to ``trace``, played from the save in ``slot``, that
    stays alive and reaches new ground.

    ``trace`` is what the AI did from the save to the step the life went.
    ``lives`` are the values whose going down is a life lost; ``watch`` gives a
    fresh screen watch for one try each time it is called (the go's
    :meth:`ScreenWatch.trial`). The game is left wherever the last try ended:
    putting it back is the caller's, and so is an emulator error.
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
    tries = 0
    for change in changes(trace, space):  # refuses a move the controls don't have
        if tries >= budget:
            break
        moves = change.apply(trace, carry_on)
        tries += 1
        blind = play(connection, slot, moves, space, frames_per_step=frames_per_step, lives=lives)
        if blind.lost:
            continue
        if tries >= budget:
            break
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
            return Search(Way(moves, change, tries, seen.ground), tries)
    return Search(None, tries)


__all__ = ["BUDGET", "CARRY_ON", "Search", "Way", "find_way"]
