"""A try from a saved moment, off the record (T-1001).

To find a way past a place it died the AI plays small changes of what it did
(:mod:`retro_ai_lab.training.ways`) from a save taken a moment before. A try is
only a try: :func:`play` talks to the emulator and to nothing else, so nothing
is learned from it, no reward is counted and the statistics of the game's
memory do not see a death that never counted.

Whether a try got past is told from two things the app already has: the values
it reads as lives or health (one going down ends the try), and the new ground
the watch of the screen counts (:meth:`ScreenWatch.trial`, T-1000). Alive is
not enough, since standing still stays alive; alive and somewhere the go had
not been is.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass

from retro_ai_lab.core import Button, ConfigurationError, EmulatorError, StepResult
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.services.connection import ConnectionService
from retro_ai_lab.training.vision import DEFAULT_LOOK_EVERY, ScreenWatch
from retro_ai_lab.understanding.knowledge import Variable, decode

PAST = 2  # steps of new ground, of the player's own size, that count as getting past


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a try came to."""

    steps: int  # steps played: all of them, or up to the one a value went down in
    lost: bool  # a life or health value went down
    ground: int  # steps of new ground the watch counted (0 without a watch)

    @property
    def past(self) -> bool:
        """Alive, and somewhere the go had not been."""
        return not self.lost and self.ground >= PAST


def _at_least_one(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{what} must be a whole number, 1 or more.")
    return value


def play(
    connection: ConnectionService,
    slot: str,
    moves: Sequence[int],
    space: ActionSpace,
    *,
    frames_per_step: int,
    lives: Sequence[Variable],
    watch: ScreenWatch | None = None,
    look_every: int = DEFAULT_LOOK_EVERY,
) -> Outcome:
    """Load the save in ``slot`` and play ``moves``, one step of ``frames_per_step``
    frames each, until they are done or one of the ``lives`` values goes down.

    ``watch`` (a :meth:`ScreenWatch.trial`) is shown a screenshot every
    ``look_every`` steps, on the steps a go would look, and its count of new ground
    is the outcome's. The pad is let go at the end. The game is left where the try
    ended: putting it back is the caller's. An emulator error comes out as it is.
    """
    if not slot or not isinstance(slot, str):
        raise ConfigurationError("A try needs the name of the save to start from.")
    if not moves:
        raise ConfigurationError("A try needs at least one move.")
    _at_least_one(frames_per_step, "frames_per_step")
    _at_least_one(look_every, "look_every")
    if watch is not None and not isinstance(watch, ScreenWatch):
        raise ConfigurationError("watch must be a ScreenWatch.")
    if any(not isinstance(v, Variable) for v in lives):
        raise ConfigurationError("lives must be variables the AI reads.")
    known = {b.name: b for b in connection.call(lambda a: a.buttons())}
    held = []
    for move in moves:
        names = space.buttons_for(move)  # refuses a move these controls don't have
        unknown = sorted(names - known.keys())
        if unknown:
            raise ConfigurationError(f"Unknown buttons for this game: {unknown}.")
        held.append(tuple(known[n] for n in sorted(names)))
    reads = tuple((v.region, v.offset, v.width) for v in lives)

    def values(raw: Sequence[bytes]) -> list[int]:
        return [decode(data, v.encoding) for data, v in zip(raw, lives, strict=True)]

    def step(pressed: tuple[Button, ...], look: bool) -> StepResult:
        return connection.call(
            lambda a: a.step(frames_per_step, pressed=pressed, reads=reads, screenshot=look)
        )

    steps, lost = 0, False
    try:
        connection.call(lambda a: a.load_state(slot))
        was = values(connection.call(lambda a: a.read_many(reads))) if reads else []
        for pressed in held:
            result = step(pressed, watch is not None and steps % look_every == 0)
            steps += 1
            if watch is not None and result.frame is not None:
                watch.look(result.frame)
            now = values(result.reads) if reads else []
            if any(n < w for n, w in zip(now, was, strict=True)):
                lost = True
                break
            was = now
    finally:
        with contextlib.suppress(EmulatorError):
            connection.call(lambda a: a.set_input(()))
    return Outcome(steps, lost, watch.take_new_ground() if watch is not None else 0)


__all__ = ["PAST", "Outcome", "play"]
