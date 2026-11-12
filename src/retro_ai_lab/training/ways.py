"""Small changes to what the AI did, to try past a place it died (T-999).

A learner that only stumbles past an obstacle now and then, and loses a life
every other time, is right to stay away from it. With an emulator that can go
back to an earlier moment there is a better way to find out: replay what it
did from a moment before the death with one small change, and see whether that
gets it past. This module is the list of changes and nothing else: no
emulator, no game and no button is named here.

A change is "at this step, make that move for this many steps". The move is
one button away from the move it made there (:meth:`ActionSpace.near`): on the
practice game every one of the 60 ways the prototype found was such a move,
held 4 to 16 steps, 15 to 33 steps before the life went.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace

HOLDS = (4, 8, 16)  # steps a changed move is held for, tried in this order
STRIDE = 2  # steps between one place a change is tried at and the next


def _whole(value: object, what: str, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ConfigurationError(f"{what} must be a whole number, {least} or more.")
    return value


def carry(trace: Sequence[int]) -> int:
    """The move it mostly made: what a try carries on with past the end of the trace.
    Of two as common, the one made later."""
    if not trace:
        raise ConfigurationError("There is nothing it did to carry on with.")
    counts = Counter(trace)
    top = max(counts.values())
    return next(move for move in reversed(trace) if counts[move] == top)


@dataclass(frozen=True, slots=True)
class Change:
    """From step ``at`` of what it did, make ``move`` for ``hold`` steps."""

    at: int
    move: int
    hold: int

    def apply(
        self, trace: Sequence[int], carry_on: int = 0, *, space: ActionSpace | None = None
    ) -> tuple[int, ...]:
        """The moves to play: the trace with the stretch replaced (cut at the trace's
        end), then ``carry_on`` steps more of the move it mostly made, so a try runs
        on past where the life went. ``space`` checks the move is one it has."""
        if not trace:
            raise ConfigurationError("There is nothing it did to change.")
        _whole(carry_on, "carry_on", 0)
        _whole(self.hold, "A change's hold", 1)
        if isinstance(self.at, bool) or not isinstance(self.at, int):
            raise ConfigurationError("A change's step must be a whole number.")
        if not 0 <= self.at < len(trace):
            raise ConfigurationError(f"No step {self.at} in what it did.")
        if space is not None:
            space[self.move]  # refuses a move these controls don't have
        moves = list(trace)
        end = min(len(moves), self.at + self.hold)
        moves[self.at : end] = [self.move] * (end - self.at)
        return (*moves, *([carry(trace)] * carry_on))


def changes(
    trace: Sequence[int],
    space: ActionSpace,
    *,
    holds: Sequence[int] = HOLDS,
    stride: int = STRIDE,
    most: int | None = None,
) -> Iterator[Change]:
    """Every change worth trying, most promising first.

    The shortest hold first; within a hold, from the end of the trace backwards
    every ``stride`` steps (the mistake is usually late); at each place, the moves
    near the one it made. A change that comes to the same moves as an earlier one
    (a long hold cut off by the end of the trace) is left out. ``most`` stops the
    list.
    """
    _whole(stride, "stride", 1)
    if not holds:
        raise ConfigurationError("There must be at least one hold to try.")
    for hold in holds:
        _whole(hold, "A hold", 1)
    if most is not None:
        _whole(most, "most", 1)
    near = [space.near(move) for move in trace]  # refuses a move that is not there
    played: set[tuple[int, ...]] = set()
    given = 0
    for hold in holds:
        for at in range(len(trace) - 1, -1, -stride):
            for move in near[at]:
                change = Change(at, move, hold)
                moves = change.apply(trace)
                if moves in played:
                    continue
                played.add(moves)
                yield change
                given += 1
                if most is not None and given >= most:
                    return


__all__ = ["HOLDS", "STRIDE", "Change", "carry", "changes"]
