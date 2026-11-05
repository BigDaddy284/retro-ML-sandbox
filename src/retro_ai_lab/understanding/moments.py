"""A person says what just happened; the AI finds the byte that shows it (T-901).

The app cannot see that Mario died: it reads the game's memory, and a life
counter changes so rarely that statistics alone pick the wrong byte. A cheat
finder solves this because the *person* says when it happened. "I just lost a
life" does the same, child-sized: each press keeps only the bytes that moved
the right way in the last few seconds (down by one for a life, up for score
or progress), and the presses are intersected. When one byte is left after
enough presses, the AI takes it as the person's word.

The AI presses the button for itself, too (T-904): a restart, when several
values jump back to their start together, usually follows a death, so the
:class:`RestartWatch` votes for the bytes that went down by one just before
each restart. A byte that did so before most restarts, and alone, is the
AI's own life-counter guess.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot
from retro_ai_lab.understanding.knowledge import VariableKind

WINDOW_FRAMES = 300  # the last five seconds at 60 frames a second
SETTLE_AFTER = 2  # presses that agree before one byte is taken as the person's word
WINDOW_WORDS = "the last 5 seconds"
RESTARTS_NEEDED = 3  # restarts a byte must go down by one before, to be the AI's own guess
RESTARTS_FORGET = 8  # restarts without a clear winner: the count starts again


class Moment(Enum):
    """Something a person can tell the AI just happened."""

    LOST_LIFE = "lost_life"
    GOT_FURTHER = "got_further"
    SCORED = "scored"

    @property
    def kind(self) -> VariableKind:
        return _KINDS[self]

    @property
    def name(self) -> str:
        """What the found value is called."""
        return _NAMES[self]

    @property
    def words(self) -> str:
        """What the person said, for sentences."""
        return _WORDS[self]

    @property
    def movement(self) -> str:
        """How the byte must have moved, for sentences."""
        return _MOVES[self]


_KINDS = {
    Moment.LOST_LIFE: VariableKind.LIVES,
    Moment.GOT_FURTHER: VariableKind.PROGRESS,
    Moment.SCORED: VariableKind.SCORE,
}
_NAMES = {Moment.LOST_LIFE: "Lives", Moment.GOT_FURTHER: "Progress", Moment.SCORED: "Score"}
_WORDS = {
    Moment.LOST_LIFE: "lost a life",
    Moment.GOT_FURTHER: "got further",
    Moment.SCORED: "scored",
}
_MOVES = {
    Moment.LOST_LIFE: "went down by one",
    Moment.GOT_FURTHER: "went up",
    Moment.SCORED: "went up",
}

Address = tuple[str, int]


@dataclass(frozen=True, slots=True)
class Narrowing:
    """What one press of a "tell the AI" button worked out."""

    moment: Moment
    presses: int  # presses that agree so far, this one included
    candidates: tuple[Address, ...]  # bytes still in the running, by region and offset
    settled: Address | None  # the one byte, once enough presses agree
    text: str  # plain words for the panel and the log

    @property
    def count(self) -> int:
        return len(self.candidates)


def _moved(moment: Moment, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    delta = after.astype(np.int16) - before.astype(np.int16)
    moved: np.ndarray = delta == -1 if moment is Moment.LOST_LIFE else delta > 0
    return moved


def _window(snapshots: Iterable[MemorySnapshot], frames: int) -> tuple[MemorySnapshot, ...]:
    """The oldest and the newest snapshot of the last ``frames`` frames."""
    items = list(snapshots)
    if len(items) < 2:
        raise ConfigurationError(
            "Play for a few seconds first, then press this right after it happens."
        )
    latest = items[-1]
    start = latest
    for s in reversed(items[:-1]):
        if s.frame > latest.frame or s.frame < latest.frame - frames:
            break  # the counter went backwards (a restore) or the window is full
        start = s
    if start is latest:
        start = items[-2]  # a single step older than the window still counts
    return start, latest


class MomentFinder:
    """Intersects the bytes that moved the right way each time a person tells the AI."""

    def __init__(self, *, window_frames: int = WINDOW_FRAMES, settle_after: int = SETTLE_AFTER):
        if window_frames < 1 or settle_after < 1:
            raise ConfigurationError("window_frames and settle_after must be positive.")
        self._window = window_frames
        self._settle_after = settle_after
        self._candidates: dict[Moment, set[Address]] = {}
        self._presses: dict[Moment, int] = {}

    def presses(self, moment: Moment) -> int:
        return self._presses.get(moment, 0)

    def candidates(self, moment: Moment) -> tuple[Address, ...]:
        return tuple(sorted(self._candidates.get(moment, ())))

    def forget(self, moment: Moment | None = None) -> None:
        """Start again, for one moment or all of them (a new game, a settled value)."""
        if moment is None:
            self._candidates.clear()
            self._presses.clear()
        else:
            self._candidates.pop(moment, None)
            self._presses.pop(moment, None)

    def tell(self, moment: Moment, snapshots: Iterable[MemorySnapshot]) -> Narrowing:
        """The person says ``moment`` just happened; narrow its byte down."""
        start, latest = _window(snapshots, self._window)
        found: set[Address] = set()
        for region, data in latest.regions.items():
            if region not in start.regions or len(start.regions[region]) != len(data):
                continue
            moved = _moved(
                moment,
                np.frombuffer(start.regions[region], dtype=np.uint8),
                np.frombuffer(data, dtype=np.uint8),
            )
            found.update((region, int(o)) for o in np.flatnonzero(moved))
        previous = self._candidates.get(moment)
        started_again = False
        if previous is None:
            kept = found
            presses = 1
        else:
            kept = previous & found
            presses = self._presses[moment] + 1
            if not kept:
                kept, presses, started_again = found, 1, True
        self._candidates[moment] = kept
        self._presses[moment] = presses
        settled = next(iter(kept)) if len(kept) == 1 and presses >= self._settle_after else None
        text = self._describe(moment, kept, presses, settled, started_again)
        if settled is not None:
            self.forget(moment)
        return Narrowing(moment, presses, tuple(sorted(kept)), settled, text)

    def _describe(
        self,
        moment: Moment,
        kept: set[Address],
        presses: int,
        settled: Address | None,
        started_again: bool,
    ) -> str:
        n = len(kept)
        bytes_word = "byte" if n == 1 else "bytes"
        if settled is not None:
            region, offset = settled
            times = "once" if presses == 1 else f"{presses} times"
            return (
                f'Got it: "{moment.name}" is {region}[{offset}]. You said you {moment.words} '
                f"{times} and that byte {moment.movement} each time. The rewards and steps "
                "follow it."
            )
        if started_again:
            lead = f"Nothing {moment.movement} both times, so the AI started again: "
        elif presses == 1:
            lead = f"You {moment.words}: "
        else:
            lead = f"You {moment.words} {presses} times: "
        if n == 0:
            return (
                f"{lead}no byte {moment.movement} in {WINDOW_WORDS}. Press this right after "
                "it happens."
            )
        if n == 1 and presses < self._settle_after:
            return (
                f"{lead}one byte {moment.movement} in {WINDOW_WORDS}. Tell the AI once more "
                "when it happens again and it will take your word."
            )
        return (
            f"{lead}{n} {bytes_word} {moment.movement} in {WINDOW_WORDS}. Tell the AI again "
            "the next time it happens."
        )


class RestartWatch:
    """Votes for the byte that went down by one before each restart (T-904).

    The window ends at the snapshot *before* the restart, because the restart
    itself puts the counter back up; it starts ``window_frames`` earlier. A
    byte wins when it has at least ``needed`` votes, more than any other byte,
    and from at least two thirds of the restarts seen; after ``forget_after``
    restarts with no winner the count starts again, so a game whose restarts
    are mostly level changes never settles on noise.
    """

    def __init__(
        self,
        *,
        window_frames: int = WINDOW_FRAMES,
        needed: int = RESTARTS_NEEDED,
        forget_after: int = RESTARTS_FORGET,
    ) -> None:
        if window_frames < 1 or needed < 1 or forget_after < needed:
            raise ConfigurationError(
                "window_frames and needed must be positive, and forget_after at least needed."
            )
        self._window = window_frames
        self._needed = needed
        self._forget_after = forget_after
        self._restarts = 0
        self._votes: dict[Address, int] = {}

    @property
    def restarts(self) -> int:
        """Restarts counted since the last winner or forget."""
        return self._restarts

    def votes(self, address: Address) -> int:
        return self._votes.get(address, 0)

    def forget(self) -> None:
        self._restarts = 0
        self._votes = {}

    def restart(
        self, snapshots: Iterable[MemorySnapshot], *, skip: Collection[Address] = ()
    ) -> Address | None:
        """The latest snapshot is a restart; vote, and return the winner if there is one.

        ``skip`` holds bytes that cannot be the counter (known to be something
        else, or rejected); they get no vote.
        """
        items = list(snapshots)
        if len(items) < 3:
            return None  # the restart and at least two snapshots before it
        start, end = _window(items[:-1], self._window)
        found: set[Address] = set()
        for region, data in end.regions.items():
            if region not in start.regions or len(start.regions[region]) != len(data):
                continue
            moved = _moved(
                Moment.LOST_LIFE,
                np.frombuffer(start.regions[region], dtype=np.uint8),
                np.frombuffer(data, dtype=np.uint8),
            )
            found.update((region, int(o)) for o in np.flatnonzero(moved))
        found.difference_update(skip)
        self._restarts += 1
        for address in found:
            self._votes[address] = self._votes.get(address, 0) + 1
        winner = self._winner()
        if winner is not None or self._restarts >= self._forget_after:
            self.forget()
        return winner

    def _winner(self) -> Address | None:
        if self._restarts < self._needed or not self._votes:
            return None
        best = max(self._votes.values())
        leaders = [a for a, n in self._votes.items() if n == best]
        if best < self._needed or len(leaders) != 1 or best * 3 < self._restarts * 2:
            return None
        return leaders[0]


__all__ = [
    "RESTARTS_FORGET",
    "RESTARTS_NEEDED",
    "SETTLE_AFTER",
    "WINDOW_FRAMES",
    "Moment",
    "MomentFinder",
    "Narrowing",
    "RestartWatch",
]
