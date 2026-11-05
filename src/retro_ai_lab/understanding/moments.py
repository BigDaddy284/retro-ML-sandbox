"""A person says what just happened; the AI finds the byte that shows it (T-901).

The app cannot see that Mario died: it reads the game's memory, and a life
counter changes so rarely that statistics alone pick the wrong byte. A cheat
finder solves this because the *person* says when it happened. "I just lost a
life" does the same, child-sized: each press keeps only the bytes that moved
the right way in the last few seconds (down by one for a life, up for score
or progress), and the presses are intersected. When one byte is left after
enough presses, the AI takes it as the person's word.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot
from retro_ai_lab.understanding.knowledge import VariableKind

WINDOW_FRAMES = 300  # the last five seconds at 60 frames a second
SETTLE_AFTER = 2  # presses that agree before one byte is taken as the person's word
WINDOW_WORDS = "the last 5 seconds"


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


__all__ = ["SETTLE_AFTER", "WINDOW_FRAMES", "Moment", "MomentFinder", "Narrowing"]
