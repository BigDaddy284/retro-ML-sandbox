"""The moments saved during a go, and what the AI did since each (T-1003).

To try small changes past a place it died (T-999), the app needs somewhere to
go back to: a save made well before the life went, and exactly what was done
from that save on. This module is that memory and nothing else. It is told
what happens (a save was made, a step was played, the story went back) and
answers three questions: is a save due, into which slot, and at a life lost
which moment to go back to with which moves since. It never touches the
emulator: making and loading the saves is the caller's work.

The numbers are the prototype's. It saved every 20 steps into 6 slots and went
back to the newest save at least 30 steps before the life went; every way past
it found on the practice game began 15 to 33 steps before. A save closer than
that is often already past the point where anything could be done.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError

EVERY = 20  # steps from one saved moment to the next
LEAD = 30  # a moment to go back to lies at least this many steps before the loss


@dataclass(frozen=True)
class Moment:
    """A save made during a go: where it is kept, and when in the story it was made."""

    slot: str
    step: int  # steps of the story played before it
    mark: object = None  # whatever the caller wants back with it (the screen watch's mark)


def _whole(value: object, what: str, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ConfigurationError(f"{what} must be a whole number, {least} or more.")
    return value


def _names(slots: object) -> tuple[str, ...]:
    if isinstance(slots, str) or not isinstance(slots, Sequence) or not slots:
        raise ConfigurationError("Saved moments need a list of slot names to be kept in.")
    for slot in slots:
        if not isinstance(slot, str) or not slot or not slot.replace("_", "").isalnum():
            raise ConfigurationError(
                f"A slot must be a name of letters, digits or underscores, got {slot!r}."
            )
    if len(set(slots)) != len(slots):
        raise ConfigurationError("Each slot can be named only once.")
    return tuple(slots)


class Moments:
    """The saved moments of one story, oldest first, and the moves made since the oldest.

    A story is what the game has been through as it now stands: going back to a
    moment takes the steps after it out of the story. ``steps`` counts the
    story's steps and so goes back with it.
    """

    def __init__(self, slots: Sequence[str], *, every: int = EVERY, lead: int = LEAD) -> None:
        self._slots = _names(slots)
        self._every = _whole(every, "The steps between saved moments", 1)
        self._lead = _whole(lead, "The steps a moment must lie before a loss", 1)
        self._steps = 0
        self._kept: list[Moment] = []
        self._moves: list[int] = []  # the moves made since the oldest moment kept

    @property
    def slots(self) -> tuple[str, ...]:
        """The names of the slots the moments are kept in."""
        return self._slots

    @property
    def steps(self) -> int:
        """Steps of the story so far."""
        return self._steps

    @property
    def kept(self) -> tuple[Moment, ...]:
        """The moments kept, oldest first."""
        return tuple(self._kept)

    def __len__(self) -> int:
        return len(self._kept)

    def due(self) -> bool:
        """Whether a save should be made before the next step."""
        return not self._kept or self._steps - self._kept[-1].step >= self._every

    def next_slot(self) -> str:
        """The slot the next save goes into: a free one, else the oldest moment's."""
        used = {moment.slot for moment in self._kept}
        for slot in self._slots:
            if slot not in used:
                return slot
        return self._kept[0].slot

    def saved(self, mark: object = None) -> Moment:
        """A save was just made into :meth:`next_slot`. Returns the moment it is.

        Where every slot was in use the oldest moment is forgotten: its save is
        the one that was written over.
        """
        if self._kept and self._kept[-1].step == self._steps:
            raise ConfigurationError("This step of the story is already saved.")
        moment = Moment(slot=self.next_slot(), step=self._steps, mark=mark)
        if len(self._kept) == len(self._slots):
            del self._kept[0]
        self._kept.append(moment)
        del self._moves[: len(self._moves) - (self._steps - self._kept[0].step)]
        return moment

    def did(self, move: int | None) -> None:
        """One step was played with this move.

        ``None`` is a step that cannot be played again (not one of the AI's own
        moves): no moment before it can be replayed up to here, so all are
        forgotten.
        """
        if move is None:
            self._steps += 1
            self.forget()
            return
        _whole(move, "A move", 0)
        self._steps += 1
        if self._kept:
            self._moves.append(move)

    def before_loss(self) -> tuple[Moment, tuple[int, ...]] | None:
        """The moment to go back to after a life lost now, and the moves made since it.

        The newest moment at least ``lead`` steps back; ``None`` when there is none.
        """
        for moment in reversed(self._kept):
            if self._steps - moment.step >= self._lead:
                return moment, tuple(self._moves[moment.step - self._kept[0].step :])
        return None

    def since(self, step: int) -> tuple[Moment, tuple[int, ...]] | None:
        """The moment to go back to for something that began at ``step`` of the story,
        and the moves made since it (T-1018).

        The newest moment at or before that step; where every moment kept is later,
        the oldest. ``None`` when none is kept.
        """
        _whole(step, "The step it began at", 0)
        if not self._kept:
            return None
        found = self._kept[0]
        for moment in self._kept:
            if moment.step <= step:
                found = moment
        return found, tuple(self._moves[found.step - self._kept[0].step :])

    def went_back(self, moment: Moment) -> None:
        """The game is at ``moment`` again: what came after it is no longer the story."""
        at = next((i for i, kept in enumerate(self._kept) if kept is moment), None)
        if at is None:
            raise ConfigurationError("That moment is not one of those kept.")
        del self._kept[at + 1 :]
        del self._moves[moment.step - self._kept[0].step :]
        self._steps = moment.step

    def forget(self) -> None:
        """Drop every moment. For a loss that stands: what was saved before it
        belongs to a story that no longer leads to where the game is."""
        self._kept.clear()
        self._moves.clear()


__all__ = ["EVERY", "LEAD", "Moment", "Moments"]
