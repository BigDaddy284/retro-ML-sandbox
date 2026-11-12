"""What goes up when something is lost is not progress (T-997).

A game's mode (playing, dying, game over), a death counter, a "hurt" state:
each steps up rarely, through a few values, more often up than down, and at
the same few places in the level, which is everything the classifier and the
long look ask of progress. On the practice game the mode byte was paid 10
points as "Progress goes up" at every death, against 7.5 taken for the life.

Nothing in such a byte's own statistics tells it from a checkpoint. When it
rises does: :class:`Aftermath` is told every event the detector finds and
counts, for each value, how many of its rises came as something was lost and
how many losses had its rise. Both must be most of them. Progress that is
often followed by a loss (a page crossed just before a jump the AI cannot make
yet) leaves most losses, the ones elsewhere, without its rise; a value that
rises at every loss and at many other times besides is not judged by this.

Where losses come thick and fast everything is near one. The first version
used a four-second window and no more, and on the toy racer (a checkpoint
every 16 frames, a life every 20) it dropped the real checkpoint, which the
existing bounce test caught. So the window is a second and a half, and nothing
is said unless the time near a loss is a third or less of the time watched:
only then is "most of its rises fall there" more than chance.
"""

from __future__ import annotations

from collections.abc import Iterable

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.understanding.events import EventKind, GameEvent

BEFORE = 90  # frames a rise may come before the loss it belongs to (a dying pause)
AFTER = 30  # ... or after it; losses this close together are one loss
MIN_COUNT = 5  # rises, and losses since it first rose, before anything is said
SHARE = 0.6  # of its rises at a loss, and of the losses with its rise
QUIET = 1 / 3  # the time near a loss is at most this share of the time watched

LOSS_KINDS = frozenset({EventKind.SETBACK, EventKind.RESET})


class _Rises:
    __slots__ = ("answered", "from_frames", "met", "rises", "since", "taken", "waiting")

    def __init__(self, since: int, from_frames: int) -> None:
        self.rises = 0
        self.met = 0  # rises that came as something was lost
        self.answered = 0  # losses that had a rise of this value
        self.since = since  # losses counted before this value first rose
        self.from_frames = from_frames  # frames watched before it first rose
        self.waiting: list[int] = []  # frames of rises no loss has claimed yet
        self.taken = False  # the last loss already has this value's rise


class Aftermath:
    """Counts, for each value, its rises against the losses around them."""

    def __init__(self) -> None:
        self._values: dict[str, _Rises] = {}
        self._losses = 0
        self._last_loss: int | None = None  # frame of the last loss, this go
        self._latest: int | None = None  # the last frame heard of, this go
        self._first: int | None = None  # the first frame heard of, this go
        self._watched = 0  # frames watched in the goes before this one

    def begin(self) -> None:
        """A go starts (the game was put back): nothing waits to be matched across it."""
        self._watched = self.watched
        self._last_loss = None
        self._latest = None
        self._first = None
        for value in self._values.values():
            value.waiting.clear()
            value.taken = False

    @property
    def watched(self) -> int:
        """Frames of play heard of so far, over every go."""
        if self._first is None or self._latest is None:
            return self._watched
        return self._watched + self._latest - self._first

    def note(self, events: Iterable[GameEvent], frame: int | None = None) -> None:
        """Take in what the detector found in one observation, made at ``frame``.

        Told the frame of every observation, eventful or not, it knows how long it
        has watched; told only of events, it knows the time between them.
        """
        if not isinstance(events, tuple) or not all(isinstance(e, GameEvent) for e in events):
            raise ConfigurationError("Aftermath.note takes a tuple of GameEvent.")
        if frame is not None:
            if isinstance(frame, bool) or not isinstance(frame, int):
                raise ConfigurationError("The frame of an observation is a whole number.")
            self._heard(frame)
        # The losses first: a rise in the same observation is "just after" its loss.
        for event in events:
            if event.kind in LOSS_KINDS:
                self._heard(event.frame)
                self._lost(event.frame)
        for event in events:
            if event.kind is EventKind.PROGRESS and event.variable_id is not None:
                self._heard(event.frame)
                self._rose(event.variable_id, event.frame)

    def counts(self, var_id: str) -> tuple[int, int, int, int]:
        """(rises, rises at a loss, losses since it first rose, losses with its rise)."""
        value = self._values.get(var_id)
        if value is None:
            return (0, 0, 0, 0)
        return (value.rises, value.met, self._losses - value.since, value.answered)

    def verdict(self, var_id: str) -> bool:
        """It goes up when something is lost: most of its rises, and at most losses,
        where the time near a loss is a small part of the time watched."""
        rises, met, losses, answered = self.counts(var_id)
        if rises < MIN_COUNT or losses < MIN_COUNT:
            return False
        if losses * (BEFORE + AFTER) > QUIET * (self.watched - self._values[var_id].from_frames):
            return False  # losses come so often that everything is near one
        return met >= SHARE * rises and answered >= SHARE * losses

    # ------------------------------------------------------------ helpers

    def _heard(self, frame: int) -> None:
        if self._latest is not None and frame < self._latest:
            self.begin()  # frames ran backwards: a saved spot was loaded
        if self._first is None:
            self._first = frame
        self._latest = frame

    def _lost(self, frame: int) -> None:
        if self._last_loss is not None and frame - self._last_loss <= AFTER:
            return  # the same loss, seen as another event
        self._losses += 1
        self._last_loss = frame
        for value in self._values.values():
            near = [f for f in value.waiting if frame - f <= BEFORE]
            value.waiting.clear()
            value.met += len(near)
            value.taken = bool(near)
            if near:
                value.answered += 1

    def _rose(self, var_id: str, frame: int) -> None:
        just_lost = self._last_loss is not None and frame - self._last_loss <= AFTER
        value = self._values.get(var_id)
        if value is None:
            # A loss it rises just after is one it was there for.
            lost_before = self._losses - (1 if just_lost else 0)
            value = self._values[var_id] = _Rises(lost_before, self.watched)
        value.rises += 1
        if not just_lost:
            value.waiting.append(frame)
            return
        value.met += 1
        if not value.taken:
            value.taken = True
            value.answered += 1


__all__ = ["AFTER", "BEFORE", "LOSS_KINDS", "MIN_COUNT", "QUIET", "SHARE", "Aftermath"]
