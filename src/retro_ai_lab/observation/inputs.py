"""Input recording: who pressed what, and when.

Every change of the held button set is an :class:`InputEvent` tagged with
the frame it took effect on and its *source* (``human`` for the pad and
keyboard, ``ai`` for an agent, ``replay`` for playback). The
:class:`InputLog` keeps them in order and can answer "what was held at
frame N" and "how long was each button held", which the understanding
engine uses to link inputs to memory and screen changes.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from retro_ai_lab.core import ConfigurationError

SOURCES = ("human", "ai", "replay")


@dataclass(frozen=True, slots=True)
class InputEvent:
    frame: int
    buttons: frozenset[str]
    source: str = "human"

    def __post_init__(self) -> None:
        if not isinstance(self.frame, int) or isinstance(self.frame, bool) or self.frame < 0:
            raise ConfigurationError("frame must be a non-negative integer.")
        if self.source not in SOURCES:
            raise ConfigurationError(f"source must be one of {SOURCES}, not {self.source!r}.")
        names = frozenset(self.buttons)
        if any(not isinstance(n, str) or not n for n in names):
            raise ConfigurationError("buttons must be non-empty names.")
        object.__setattr__(self, "buttons", names)

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "buttons": sorted(self.buttons), "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputEvent:
        try:
            return cls(data["frame"], frozenset(data["buttons"]), data.get("source", "human"))
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(f"Bad input event: {exc}") from exc


@dataclass(frozen=True, slots=True)
class HeldSpan:
    """One stretch of frames during which the held set did not change."""

    start: int
    end: int | None  # exclusive; None while still current
    buttons: frozenset[str]
    source: str

    @property
    def length(self) -> int | None:
        return None if self.end is None else self.end - self.start


class InputLog:
    """Ordered record of input changes with a bounded memory."""

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ConfigurationError("capacity must be positive.")
        self._events: deque[InputEvent] = deque(maxlen=capacity)
        self._dropped = 0

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[InputEvent]:
        return iter(self._events)

    @property
    def capacity(self) -> int:
        return self._events.maxlen or 0

    @property
    def dropped(self) -> int:
        """Oldest events forgotten because the log was full."""
        return self._dropped

    @property
    def events(self) -> tuple[InputEvent, ...]:
        return tuple(self._events)

    @property
    def latest(self) -> InputEvent | None:
        return self._events[-1] if self._events else None

    def clear(self) -> None:
        self._events.clear()
        self._dropped = 0

    def record(
        self, frame: int, buttons: Iterable[str], source: str = "human"
    ) -> InputEvent | None:
        """Append a change. Returns None (and records nothing) if nothing changed.

        Frames must not go backwards; a change on the same frame as the
        previous one replaces it (only the final state of a frame matters).
        """
        event = InputEvent(frame, frozenset(buttons), source)
        last = self.latest
        if last is not None:
            if frame < last.frame:
                raise ConfigurationError(
                    f"Input at frame {frame} arrived after frame {last.frame}; "
                    "frames must not go back."
                )
            if frame == last.frame:
                self._events.pop()
                previous = self.latest
                if previous is not None and previous.buttons == event.buttons:
                    return None  # replaced back to what it already was
                self._append(event)
                return event
            if last.buttons == event.buttons:
                return None
        elif not event.buttons:
            return None  # releasing nothing at the very start is not an event
        self._append(event)
        return event

    def _append(self, event: InputEvent) -> None:
        if len(self._events) == self._events.maxlen:
            self._dropped += 1
        self._events.append(event)

    # ----------------------------------------------------------- queries

    def held_at(self, frame: int) -> frozenset[str]:
        """Buttons held on ``frame`` (empty before the first recorded event)."""
        held: frozenset[str] = frozenset()
        for e in self._events:
            if e.frame > frame:
                break
            held = e.buttons
        return held

    def spans(self) -> tuple[HeldSpan, ...]:
        """Stretches of constant input, in order; the last one is open-ended."""
        out: list[HeldSpan] = []
        events = list(self._events)
        for i, e in enumerate(events):
            end = events[i + 1].frame if i + 1 < len(events) else None
            out.append(HeldSpan(e.frame, end, e.buttons, e.source))
        return tuple(out)

    def frames_held(self, button: str, *, until: int | None = None) -> int:
        """How many frames ``button`` was down, up to ``until`` (exclusive)."""
        total = 0
        for span in self.spans():
            if button not in span.buttons:
                continue
            end = span.end if span.end is not None else until
            if end is None:
                raise ConfigurationError("Give `until` to count an input that is still held.")
            if until is not None:
                end = min(end, until)
            total += max(0, end - span.start)
        return total

    def presses(self, button: str) -> int:
        """How many times ``button`` went from up to down."""
        count = 0
        down = False
        for e in self._events:
            now = button in e.buttons
            if now and not down:
                count += 1
            down = now
        return count

    def by_source(self, source: str) -> tuple[InputEvent, ...]:
        return tuple(e for e in self._events if e.source == source)

    # ------------------------------------------------------ serialisation

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events]

    @classmethod
    def from_list(cls, items: Sequence[dict[str, Any]], capacity: int = 10_000) -> InputLog:
        log = cls(capacity)
        for item in items:
            e = InputEvent.from_dict(item)
            if log.latest is not None and e.frame < log.latest.frame:
                raise ConfigurationError("Input events are out of order.")
            log._append(e)
        return log


__all__ = ["SOURCES", "HeldSpan", "InputEvent", "InputLog"]
