"""Event detection: turn variable changes into things that happened.

Once the knowledge holds a few believed variables, the stream of memory
snapshots can be read as a story: the lap counter went up (progress), a
life was lost (setback), the game-over flag came on, everything jumped back
to its starting values (reset). Events are what rewards (Phase 4) and
relationships (T-305) are built on, so they are plain, timestamped data.

The detector only uses variables the knowledge is at least a little
confident about and never invents a variable itself.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot
from retro_ai_lab.understanding.knowledge import Confidence, Knowledge, Variable, VariableKind


class EventKind(Enum):
    PROGRESS = "progress"  # a progress value stepped up
    SCORE_GAIN = "score_gain"  # score jumped by much more than usual
    SETBACK = "setback"  # lives or health went down
    RECOVERY = "recovery"  # health or lives went up
    FLAG_ON = "flag_on"  # a flag switched on (game over, menu, pause)
    FLAG_OFF = "flag_off"
    RESET = "reset"  # several values jumped back to their starting values

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").capitalize()


@dataclass(frozen=True, slots=True)
class GameEvent:
    kind: EventKind
    frame: int
    variable_id: str | None
    detail: str
    before: int | None = None
    after: int | None = None

    @property
    def delta(self) -> int | None:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before


@dataclass(slots=True)
class _Watch:
    variable: Variable
    first: int
    last: int
    typical_gain: float = 0.0  # running average of upward steps (scores)


class EventDetector:
    """Feed snapshots in order; get the events each one implies."""

    def __init__(
        self,
        knowledge: Knowledge,
        *,
        at_least: Confidence = Confidence.LOW,
        gain_factor: float = 3.0,
    ) -> None:
        if gain_factor <= 1:
            raise ConfigurationError("gain_factor must be greater than 1.")
        self._knowledge = knowledge
        self._at_least = at_least
        self._gain_factor = gain_factor
        self._watches: dict[str, _Watch] = {}
        self._observed = 0

    @property
    def observed(self) -> int:
        return self._observed

    @property
    def watching(self) -> tuple[str, ...]:
        return tuple(self._watches)

    def reset(self) -> None:
        self._watches.clear()
        self._observed = 0

    def _tracked(self) -> list[Variable]:
        return [
            v
            for v in self._knowledge
            if not v.rejected
            and self._at_least <= v.confidence
            and v.kind
            in (
                VariableKind.PROGRESS,
                VariableKind.SCORE,
                VariableKind.LIVES,
                VariableKind.HEALTH,
                VariableKind.FLAG,
            )
        ]

    def observe(self, snapshot: MemorySnapshot) -> tuple[GameEvent, ...]:
        if not isinstance(snapshot, MemorySnapshot):
            raise ConfigurationError("EventDetector.observe needs a MemorySnapshot.")
        self._observed += 1
        events: list[GameEvent] = []
        returned_home: list[_Watch] = []
        dropped = False
        for var in self._tracked():
            try:
                value = var.decode(snapshot)
            except ConfigurationError:
                continue  # snapshot does not cover this variable
            watch = self._watches.get(var.id)
            if watch is None or watch.variable != var:
                # New (or changed) variable: start watching, no event yet.
                self._watches[var.id] = _Watch(var, value, value)
                continue
            if value != watch.last:
                events.extend(self._judge(watch, value, snapshot.frame))
                if value == watch.first and value != watch.last:
                    returned_home.append(watch)
                    if value < watch.last:
                        dropped = True
            watch.last = value
        if len(returned_home) >= 2 and dropped:
            names = ", ".join(w.variable.display_name for w in returned_home)
            events.append(
                GameEvent(
                    EventKind.RESET,
                    snapshot.frame,
                    None,
                    f"{names} all went back to their starting values.",
                )
            )
        return tuple(events)

    def _judge(self, watch: _Watch, value: int, frame: int) -> list[GameEvent]:
        var = watch.variable
        before = watch.last
        name = var.display_name
        kind = var.kind
        out: list[GameEvent] = []
        if kind is VariableKind.PROGRESS and value > before:
            out.append(
                GameEvent(
                    EventKind.PROGRESS, frame, var.id, f"{name} went up to {value}.", before, value
                )
            )
        elif kind is VariableKind.SCORE and value > before:
            gain = value - before
            if watch.typical_gain == 0.0:
                watch.typical_gain = float(gain)
            elif gain >= self._gain_factor * watch.typical_gain:
                out.append(
                    GameEvent(
                        EventKind.SCORE_GAIN,
                        frame,
                        var.id,
                        f"{name} jumped by {gain}.",
                        before,
                        value,
                    )
                )
            # Learn what an ordinary step looks like (big jumps pull it up slowly).
            watch.typical_gain += (min(gain, 4 * watch.typical_gain) - watch.typical_gain) * 0.2
        elif kind in (VariableKind.LIVES, VariableKind.HEALTH):
            if value < before:
                out.append(
                    GameEvent(
                        EventKind.SETBACK,
                        frame,
                        var.id,
                        f"{name} dropped to {value}.",
                        before,
                        value,
                    )
                )
            else:
                out.append(
                    GameEvent(
                        EventKind.RECOVERY, frame, var.id, f"{name} rose to {value}.", before, value
                    )
                )
        elif kind is VariableKind.FLAG:
            on = value != 0
            out.append(
                GameEvent(
                    EventKind.FLAG_ON if on else EventKind.FLAG_OFF,
                    frame,
                    var.id,
                    f"{name} switched {'on' if on else 'off'}.",
                    before,
                    value,
                )
            )
        return out


class EventLog:
    """The most recent events, newest last."""

    def __init__(self, capacity: int = 1000) -> None:
        if capacity < 1:
            raise ConfigurationError("capacity must be positive.")
        self._items: deque[GameEvent] = deque(maxlen=capacity)
        self._total = 0

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[GameEvent]:
        return iter(self._items)

    @property
    def total(self) -> int:
        """Events ever recorded, including ones that have scrolled off."""
        return self._total

    def extend(self, events: tuple[GameEvent, ...]) -> None:
        for e in events:
            self._items.append(e)
        self._total += len(events)

    def recent(self, n: int) -> tuple[GameEvent, ...]:
        if n < 0:
            raise ConfigurationError("n must not be negative.")
        return tuple(list(self._items)[-n:]) if n else ()

    def count(self, kind: EventKind) -> int:
        return sum(1 for e in self._items if e.kind is kind)

    def since(self, frame: int) -> tuple[GameEvent, ...]:
        return tuple(e for e in self._items if e.frame >= frame)

    def clear(self) -> None:
        self._items.clear()
        self._total = 0


__all__ = ["EventDetector", "EventKind", "EventLog", "GameEvent"]
