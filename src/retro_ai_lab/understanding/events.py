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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot
from retro_ai_lab.understanding.knowledge import Confidence, Knowledge, Variable, VariableKind

RESET_NEEDS = 2  # values back at their starting values together, for a reset
RESET_SURE = Confidence.MEDIUM  # and at least one of them this sure (T-838)
RESET_SHARE = 0.5  # and at least this share of every value that has moved (T-846)
"""A restart is most of the game going home, not a few bytes (T-846).

With 1,600 believed values on the user's Super Mario Bros., some handful of
them was back at its first value on nearly every step while hundreds of
others were not, and every go ended within eight frames as a "reset" blaming
a different handful each time. The game starting over puts most of what has
moved back where it began, so the values going home together must be at
least half of all the values that have moved since the detector saw them.
"""
"""Shaky values going home together are not a restart (T-838).

On the user's Super Mario Bros. the 6502's stack page and other scratch
bytes in RAM bounce between a few values and land back on their first ones
together with no restart, and at LOW confidence two of them were enough to
end every go after a few steps. One of the values must be at least MEDIUM.
"""


class EventKind(Enum):
    PROGRESS = "progress"  # a progress value stepped up
    SCORE_GAIN = "score_gain"  # score jumped by much more than usual
    SETBACK = "setback"  # lives or health went down
    RECOVERY = "recovery"  # health or lives went up
    FLAG_ON = "flag_on"  # a flag switched on (game over, menu, pause)
    FLAG_OFF = "flag_off"
    RESET = "reset"  # several values jumped back to their starting values
    UP = "up"  # a value a person asked about went up, by any amount (T-860)
    DOWN = "down"  # ... or down
    PRESS = "press"  # a button was held this step; variable_id is the button's name (T-874)
    REPEAT = "repeat"  # ... and was held the step before as well
    NEW_GROUND = "new_ground"  # a step of new ground, measured off the screen (T-993)

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
    variables: tuple[str, ...] = ()  # a reset: the values that went back to their start

    @property
    def delta(self) -> int | None:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before


# Evidence kind on a value the AI found returning to its first value every few
# frames (T-845): its return is not the game starting over.
CYCLES = "cycles"

TRACKED_KINDS = frozenset(
    {
        VariableKind.PROGRESS,
        VariableKind.SCORE,
        VariableKind.LIVES,
        VariableKind.HEALTH,
        VariableKind.FLAG,
    }
)


def _cycles(var: Variable) -> bool:
    return any(e.kind == CYCLES for e in var.evidence)


@dataclass(slots=True)
class _Watch:
    variable: Variable
    first: int
    last: int
    typical_gain: float = 0.0  # running average of upward steps (scores)
    moved: bool = False  # has ever left its first value (T-846)


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
        self._moved = 0  # watches that have moved at least once (T-846)
        # The watch list, kept until the knowledge or the snapshot's regions change
        # (T-841): rebuilding it from 1,600 variables on every snapshot cost more than
        # the step itself, and a variable whose region the snapshot does not carry is
        # left out here instead of failing to decode every time.
        self._tracked_for: tuple[int, tuple[str, ...], frozenset[str]] | None = None
        self._tracked_list: list[Variable] = []
        self._asked: frozenset[str] = frozenset()  # values a person wants UP/DOWN for (T-860)

    @property
    def observed(self) -> int:
        return self._observed

    @property
    def watching(self) -> tuple[str, ...]:
        return tuple(self._watches)

    @property
    def moved(self) -> int:
        """How many watched values have ever left their first value."""
        return self._moved

    @property
    def asked(self) -> frozenset[str]:
        return self._asked

    def ask(self, var_ids: Iterable[str]) -> None:
        """Report every move of these values as UP or DOWN, whatever their kind (T-860)."""
        self._asked = frozenset(str(i) for i in var_ids)

    def reset(self) -> None:
        self._watches.clear()
        self._observed = 0
        self._moved = 0

    def _tracked(self, regions: tuple[str, ...]) -> list[Variable]:
        key = (self._knowledge.version, regions, self._asked)
        if key != self._tracked_for:
            self._tracked_list = [
                v
                for v in self._knowledge.believed(self._at_least, regions=regions)
                if v.kind in TRACKED_KINDS or v.id in self._asked
            ]
            seen = {v.id for v in self._tracked_list}
            for var_id in sorted(self._asked - seen):
                if var_id not in self._knowledge:
                    continue
                var = self._knowledge.get(var_id)
                if not var.rejected and var.region in regions:
                    self._tracked_list.append(var)
            self._tracked_for = key
            # A value that lost its seat (T-850) or its belief is no longer watched.
            keep = {v.id for v in self._tracked_list}
            for var_id in [i for i in self._watches if i not in keep]:
                if self._watches.pop(var_id).moved:
                    self._moved -= 1
        return self._tracked_list

    def observe(self, snapshot: MemorySnapshot) -> tuple[GameEvent, ...]:
        if not isinstance(snapshot, MemorySnapshot):
            raise ConfigurationError("EventDetector.observe needs a MemorySnapshot.")
        self._observed += 1
        events: list[GameEvent] = []
        returned_home: list[_Watch] = []
        dropped = False
        for var in self._tracked(snapshot.region_names):
            try:
                value = var.decode(snapshot)
            except ConfigurationError:
                continue  # snapshot does not cover this variable
            watch = self._watches.get(var.id)
            if watch is None or watch.variable != var:
                # New (or changed) variable: start watching, no event yet.
                if watch is not None and watch.moved:
                    self._moved -= 1
                self._watches[var.id] = _Watch(var, value, value)
                continue
            if value != watch.last:
                if not watch.moved:
                    watch.moved = True
                    self._moved += 1
                events.extend(self._judge(watch, value, snapshot.frame))
                if value == watch.first and value != watch.last and not _cycles(var):
                    returned_home.append(watch)
                    # A flag switching back is what flags do; a restart is a number
                    # (score, lives, progress) going back down to its start (T-867).
                    if value < watch.last and var.kind is not VariableKind.FLAG:
                        dropped = True
            watch.last = value
        sure = any(w.variable.confidence >= RESET_SURE for w in returned_home)
        most = len(returned_home) >= RESET_SHARE * self._moved
        if len(returned_home) >= RESET_NEEDS and most and dropped and sure:
            names = ", ".join(w.variable.display_name for w in returned_home)
            events.append(
                GameEvent(
                    EventKind.RESET,
                    snapshot.frame,
                    None,
                    f"{names} all went back to their starting values.",
                    variables=tuple(w.variable.id for w in returned_home),
                )
            )
        return tuple(events)

    def _judge(self, watch: _Watch, value: int, frame: int) -> list[GameEvent]:
        var = watch.variable
        before = watch.last
        name = var.display_name
        kind = var.kind
        out: list[GameEvent] = []
        if var.id in self._asked:
            up = value > before
            out.append(
                GameEvent(
                    EventKind.UP if up else EventKind.DOWN,
                    frame,
                    var.id,
                    f"{name} went {'up' if up else 'down'} to {value}.",
                    before,
                    value,
                )
            )
        if kind not in TRACKED_KINDS:
            return out
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


__all__ = [
    "CYCLES",
    "RESET_NEEDS",
    "RESET_SHARE",
    "RESET_SURE",
    "EventDetector",
    "EventKind",
    "EventLog",
    "GameEvent",
]
