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
from retro_ai_lab.understanding.knowledge import (
    Confidence,
    Encoding,
    Knowledge,
    Variable,
    VariableKind,
)

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
HOME_OFTEN = 3  # times a value came to its first value in mid-play: a home-goer
HOMES_KEPT = 64  # different values of one variable whose comings are counted
"""Values that come home in the middle of play cannot make a restart by themselves
(T-1021).

After a jump the fall speed is back at nothing and "on the ground" is back on,
together, and one of them is a number that went down. Early in a go, before much
else has moved, that was the whole rule for a restart, and goes ended at their
first landing, three to six in a row. By then the AI had seen both values come to
those very values dozens of times while nearly everything else it follows did not
so much as change.

That is what the detector counts. A snapshot in which fewer than ``RESET_SHARE``
of the values that have moved changed at all is the game playing on: whatever the
game starting over does, it changes most of what has moved. For every value that
changed in such a snapshot, the value it came to is counted. A watched value whose
first value has been come to ``HOME_OFTEN`` times that way is a home-goer, and a
restart needs at least one value going home that is not one. A snapshot the app
made itself by loading a saved state is not counted.

Counted by the value come to, not by the variable: a switch has two resting
places, and which of them a watch calls its first value changes whenever the AI
learns something about the switch and the watch is made anew. A life count that
is watched from 1 comes to 1 in play and to 3 only when the game starts over: at 1
it is a home-goer, at 3 it is the sign of a restart that it should be. Only the
first ``HOMES_KEPT`` different values of a variable are counted; what a roaming
number comes to after that stays a sign of a restart, which is the safe side.

Three other ways of counting were built first and dropped. (1) Every return at
which no restart was called: a restart that goes uncalled for any other reason is
then a return "in play" for all it sends home, and the life count and the score
were home-goers after three. (2) A return within sixty snapshots of leaving: on
the racing game a whole game is shorter than that while the AI is still looking
about, the "game over" switch and the life count were home-goers before the first
go, no restart was called any more and the training ran aground (the first
demonstration's test failed). (3) A return with fewer than half of what had moved
going home with it: no better there, because at a restart nearly everything
changes but little of it lands on what its watch calls its first value. So the
measure is what changed, not what came home.

A home-goer still counts with the others. Leaving such values out altogether was
tried before any of this and dropped: on the racing game the next checkpoint goes
home every lap and also when the game starts over, and with it left out the game
starting over was no longer seen.
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


# Where a value is held and how it is read (T-1021): what was seen of a value is kept
# by this and not by its id, which the next guess takes over when a guess is removed.
_Place = tuple[str, int, Encoding, int]


def _place(var: Variable) -> _Place:
    return (var.region, var.offset, var.encoding, var.length)


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
        # The values each variable has come to while most of what had moved did not
        # stir, and how often (T-1021); and the variables that have just come to
        # their first value often enough. Kept across goes: it is what the AI has
        # seen of the game.
        self._homes: dict[_Place, dict[int, int]] = {}  # by place, then by the value come to
        self._said: set[_Place] = set()  # the places given out as home-goers
        self._home_goers: list[str] = []

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

    def homes(self, variable_id: str) -> int:
        """How often this value has come to the first value its watch has now, in
        snapshots in which most of what had moved did not change (T-1021)."""
        watch = self._watches.get(variable_id)
        return 0 if watch is None else self._times_home(watch)

    def take_home_goers(self) -> tuple[str, ...]:
        """The values found to be home-goers since last asked, each given once
        (T-1021)."""
        out, self._home_goers = tuple(self._home_goers), []
        return out

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
            # ... and what was seen at a place the knowledge holds nothing at any more
            # is dropped (T-1021).
            for place in [p for p in self._homes if self._knowledge.at(p[0], p[1]) is None]:
                del self._homes[place]
            self._said = {p for p in self._said if self._knowledge.at(p[0], p[1]) is not None}
        return self._tracked_list

    def rebase(self, snapshot: MemorySnapshot) -> None:
        """Take the values as they are in ``snapshot`` without judging how they moved.

        For a snapshot that follows the app putting the game back to an earlier
        moment of a go (T-1004): the life that came back, the progress that went
        down and the values that are at their first value again are the app's doing,
        not the game's. No event is raised and nothing is counted; each value's first
        value, its usual gain and whether it ever moved are kept, so what the game
        does next is judged from here as it would have been from there.
        """
        if not isinstance(snapshot, MemorySnapshot):
            raise ConfigurationError("EventDetector.rebase needs a MemorySnapshot.")
        for var in self._tracked(snapshot.region_names):
            try:
                value = var.decode(snapshot)
            except ConfigurationError:
                continue  # snapshot does not cover this variable
            watch = self._watches.get(var.id)
            if watch is None or watch.variable != var:
                if watch is not None and watch.moved:
                    self._moved -= 1
                self._watches[var.id] = _Watch(var, value, value)
            else:
                watch.last = value

    def observe(self, snapshot: MemorySnapshot, *, put_back: bool = False) -> tuple[GameEvent, ...]:
        """The events ``snapshot`` implies. ``put_back`` says it is the first snapshot
        since the app loaded a saved state (T-1021): its events are as ever, but a
        value at its first value again in it was put there, and is not counted as
        having come home in play."""
        if not isinstance(snapshot, MemorySnapshot):
            raise ConfigurationError("EventDetector.observe needs a MemorySnapshot.")
        if not isinstance(put_back, bool):
            raise ConfigurationError("put_back must be true or false.")
        self._observed += 1
        events: list[GameEvent] = []
        returned_home: list[_Watch] = []
        dropped = False
        stirred: list[_Watch] = []  # the values that changed at all (T-1021)
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
                stirred.append(watch)
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
        # ... and one of them is no home-goer: it does not come home in mid-play (T-1021).
        rare = any(self._times_home(w) < HOME_OFTEN for w in returned_home)
        would = len(returned_home) >= RESET_NEEDS and most and dropped and sure
        if would and rare:
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
        elif would:
            # Home-goers and nothing else: no restart. Any of them not yet given out
            # (its watch was made anew at a value already counted) is given out now.
            for watch in returned_home:
                if _place(watch.variable) not in self._said:
                    self._said.add(_place(watch.variable))
                    self._home_goers.append(watch.variable.id)
        elif len(stirred) < RESET_SHARE * self._moved and not put_back:
            # Most of what has moved did not stir: the game is playing on, and where
            # each value that changed has come to is somewhere it comes to in play.
            for watch in stirred:
                if _cycles(watch.variable):
                    continue
                place = _place(watch.variable)
                come_to = self._homes.setdefault(place, {})
                if watch.last not in come_to and len(come_to) >= HOMES_KEPT:
                    continue  # a number that roams: no more of its values are kept
                seen = come_to.get(watch.last, 0) + 1
                come_to[watch.last] = seen
                if seen == HOME_OFTEN and watch.last == watch.first and place not in self._said:
                    self._said.add(place)
                    self._home_goers.append(watch.variable.id)
        return tuple(events)

    def _times_home(self, watch: _Watch) -> int:
        """How often this value has come to its watch's first value in mid-play."""
        return self._homes.get(_place(watch.variable), {}).get(watch.first, 0)

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
    "HOMES_KEPT",
    "HOME_OFTEN",
    "RESET_NEEDS",
    "RESET_SHARE",
    "RESET_SURE",
    "EventDetector",
    "EventKind",
    "EventLog",
    "GameEvent",
]
