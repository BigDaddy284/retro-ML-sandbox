"""A longer look at progress-like values (DESIGN.md §16).

The first-guess classifier works from how each byte moves on its own. Some
things only show over time and between values: a lap counter looks like a
flag while it has only ever been 0 and 1; a checkpoint counter is just a
slow number until you notice it only steps up when the position is at the
same few places; and a lap steps up each time the checkpoints start over.
The long look watches believed flags, progress values and positions
("steppers") against believed positions and progress values ("drivers")
and, when asked to review, writes what it found into the knowledge as
evidence, re-kinding a flag or a few-valued "position" that turned out to
count progress.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot
from retro_ai_lab.understanding.knowledge import (
    Confidence,
    Evidence,
    Knowledge,
    Variable,
    VariableKind,
    is_mover,
)

STEPPER_KINDS = frozenset({VariableKind.FLAG, VariableKind.PROGRESS, VariableKind.POSITION})
DRIVER_KINDS = frozenset({VariableKind.POSITION, VariableKind.PROGRESS})
MIN_STEPS = 3  # steps up before a pattern is believed
FEW_PLACES = 8  # distinct driver values that still count as "the same few places"
FEW_VALUES = 16  # a stepper with more values than this is a position, not a checkpoint
WIDE = 2 * FEW_PLACES  # a driver needs this much range for "the same few places" to mean much
PLACE_SLACK = 4  # driver values this close together count as one place (it moves a few a frame)
# Bounds on the pairing work (T-849): on a title demo hundreds of believed values
# step every frame, and recording where every driver was for each of them was
# quadratic in the believed values and the largest cost of a training step.
SETTLED = 64  # observations of a stepper before its rate is judged
TOO_FAST = 8  # a stepper up on more than one observation in this many is a timer, not a counter
MAX_PLACE_VALUES = FEW_PLACES * (PLACE_SLACK + 1)  # more driver values than can make FEW_PLACES
THIRD_VALUE = "third-value"
SAME_PLACES = "same-places"
COUNTS_ROUNDS = "counts-rounds"
LONG_LOOK_KINDS = frozenset({THIRD_VALUE, SAME_PLACES, COUNTS_ROUNDS})


@dataclass
class _Stepper:
    last: int
    distinct: set[int] = field(default_factory=set)
    ups: int = 0
    downs: int = 0
    # driver id -> driver value at a step -> what the stepper stepped to there
    places: dict[str, dict[int, Counter[int]]] = field(default_factory=dict)
    rounds: Counter[str] = field(default_factory=Counter)  # driver id -> starts-over at steps
    seen: int = 0  # observations since first watched (T-849)
    tracking: bool = True  # still worth pairing with the drivers
    dead: set[str] = field(default_factory=set)  # drivers it stepped at too many places of

    def stop_tracking(self) -> None:
        self.tracking = False
        self.places.clear()
        self.rounds.clear()
        self.dead.clear()


@dataclass
class _Driver:
    last: int
    low: int
    high: int

    def starts_over(self, value: int) -> bool:
        """Dropped by at least half of its seen range: back to the start."""
        span = self.high - self.low
        return span > 0 and value < self.last and (self.last - value) * 2 >= span


def cluster_places(
    values: Mapping[int, Counter[int]], slack: int = PLACE_SLACK
) -> tuple[int, int, bool]:
    """(places, coverage, consistent) for the driver values a stepper stepped at.

    Driver values within ``slack`` of each other are one place (a position moves a few
    units a frame); ``coverage`` is how many driver values the places span in all; and
    the places are ``consistent`` when the stepper stepped to the same value nearly every
    time at each place (a checkpoint counter does; a speed ramping up after a respawn at
    the same spot does not).
    """
    places = 0
    coverage = 0
    consistent = True
    start: int | None = None
    last: int | None = None
    at_place: Counter[int] = Counter()

    def close() -> None:
        nonlocal coverage, consistent
        if start is not None and last is not None:
            coverage += last - start + 1
        total = sum(at_place.values())
        if total and max(at_place.values()) * 10 < total * 9:
            consistent = False

    for value in sorted(values):
        if last is None or value - last > slack:
            close()
            at_place = Counter()
            places += 1
            start = value
        at_place.update(values[value])
        last = value
    close()
    return places, coverage, consistent


class LongLook:
    def __init__(self, knowledge: Knowledge, *, at_least: Confidence = Confidence.LOW) -> None:
        self._knowledge = knowledge
        self._at_least = at_least
        self._steppers: dict[str, _Stepper] = {}
        self._drivers: dict[str, _Driver] = {}
        self._observed = 0
        # Believed lists per kind set, kept until the knowledge or the snapshot's
        # regions change (T-841); see EventDetector._tracked.
        self._believed_for: tuple[int, tuple[str, ...]] | None = None
        self._believed_lists: dict[frozenset[VariableKind], list[Variable]] = {}

    @property
    def observed(self) -> int:
        return self._observed

    @property
    def watching(self) -> tuple[str, ...]:
        return tuple(self._steppers)

    @property
    def tracking(self) -> tuple[str, ...]:
        """The watched steppers still being paired with the drivers (T-849)."""
        return tuple(i for i, s in self._steppers.items() if s.tracking)

    def reset(self) -> None:
        self._steppers.clear()
        self._drivers.clear()
        self._observed = 0

    def _believed(self, kinds: frozenset[VariableKind], regions: tuple[str, ...]) -> list[Variable]:
        key = (self._knowledge.version, regions)
        if key != self._believed_for:
            self._believed_lists.clear()
            self._believed_for = key
        found = self._believed_lists.get(kinds)
        if found is None:
            found = [
                v
                for v in self._knowledge.believed(self._at_least, regions=regions)
                if v.kind in kinds and not is_mover(v)  # a thing moving on its own (T-911)
            ]
            self._believed_lists[kinds] = found
        return found

    def rebase(self, snapshot: MemorySnapshot) -> None:
        """Take the values as they are in ``snapshot`` without counting how they moved.

        For a snapshot that follows the app putting the game back to an earlier
        moment of a go (T-1004): no step up or down, no place a value stepped at and
        no starting over is counted across the jump.
        """
        if not isinstance(snapshot, MemorySnapshot):
            raise ConfigurationError("LongLook.rebase needs a MemorySnapshot.")
        for var in self._believed(DRIVER_KINDS, snapshot.region_names):
            try:
                value = var.decode(snapshot)
            except ConfigurationError:
                continue
            driver = self._drivers.get(var.id)
            if driver is None:
                self._drivers[var.id] = _Driver(value, value, value)
            else:
                driver.last = value
        for var in self._believed(STEPPER_KINDS, snapshot.region_names):
            try:
                value = var.decode(snapshot)
            except ConfigurationError:
                continue
            stepper = self._steppers.get(var.id)
            if stepper is None:
                self._steppers[var.id] = _Stepper(value, {value})
            else:
                stepper.last = value

    def observe(self, snapshot: MemorySnapshot) -> None:
        if not isinstance(snapshot, MemorySnapshot):
            raise ConfigurationError("LongLook.observe needs a MemorySnapshot.")
        self._observed += 1
        # Drivers first: a stepper wants to know where every driver is on this frame.
        now: dict[str, int] = {}
        started_over: set[str] = set()
        for var in self._believed(DRIVER_KINDS, snapshot.region_names):
            try:
                value = var.decode(snapshot)
            except ConfigurationError:
                continue
            now[var.id] = value
            driver = self._drivers.get(var.id)
            if driver is None:
                self._drivers[var.id] = _Driver(value, value, value)
                continue
            if driver.starts_over(value):
                started_over.add(var.id)
            driver.last = value
            driver.low = min(driver.low, value)
            driver.high = max(driver.high, value)
        for var in self._believed(STEPPER_KINDS, snapshot.region_names):
            try:
                value = var.decode(snapshot)
            except ConfigurationError:
                continue
            stepper = self._steppers.get(var.id)
            if stepper is None:
                self._steppers[var.id] = _Stepper(value, {value})
                continue
            stepper.distinct.add(value)
            stepper.seen += 1
            if value > stepper.last:
                stepper.ups += 1
                if stepper.tracking and (
                    len(stepper.distinct) > FEW_VALUES
                    or (stepper.seen >= SETTLED and stepper.ups * TOO_FAST > stepper.seen)
                ):
                    stepper.stop_tracking()  # too many faces, or too fast, for a counter
                if stepper.tracking:
                    self._record_places(var.id, stepper, value, now, started_over)
            elif value < stepper.last:
                stepper.downs += 1
            stepper.last = value

    @staticmethod
    def _record_places(
        var_id: str,
        stepper: _Stepper,
        value: int,
        now: dict[str, int],
        started_over: set[str],
    ) -> None:
        """Note where every driver is as this stepper steps up."""
        dead = stepper.dead
        places = stepper.places
        for driver_id, driver_value in now.items():
            if driver_id == var_id or driver_id in dead:
                continue
            at = places.get(driver_id)
            if at is None:
                at = places[driver_id] = {}
            counter = at.get(driver_value)
            if counter is None:
                if len(at) >= MAX_PLACE_VALUES:
                    # Stepped at too many places of this driver for "the same few".
                    del places[driver_id]
                    dead.add(driver_id)
                    continue
                counter = at[driver_value] = Counter()
            counter[value] += 1
            if driver_id in started_over:
                stepper.rounds[driver_id] += 1

    def review(self) -> tuple[Variable, ...]:
        """Write what the long look found into the knowledge; returns the variables changed."""
        changed: list[Variable] = []
        for var_id, stepper in list(self._steppers.items()):
            if var_id not in self._knowledge:
                del self._steppers[var_id]
                continue
            var = self._knowledge.get(var_id)
            if var.rejected or var.kind not in STEPPER_KINDS:
                continue  # rejected, or re-kinded by someone else since it was first watched
            kinds = {e.kind for e in var.evidence}
            evidence: list[Evidence] = []
            counts = False  # found to count progress (a flag or a "position" is re-kinded)
            few = len(stepper.distinct) <= FEW_VALUES
            # A value that has only ever shown two faces is a flag until a third shows up:
            # a game-over flag also "steps up at the same place" and "when the track starts
            # over", and that proves nothing about counting.
            several = len(stepper.distinct) >= 3
            if (
                var.kind is VariableKind.FLAG
                and several
                and stepper.ups >= stepper.downs
                and THIRD_VALUE not in kinds
            ):
                counts = True
                evidence.append(
                    Evidence(
                        THIRD_VALUE,
                        "Reached a third value, so it counts up rather than switches.",
                        1.5,
                    )
                )
            if stepper.ups >= MIN_STEPS and few and several:
                for driver_id, places in stepper.places.items():
                    if driver_id not in self._knowledge:
                        continue
                    driver = self._knowledge.get(driver_id)
                    track = self._drivers.get(driver_id)
                    if driver.rejected or driver.kind not in DRIVER_KINDS or track is None:
                        continue
                    spots, coverage, consistent = cluster_places(places)
                    steps_there = sum(sum(c.values()) for c in places.values())
                    if (
                        driver.kind is VariableKind.POSITION
                        and track.high - track.low >= WIDE
                        and SAME_PLACES not in kinds
                        and consistent
                        and spots <= FEW_PLACES
                        and coverage * 4 <= track.high - track.low + 1  # a quarter of the way
                        and steps_there >= 2 * spots
                    ):
                        counts = True
                        evidence.append(
                            Evidence(
                                SAME_PLACES,
                                f"Steps up when {driver.display_name} is at the same few places.",
                                1.0,
                            )
                        )
                        kinds.add(SAME_PLACES)
                    rounds = stepper.rounds[driver_id]
                    if (
                        COUNTS_ROUNDS not in kinds
                        and rounds >= 2
                        and rounds * 5 >= stepper.ups * 4  # nearly every step is a start-over
                    ):
                        counts = True
                        evidence.append(
                            Evidence(
                                COUNTS_ROUNDS,
                                f"Goes up each time {driver.display_name} starts over.",
                                1.5,
                            )
                        )
                        kinds.add(COUNTS_ROUNDS)
            if not evidence:
                continue
            changes: dict[str, object] = {"evidence": var.evidence + tuple(evidence)}
            if counts and few and var.kind is not VariableKind.PROGRESS:
                changes["kind"] = VariableKind.PROGRESS
            changed.append(self._knowledge.update(var_id, **changes))
        return tuple(changed)


__all__ = [
    "COUNTS_ROUNDS",
    "FEW_PLACES",
    "FEW_VALUES",
    "LONG_LOOK_KINDS",
    "MAX_PLACE_VALUES",
    "MIN_STEPS",
    "PLACE_SLACK",
    "SAME_PLACES",
    "SETTLED",
    "THIRD_VALUE",
    "TOO_FAST",
    "WIDE",
    "LongLook",
    "cluster_places",
]
