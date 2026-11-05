"""Discrete states for a beginner learner (DESIGN.md §6.1).

The understanding engine believes some bytes mean something: a lane, a
speed, lives, a lap counter. A tabular learner needs a small, stable key
for "the situation I'm in", so the encoder decodes every believed variable
and bins its value with one rule:

* a flag is 0 or 1;
* a value below ``bins`` is kept exactly (lanes, speeds, lives and laps
  stay precise);
* a larger value is grouped by doubling (``bins + log2(value / bins)``), so
  a position or a score becomes "roughly where" or "roughly how much";
* a negative value mirrors the positive rule below zero.

The better the engine's beliefs, the more the learner can see. Nothing
believed yet means an empty state, and the learner still works, just
blindly.

Where things are relative to the player matters more than where they are
(T-912): the user's AI died to the first enemy because, in doubling bins, an
enemy 60 pixels off and one touching the player looked the same. So for
every position that moves on its own (a mover, T-911) the state also holds
its distance from each position of the player's own (a position that is not
a mover), as ``mover - mine`` under the same doubling rule: exact within
``bins``, then "near", "a bit off", "far", negative when it is behind. A
distance part's id is ``distance_id(mover, mine)``. When any mover is
believed, ``MOST_DISTANCES`` of the ``most`` seats are kept for distances.

Which beliefs get in matters as much (T-908): the brain's feature map has
``most`` slots, and the user's game had hundreds of flag guesses at low
addresses taking every one of them before a position value got a look in.
The encoder picks kind by kind in :data:`KIND_RANK` order, the most trusted
of each kind first, going round the kinds until ``most`` are chosen, so a
position, the lives and the score are in before the twentieth switch. The
chosen ones are still ordered by address, so a tabular key stays stable.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot
from retro_ai_lab.understanding.knowledge import (
    Confidence,
    Knowledge,
    Variable,
    VariableKind,
    belief_rank,
    is_mover,
)

DEFAULT_BINS = 8
DEFAULT_MOST = 32  # the feature map's width (learning/features.py); kept in step by a test
MOST_DISTANCES = 8  # seats kept for mover-to-player distances when anything moves (T-912)
DISTANCE_SEP = "~"  # a distance part's id is "<mover id>~<my position's id>"
IGNORED_KINDS = frozenset({VariableKind.UNKNOWN, VariableKind.INPUT, VariableKind.COUNTER})
# The kinds a brain needs first: where things are, how it is doing, then the switches.
KIND_RANK = (
    VariableKind.POSITION,
    VariableKind.LIVES,
    VariableKind.HEALTH,
    VariableKind.SPEED,
    VariableKind.SCORE,
    VariableKind.PROGRESS,
    VariableKind.FLAG,
)


def bucket(value: int, kind: VariableKind, bins: int = DEFAULT_BINS) -> int:
    """The bin a value falls in (see the module docstring)."""
    if kind is VariableKind.FLAG:
        return 1 if value else 0
    if value < 0:
        return -1 - bucket(-value, kind, bins)
    if value < bins:
        return value
    return bins + (value // bins).bit_length() - 1


def distance_id(mover: str, mine: str) -> str:
    """The id of the state part holding how far ``mover`` is from ``mine`` (T-912)."""
    if not mover or not mine or DISTANCE_SEP in mover or DISTANCE_SEP in mine:
        raise ConfigurationError("distance_id needs two plain variable ids.")
    return f"{mover}{DISTANCE_SEP}{mine}"


def split_distance(vid: str) -> tuple[str, str] | None:
    """``(mover id, my position's id)`` for a distance part's id, else None."""
    mover, sep, mine = vid.partition(DISTANCE_SEP)
    return (mover, mine) if sep and mover and mine else None


@dataclass(frozen=True, slots=True)
class State:
    """Which bucket each believed variable is in, in a fixed order."""

    parts: tuple[tuple[str, int], ...] = ()

    @property
    def key(self) -> str:
        return ";".join(f"{vid}={b}" for vid, b in self.parts)

    @classmethod
    def from_key(cls, key: str) -> State:
        if not key:
            return cls()
        try:
            parts = tuple((vid, int(b)) for vid, _, b in (p.partition("=") for p in key.split(";")))
        except ValueError as exc:
            raise ConfigurationError(f"Bad state key {key!r}.") from exc
        return cls(parts)

    def describe(self, knowledge: Knowledge | None = None, *, most: int | None = None) -> str:
        """The situation in words; ``most`` caps the values named ("and N more", T-889)."""
        if not self.parts:
            return "nothing known yet"
        if most is not None and most < 1:
            raise ConfigurationError("most must be at least 1 (or None).")
        words = []
        for vid, b in self.parts:
            words.append(f"{_part_name(vid, knowledge)}={b}")
        if most is not None and len(words) > most:
            left = len(words) - most
            return ", ".join(words[:most]) + f" and {left} more"
        return ", ".join(words)


def _part_name(vid: str, knowledge: Knowledge | None) -> str:
    pair = split_distance(vid)
    if pair is not None:
        mover, mine = (_part_name(x, knowledge) for x in pair)
        return f"{mover} from {mine}"
    return knowledge.get(vid).display_name if knowledge and vid in knowledge else vid


class StateEncoder:
    def __init__(
        self,
        knowledge: Knowledge,
        *,
        at_least: Confidence = Confidence.LOW,
        bins: int = DEFAULT_BINS,
        most: int | None = DEFAULT_MOST,
    ) -> None:
        if not 2 <= bins <= 64:
            raise ConfigurationError("bins must be between 2 and 64.")
        if most is not None and most < 1:
            raise ConfigurationError("most must be at least 1 (or None for no cap).")
        self._knowledge = knowledge
        self._at_least = at_least
        self._bins = bins
        self._most = most
        self._variables: tuple[Variable, ...] = ()
        self._pairs: tuple[tuple[Variable, Variable], ...] = ()
        self.refresh()

    @property
    def bins(self) -> int:
        return self._bins

    @property
    def most(self) -> int | None:
        """The most variables a state holds (None: every believed one)."""
        return self._most

    @property
    def variables(self) -> tuple[Variable, ...]:
        """The believed variables the state is built from, lowest address first."""
        return self._variables

    @property
    def pairs(self) -> tuple[tuple[Variable, Variable], ...]:
        """``(mover, mine)`` pairs whose distance the state holds (T-912)."""
        return self._pairs

    def refresh(self) -> tuple[Variable, ...]:
        """Re-read the knowledge; call after discoveries change."""
        believed = [
            v for v in self._knowledge.believed(self._at_least) if v.kind not in IGNORED_KINDS
        ]
        most = self._most
        if most is not None and any(is_mover(v) for v in believed):
            most = max(1, most - MOST_DISTANCES)
        self._variables = tuple(sorted(_choose(believed, most), key=lambda v: (v.region, v.offset)))
        self._pairs = _pair_up(self._variables)
        return self._variables

    def encode(self, snapshot: MemorySnapshot) -> State:
        parts: list[tuple[str, int]] = []
        values: dict[str, int] = {}
        for v in self._variables:
            try:
                value = v.decode(snapshot)
            except ConfigurationError:
                continue  # the snapshot does not cover this variable
            values[v.id] = value
            parts.append((v.id, bucket(value, v.kind, self._bins)))
        for mover, mine in self._pairs:
            if mover.id in values and mine.id in values:
                apart = values[mover.id] - values[mine.id]
                parts.append(
                    (distance_id(mover.id, mine.id), bucket(apart, mover.kind, self._bins))
                )
        return State(tuple(parts))


def _pair_up(variables: tuple[Variable, ...]) -> tuple[tuple[Variable, Variable], ...]:
    """Each mover with each of the player's own positions of the same size and region.

    The best-believed own position's pairs come first, then the next's, so the
    ``MOST_DISTANCES`` kept are the ones most likely to matter.
    """
    movers = sorted((v for v in variables if is_mover(v)), key=belief_rank)
    mine = sorted(
        (v for v in variables if v.kind is VariableKind.POSITION and not is_mover(v)),
        key=belief_rank,
    )
    pairs = [
        (m, p) for p in mine for m in movers if m.region == p.region and m.encoding is p.encoding
    ]
    return tuple(pairs[:MOST_DISTANCES])


def _choose(believed: list[Variable], most: int | None) -> list[Variable]:
    """Up to ``most`` of the believed variables, kind by kind in rank order (T-908).

    Each kind's queue is in :func:`belief_rank` order; the kinds are visited
    in :data:`KIND_RANK` order (others after, in the enum's order) and one is
    taken from each queue per round, so every kind gets a voice and the
    first round holds the best position, lives, health, speed, score,
    progress value and flag.
    """
    if most is None or len(believed) <= most:
        return believed
    order = (*KIND_RANK, *(k for k in VariableKind if k not in KIND_RANK))
    queues: dict[VariableKind, list[Variable]] = {k: [] for k in order}
    for v in sorted(believed, key=belief_rank):
        queues[v.kind].append(v)
    chosen: list[Variable] = []
    while len(chosen) < most:
        took = False
        for kind in order:
            queue = queues[kind]
            if queue:
                chosen.append(queue.pop(0))
                took = True
                if len(chosen) == most:
                    break
        if not took:
            break
    return chosen


__all__ = [
    "DEFAULT_BINS",
    "DEFAULT_MOST",
    "DISTANCE_SEP",
    "IGNORED_KINDS",
    "KIND_RANK",
    "MOST_DISTANCES",
    "State",
    "StateEncoder",
    "bucket",
    "distance_id",
    "split_distance",
]
