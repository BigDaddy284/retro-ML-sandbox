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

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot
from retro_ai_lab.understanding.knowledge import (
    INPUT_LINKED,
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
SEEN_WORDS: dict[VariableKind, tuple[str, str]] = {  # how describe() counts each kind (T-913)
    VariableKind.POSITION: ("position", "positions"),
    VariableKind.LIVES: ("life count", "life counts"),
    VariableKind.HEALTH: ("health value", "health values"),
    VariableKind.SPEED: ("speed", "speeds"),
    VariableKind.SCORE: ("score", "scores"),
    VariableKind.PROGRESS: ("progress value", "progress values"),
    VariableKind.FLAG: ("switch", "switches"),
}
DISTANCE_SEP = "~"  # a distance part's id is "<mover id>~<my position's id>"
SCREEN_PREFIX = "screen:"  # parts the AI reads off the screen, not memory (T-978)
SCREEN_SEATS = 4  # seats kept for them when the AI looks at the screen
SCREEN_WORDS = {
    "screen:mover dx": "nearest moving thing across",
    "screen:mover dy": "nearest moving thing up/down",
    "screen:hurts dx": "nearest thing that hurts across",
    "screen:hurts dy": "nearest thing that hurts up/down",
}
FINE_STEP = 8  # values per step where the AI looks closer (T-979)
FINE_KEY = 1 << 32  # a fine bucket is coarse * FINE_KEY + (value - the bucket's low) // FINE_STEP
MOST_CLOSER = 8  # sights the encoder looks closer at, at most
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


def fine_bucket(value: int, kind: VariableKind, bins: int = DEFAULT_BINS) -> int:
    """A coarse bucket split into steps of ``FINE_STEP`` (T-979): the user's AI died at
    one stretch of ground that shared its 128-wide bucket with the safe ground before it.
    ``coarse * FINE_KEY + (value - low) // FINE_STEP`` with ``low`` the bucket's first
    value, negatives mirrored as :func:`bucket` does."""
    if value < 0:
        return -1 - fine_bucket(-value, kind, bins)
    coarse = bucket(value, kind, bins)
    return coarse * FINE_KEY + (value - bucket_low(coarse, bins)) // FINE_STEP


def is_fine(b: int) -> bool:
    """True for a bucket :func:`fine_bucket` made."""
    return abs(b) >= FINE_KEY


def coarse_bucket(b: int) -> int:
    """The coarse bucket a fine one was split from; a coarse bucket is itself."""
    if not is_fine(b):
        return b
    if b < 0:
        return -1 - ((-1 - b) // FINE_KEY)
    return b // FINE_KEY


def bucket_low(b: int, bins: int = DEFAULT_BINS) -> int:
    """The first (lowest magnitude) value of a coarse bucket of a value ``>= 0``."""
    if b < 0 or is_fine(b):
        raise ConfigurationError("bucket_low takes a coarse bucket of a value >= 0.")
    return b if b < bins else bins << (b - bins)


def bucket_span(b: int, bins: int = DEFAULT_BINS) -> int:
    """How many values a coarse bucket holds (1 below ``bins``, then doubling)."""
    if is_fine(b):
        raise ConfigurationError("bucket_span takes a coarse bucket.")
    coarse = -1 - b if b < 0 else b
    return 1 if coarse < bins else bins << (coarse - bins)


def bucket_word(b: int) -> str:
    """A bucket in words: the number, or "12 (128..135)" for a fine one (T-979)."""
    if not is_fine(b):
        return str(b)
    flip = b < 0
    if flip:
        b = -1 - b
    coarse, sub = divmod(b, FINE_KEY)
    low = bucket_low(coarse) + sub * FINE_STEP
    high = low + FINE_STEP - 1
    if flip:
        return f"-{coarse + 1} (-{high}..-{low})"
    return f"{coarse} ({low}..{high})"


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

    @property
    def is_fine(self) -> bool:
        """True when any part is read in fine steps (T-979)."""
        return any(is_fine(b) for _, b in self.parts)

    def coarse(self) -> State:
        """The same situation with every fine bucket read coarsely again (T-982): what
        the brain knew of the whole bucket before it looked closer."""
        return State(tuple((vid, coarse_bucket(b)) for vid, b in self.parts))

    def describe(self, knowledge: Knowledge | None = None, *, most: int | None = None) -> str:
        """The situation in words; ``most`` caps the values named ("and N more", T-889)."""
        if not self.parts:
            return "nothing known yet"
        if most is not None and most < 1:
            raise ConfigurationError("most must be at least 1 (or None).")
        words = []
        for vid, b in self.parts:
            words.append(f"{part_name(vid, knowledge)}={bucket_word(b)}")
        if most is not None and len(words) > most:
            left = len(words) - most
            return ", ".join(words[:most]) + f" and {left} more"
        return ", ".join(words)


def part_name(vid: str, knowledge: Knowledge | None) -> str:
    """A state part in words: a variable's display name, "<mover> from <mine>", or
    what the AI reads off the screen (T-978)."""
    if vid.startswith(SCREEN_PREFIX):
        return SCREEN_WORDS.get(vid, vid[len(SCREEN_PREFIX) :])
    pair = split_distance(vid)
    if pair is not None:
        mover, mine = (part_name(x, knowledge) for x in pair)
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
        self._screen: Callable[[], tuple[tuple[str, int], ...]] | None = None
        self._closer: dict[str, set[int]] = {}  # part id -> coarse buckets read finely
        self.refresh()

    def set_screen(self, source: Callable[[], tuple[tuple[str, int], ...]] | None) -> None:
        """Add what the AI reads off the screen to every state (T-978); None stops it.

        ``source`` gives ``(id, bucket)`` parts whose ids start with ``screen:``.
        At most ``SCREEN_SEATS`` of them are added, on top of the memory seats.
        """
        changed = (source is None) != (self._screen is None)
        self._screen = source
        if changed:
            self.refresh()

    @property
    def sees_screen(self) -> bool:
        return self._screen is not None

    @property
    def knowledge(self) -> Knowledge:
        """The knowledge the state is read from (T-940)."""
        return self._knowledge

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

    @property
    def closer(self) -> tuple[tuple[str, int], ...]:
        """The (part id, coarse bucket) sights read in fine steps (T-979)."""
        return tuple(sorted((vid, b) for vid, bs in self._closer.items() for b in bs))

    def look_closer(self, sights: Iterable[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
        """From now on read each part in fine steps where its value falls in the given
        coarse bucket (T-979). Exact buckets (within ``bins``), fine ones and sights past
        ``MOST_CLOSER`` are left alone. Returns the sights newly taken."""
        taken: list[tuple[str, int]] = []
        for vid, b in sights:
            if is_fine(b) or bucket_span(b, self._bins) <= FINE_STEP:
                continue
            if b in self._closer.get(vid, ()):
                continue
            if sum(len(bs) for bs in self._closer.values()) >= MOST_CLOSER:
                break
            self._closer.setdefault(vid, set()).add(b)
            taken.append((vid, b))
        return tuple(taken)

    def forget_closer(self) -> None:
        """Read every part coarsely again (a brain forgotten or swapped, T-981)."""
        self._closer.clear()

    def _bucket(self, vid: str, value: int, kind: VariableKind) -> int:
        b = bucket(value, kind, self._bins)
        if b in self._closer.get(vid, ()):
            return fine_bucket(value, kind, self._bins)
        return b

    def describe(self) -> str:
        """What the brain sees, in words, for the log (T-913)."""
        n = len(self._variables)
        if not n:
            return "nothing yet"
        counts = Counter(v.kind for v in self._variables)
        order = (*KIND_RANK, *(k for k in VariableKind if k not in KIND_RANK))
        words: list[str] = []
        for kind in order:
            c = counts.get(kind, 0)
            if not c:
                continue
            word = _plural(c, *SEEN_WORDS.get(kind, (kind.value, kind.value + "s")))
            if kind is VariableKind.POSITION:
                moving = {v.id for v in self._variables if is_mover(v)}
                moving.update(m.id for m, _ in self._pairs)  # paired without a seat (T-923)
                if moving:
                    word += f" ({len(moving)} moving)"
                words.append(word)
                if self._pairs:
                    d = len(self._pairs)
                    mine = self._pairs[0][1]  # the best-believed own position (T-929)
                    whose = " which the stick moves" if _input_linked(mine) else ""
                    words.append(
                        _plural(d, "distance to a moving thing", "distances to moving things")
                        + f" from {mine.display_name}{whose}"
                    )
                continue
            words.append(word)
        seen = f"{_plural(n, 'value', 'values')}: {', '.join(words)}"
        if self._closer:
            c = sum(len(bs) for bs in self._closer.values())
            seen += f"; looking closer at {_plural(c, 'sight', 'sights')} where it keeps dying"
        if self._screen is not None:
            seen += "; and off the screen, how far the nearest moving thing is"
        return seen

    def refresh(self) -> tuple[Variable, ...]:
        """Re-read the knowledge; call after discoveries change."""
        believed = [
            v for v in self._knowledge.believed(self._at_least) if v.kind not in IGNORED_KINDS
        ]
        most = self._most
        if most is not None and any(is_mover(v) for v in believed):
            most = max(1, most - MOST_DISTANCES)
        self._variables = tuple(sorted(_choose(believed, most), key=lambda v: (v.region, v.offset)))
        # The pairs come from everything believed, not the seats (T-923): a mover's
        # evidence (2.0) loses a position seat to any climbing or played position, and
        # the distance seats are kept for movers whether or not they hold a seat.
        self._pairs = _pair_up(tuple(believed))
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
            parts.append((v.id, self._bucket(v.id, value, v.kind)))
        for pair in self._pairs:
            for v in pair:  # a mover or own position without a seat is still read (T-923)
                if v.id not in values:
                    try:
                        values[v.id] = v.decode(snapshot)
                    except ConfigurationError:
                        continue
        for mover, mine in self._pairs:
            if mover.id in values and mine.id in values:
                apart = values[mover.id] - values[mine.id]
                did = distance_id(mover.id, mine.id)
                parts.append((did, self._bucket(did, apart, mover.kind)))
        if self._screen is not None:
            parts.extend(self._screen()[:SCREEN_SEATS])
        return State(tuple(parts))


def _input_linked(v: Variable) -> bool:
    return any(e.kind == INPUT_LINKED for e in v.evidence)


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _pair_up(variables: tuple[Variable, ...]) -> tuple[tuple[Variable, Variable], ...]:
    """Each mover with each of the player's own positions of the same size and region.

    Own means confirmed by a person or moved by the stick (``INPUT_LINKED``), when
    any such position is believed; otherwise any position that is not a mover.
    ``variables`` is everything believed, seated or not (T-923). Own positions
    the same buttons move count once (T-931).

    The best-believed own position's pairs come first, then the next's, so the
    ``MOST_DISTANCES`` kept are the ones most likely to matter.
    """
    movers = sorted((v for v in variables if is_mover(v)), key=belief_rank)
    own = [v for v in variables if v.kind is VariableKind.POSITION and not is_mover(v)]
    # The player's position is the one the stick moves, or one a person confirmed; a
    # scrolling position outranks it on evidence but is nobody's (T-915). Without any
    # such position, every position that is not a mover will do, as before.
    surely = [v for v in own if v.confirmed or _input_linked(v)]
    mine = sorted(surely or own, key=belief_rank)
    # Own positions the same buttons move are one position drawn several times (T-931:
    # the user's four sprite columns); the best-believed one stands for them, and an own
    # position other buttons move (the other axis) is kept beside it.
    seen: set[frozenset[str]] = set()
    distinct: list[Variable] = []
    for p in mine:
        links = frozenset(e.detail for e in p.evidence if e.kind == INPUT_LINKED)
        if links and links in seen:
            continue
        seen.add(links)
        distinct.append(p)
    pairs = [
        (m, p)
        for p in distinct
        for m in movers
        if m.region == p.region and m.encoding is p.encoding
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
    "FINE_KEY",
    "FINE_STEP",
    "IGNORED_KINDS",
    "KIND_RANK",
    "MOST_CLOSER",
    "MOST_DISTANCES",
    "SEEN_WORDS",
    "State",
    "StateEncoder",
    "bucket",
    "bucket_low",
    "bucket_span",
    "bucket_word",
    "coarse_bucket",
    "distance_id",
    "fine_bucket",
    "is_fine",
    "part_name",
    "split_distance",
]
