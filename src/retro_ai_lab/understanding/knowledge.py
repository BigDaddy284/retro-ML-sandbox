"""The knowledge model: what the app believes about a game, and how sure it is.

DESIGN.md §3.5 and §3.6 drive the shape of this module:

* every belief carries a :class:`Confidence`, and ``CONFIRMED`` can only be
  reached by a person confirming it (the system never pretends);
* beliefs carry their :class:`Evidence` so a child can see *why* the app
  thinks a byte is the score;
* knowledge is plain data that round-trips through JSON, so it can be saved,
  edited, exported and shared without any pre-built game profile.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from retro_ai_lab.core import ConfigurationError

if TYPE_CHECKING:
    from retro_ai_lab.observation.memory import MemorySnapshot

BELIEVED_PER_KIND = 16  # the most guesses of one kind the AI acts on at once (T-850)
OFFERED_PER_KIND = 1  # guesses of one kind offered as rewards and steps (T-858) ...
OFFERED_MANY = 3  # ... except progress values and switches: a level, a checkpoint and a lap
SCHEMA_VERSION = 2  # 2: relationships added (older files load fine)


class Confidence(Enum):
    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CONFIRMED = 4

    @property
    def label(self) -> str:
        return self.name.capitalize()

    def __lt__(self, other: Confidence) -> bool:
        return self.value < other.value

    def __le__(self, other: Confidence) -> bool:
        return self.value <= other.value


# Evidence weight needed to reach each automatic level. CONFIRMED is never
# reached automatically, whatever the weight.
THRESHOLDS: tuple[tuple[float, Confidence], ...] = (
    (6.0, Confidence.HIGH),
    (3.0, Confidence.MEDIUM),
    (1.0, Confidence.LOW),
)


def belief_rank(v: Variable) -> tuple[bool, int, float, str, int]:
    """Sort key for the guesses the AI acts on: the most trusted first (T-850, T-908).

    Confirmed first, then by confidence, then by evidence weight, then by
    address. :meth:`Knowledge.believed` seats guesses in this order and the
    state encoder fills the brain's slots in it.
    """
    return (
        not v.confirmed,
        -v.confidence.value,
        -sum(e.weight for e in v.evidence),
        v.region,
        v.offset,
    )


def confidence_from_evidence(weights: Iterable[float]) -> Confidence:
    """Map accumulated evidence weight to a confidence level (never CONFIRMED)."""
    total = 0.0
    for w in weights:
        if w < 0:
            raise ConfigurationError("Evidence weight must not be negative.")
        total += w
    for needed, level in THRESHOLDS:
        if total >= needed:
            return level
    return Confidence.UNKNOWN


MAX_DIGITS = 8  # the most decimal digits a value holds (a score of 99,999,999)
MAX_BINARY = 4  # the most bytes a binary or packed-BCD value holds (the same number)


class Encoding(Enum):
    U8 = "u8"
    S8 = "s8"
    U16LE = "u16le"
    U16BE = "u16be"
    BCD = "bcd"  # one decimal digit per nibble, one byte
    FLAG = "flag"  # 0 / non-zero
    # Sized layouts (T-941): the variable's own ``length`` says how many bytes.
    UINT_LE = "uint_le"  # binary, low byte first
    UINT_BE = "uint_be"  # binary, high byte first
    BCD_LE = "bcd_le"  # packed decimal, two digits a byte, lowest pair first
    BCD_BE = "bcd_be"  # packed decimal, highest pair first
    DIGITS_LE = "digits_le"  # one decimal digit per byte, lowest digit first
    DIGITS_BE = "digits_be"  # one decimal digit per byte, highest first (as it reads on screen)

    @property
    def sized(self) -> bool:
        """Whether the variable's ``length`` sets how many bytes this takes (T-941)."""
        return self in _SIZED

    @property
    def width(self) -> int:
        """Bytes taken; 0 for a sized layout, whose variable says."""
        if self.sized:
            return 0
        return 2 if self in (Encoding.U16LE, Encoding.U16BE) else 1

    @property
    def max_length(self) -> int:
        if self in (Encoding.DIGITS_LE, Encoding.DIGITS_BE):
            return MAX_DIGITS
        return MAX_BINARY if self.sized else self.width

    def describe(self, length: int) -> str:
        """How a value is held, in plain words: "1 byte", "6 digits", "2 bytes, low first"."""
        if self in (Encoding.DIGITS_LE, Encoding.DIGITS_BE):
            order = "highest first" if self is Encoding.DIGITS_BE else "lowest first"
            return f"{length} digit{'s' if length != 1 else ''}, {order}"
        if self in (Encoding.BCD, Encoding.BCD_LE, Encoding.BCD_BE):
            n = 1 if self is Encoding.BCD else length
            text = f"packed decimal, {n} byte{'s' if n != 1 else ''}"
            if n > 1:
                text += ", high first" if self is Encoding.BCD_BE else ", low first"
            return text
        n = self.width or length
        if n == 1:
            return "1 byte" if self is not Encoding.FLAG else "1 byte (on or off)"
        order = "high first" if self in (Encoding.U16BE, Encoding.UINT_BE) else "low first"
        return f"{n} bytes, {order}"


_SIZED = frozenset(
    {
        Encoding.UINT_LE,
        Encoding.UINT_BE,
        Encoding.BCD_LE,
        Encoding.BCD_BE,
        Encoding.DIGITS_LE,
        Encoding.DIGITS_BE,
    }
)


class VariableKind(Enum):
    UNKNOWN = "unknown"
    COUNTER = "counter"  # goes up as the game goes on (frame counter, timer)
    SCORE = "score"
    HEALTH = "health"
    LIVES = "lives"
    POSITION = "position"
    SPEED = "speed"
    PROGRESS = "progress"  # level, lap, checkpoint, quest step
    FLAG = "flag"  # game over, paused, in menu
    INPUT = "input"  # a mirror of the controller itself, not game state

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").capitalize()


@dataclass(frozen=True, slots=True)
class Evidence:
    """One observation that supports (or weakens) a belief."""

    kind: str  # short machine tag, e.g. "monotonic", "input-linked", "user"
    detail: str  # plain words for the UI
    weight: float = 1.0
    frame: int | None = None

    def __post_init__(self) -> None:
        if not self.kind or not isinstance(self.kind, str):
            raise ConfigurationError("Evidence needs a kind.")
        if self.weight < 0:
            raise ConfigurationError("Evidence weight must not be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "weight": self.weight,
            "frame": self.frame,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        try:
            return cls(
                data["kind"], data.get("detail", ""), data.get("weight", 1.0), data.get("frame")
            )
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(f"Bad evidence: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Variable:
    """A memory location the app believes means something."""

    id: str
    region: str
    offset: int
    encoding: Encoding = Encoding.U8
    kind: VariableKind = VariableKind.UNKNOWN
    name: str = ""
    evidence: tuple[Evidence, ...] = ()
    confirmed: bool = False  # set only by a person
    rejected: bool = False  # a person said "no, that's wrong"
    notes: str = ""
    judged: int = 0  # how many changes of the byte the classifier last judged it on (T-917)
    length: int = 0  # bytes taken by a sized encoding (T-941); 0 means the encoding's own width
    # Worked out once from the fields above (T-841): the detector, the long look and
    # the suggestion rules read it for every variable on every snapshot.
    confidence: Confidence = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ConfigurationError("Variable needs an id.")
        if not self.region:
            raise ConfigurationError("Variable needs a region.")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise ConfigurationError("Variable offset must be a non-negative integer.")
        if self.confirmed and self.rejected:
            raise ConfigurationError("A variable can't be both confirmed and rejected.")
        if isinstance(self.length, bool) or not isinstance(self.length, int) or self.length < 0:
            raise ConfigurationError("Variable length must be a non-negative integer.")
        if self.encoding.sized:
            if not 1 <= self.length <= self.encoding.max_length:
                raise ConfigurationError(
                    f"A {self.encoding.value} value takes 1 to {self.encoding.max_length} bytes, "
                    f"not {self.length}."
                )
        elif self.length not in (0, self.encoding.width):
            raise ConfigurationError(
                f"A {self.encoding.value} value takes {self.encoding.width} byte(s), "
                f"not {self.length}."
            )
        if self.rejected:
            confidence = Confidence.UNKNOWN
        elif self.confirmed:
            confidence = Confidence.CONFIRMED
        else:
            confidence = confidence_from_evidence(e.weight for e in self.evidence)
        object.__setattr__(self, "confidence", confidence)

    @property
    def width(self) -> int:
        return self.length if self.encoding.sized else self.encoding.width

    @property
    def address(self) -> str:
        """Where it is: ``ram[12]``, or ``ram[2013..2018]`` over several bytes (T-941)."""
        return address_text(self.region, self.offset, self.width)

    @property
    def held_as(self) -> str:
        """How the bytes hold the number, in plain words (T-941)."""
        return self.encoding.describe(self.width)

    @property
    def display_name(self) -> str:
        return self.name or f"{self.kind.label} at {self.region}[{self.offset}]"

    def decode(self, snapshot: MemorySnapshot) -> int:
        """Read this variable's value out of a snapshot."""
        raw = snapshot.get(self.region, self.offset, self.width)
        return decode(raw, self.encoding)

    def with_evidence(self, *items: Evidence) -> Variable:
        return replace(self, evidence=self.evidence + tuple(items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "region": self.region,
            "offset": self.offset,
            "encoding": self.encoding.value,
            "kind": self.kind.value,
            "name": self.name,
            "evidence": [e.to_dict() for e in self.evidence],
            "confirmed": self.confirmed,
            "rejected": self.rejected,
            "notes": self.notes,
            "judged": self.judged,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Variable:
        try:
            return cls(
                id=data["id"],
                region=data["region"],
                offset=data["offset"],
                encoding=Encoding(data.get("encoding", "u8")),
                kind=VariableKind(data.get("kind", "unknown")),
                name=data.get("name", ""),
                evidence=tuple(Evidence.from_dict(e) for e in data.get("evidence", [])),
                confirmed=bool(data.get("confirmed", False)),
                rejected=bool(data.get("rejected", False)),
                notes=data.get("notes", ""),
                judged=int(data.get("judged", 0)),
                length=int(data.get("length", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad variable: {exc}") from exc


def address_text(region: str, offset: int, width: int = 1, *, hex_form: bool = False) -> str:
    """``ram[12]`` or ``ram[2013..2018]``; in hex, ``0x000C`` or ``0x07DD..0x07E2`` (T-941)."""
    last = offset + max(1, width) - 1
    if hex_form:
        return f"0x{offset:04X}" if last == offset else f"0x{offset:04X}..0x{last:04X}"
    inside = f"{offset}" if last == offset else f"{offset}..{last}"
    return f"{region}[{inside}]"


def decode(raw: bytes, encoding: Encoding) -> int:
    if encoding.sized:
        if not 1 <= len(raw) <= encoding.max_length:
            raise ConfigurationError(
                f"{encoding.value} takes 1 to {encoding.max_length} bytes, got {len(raw)}."
            )
        return _decode_sized(raw, encoding)
    if len(raw) != encoding.width:
        raise ConfigurationError(
            f"{encoding.value} needs {encoding.width} byte(s), got {len(raw)}."
        )
    if encoding is Encoding.U8:
        return raw[0]
    if encoding is Encoding.S8:
        return raw[0] - 256 if raw[0] > 127 else raw[0]
    if encoding is Encoding.U16LE:
        return raw[0] | (raw[1] << 8)
    if encoding is Encoding.U16BE:
        return (raw[0] << 8) | raw[1]
    if encoding is Encoding.BCD:
        hi, lo = raw[0] >> 4, raw[0] & 0xF
        if hi > 9 or lo > 9:
            raise ConfigurationError(f"{raw[0]:#04x} is not a BCD byte.")
        return hi * 10 + lo
    return 1 if raw[0] else 0


def _decode_sized(raw: bytes, encoding: Encoding) -> int:
    """Read a value over several bytes (T-941); highest part first after the turn."""
    high_first = (
        raw if encoding in (Encoding.UINT_BE, Encoding.BCD_BE, Encoding.DIGITS_BE) else raw[::-1]
    )
    if encoding in (Encoding.UINT_LE, Encoding.UINT_BE):
        return int.from_bytes(high_first, "big")
    value = 0
    if encoding in (Encoding.BCD_LE, Encoding.BCD_BE):
        for b in high_first:
            hi, lo = b >> 4, b & 0xF
            if hi > 9 or lo > 9:
                raise ConfigurationError(f"{b:#04x} is not a BCD byte.")
            value = value * 100 + hi * 10 + lo
        return value
    for b in high_first:
        if b > 9:
            raise ConfigurationError(f"{b} is not a decimal digit.")
        value = value * 10 + b
    return value


@dataclass(frozen=True, slots=True)
class Relationship:
    """``cause`` tends to be followed by ``effect`` within a few frames.

    Keys are ``"<event kind>:<variable id or *>"`` strings so the model does
    not depend on the event module. ``support`` counts the times the pair
    was seen; ``cause_count`` / ``effect_count`` how often each side
    happened at all, so the rates say how reliable the link is.
    """

    cause: str
    effect: str
    support: int
    cause_count: int
    effect_count: int
    window: int
    confirmed: bool = False
    rejected: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.cause or not self.effect or self.cause == self.effect:
            raise ConfigurationError("A relationship needs two different event keys.")
        if min(self.support, self.cause_count, self.effect_count) < 0 or self.window < 0:
            raise ConfigurationError("Relationship counts must not be negative.")
        if self.support > min(self.cause_count, self.effect_count):
            raise ConfigurationError("support can't exceed either side's count.")
        if self.confirmed and self.rejected:
            raise ConfigurationError("A relationship can't be both confirmed and rejected.")

    @property
    def id(self) -> str:
        return f"{self.cause}->{self.effect}"

    @property
    def follow_rate(self) -> float:
        """How often the cause was followed by the effect."""
        return self.support / self.cause_count if self.cause_count else 0.0

    @property
    def precede_rate(self) -> float:
        """How often the effect was preceded by the cause."""
        return self.support / self.effect_count if self.effect_count else 0.0

    @property
    def confidence(self) -> Confidence:
        if self.rejected:
            return Confidence.UNKNOWN
        if self.confirmed:
            return Confidence.CONFIRMED
        rate = min(self.follow_rate, self.precede_rate)
        if self.support >= 10 and rate >= 0.9:
            return Confidence.HIGH
        if self.support >= 5 and rate >= 0.75:
            return Confidence.MEDIUM
        if self.support >= 2 and rate >= 0.6:
            return Confidence.LOW
        return Confidence.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "support": self.support,
            "cause_count": self.cause_count,
            "effect_count": self.effect_count,
            "window": self.window,
            "confirmed": self.confirmed,
            "rejected": self.rejected,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relationship:
        try:
            return cls(
                cause=data["cause"],
                effect=data["effect"],
                support=int(data["support"]),
                cause_count=int(data["cause_count"]),
                effect_count=int(data["effect_count"]),
                window=int(data.get("window", 0)),
                confirmed=bool(data.get("confirmed", False)),
                rejected=bool(data.get("rejected", False)),
                notes=data.get("notes", ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad relationship: {exc}") from exc


OFFERED_MANY_KINDS = frozenset({VariableKind.PROGRESS, VariableKind.FLAG})
# Evidence tags other modules read back (T-911): a position found by its own movement
# (an enemy, a platform) is "something moving" until a button is shown to move it.
INPUT_LINKED = "input-linked"
MOVER = "moves-both-ways"
MOVING_WORD = "Something moving"


def is_mover(v: Variable) -> bool:
    """A position believed from its movement alone, not from the controller (T-911)."""
    kinds = {e.kind for e in v.evidence}
    return v.kind is VariableKind.POSITION and MOVER in kinds and INPUT_LINKED not in kinds


AI = "the AI"  # who a guess the AI drops itself is "rejected by" (T-864, T-916)


def rejected_by_ai(v: Variable) -> bool:
    """Dropped by the AI's own doubts, not by a person; such a guess can be rethought (T-916)."""
    return v.rejected and any(
        e.kind == "user" and e.detail == f"Rejected by {AI}." for e in v.evidence
    )


ROLE_WORDS: dict[VariableKind, str] = {
    VariableKind.SCORE: "Your score",
    VariableKind.LIVES: "a life",
    VariableKind.HEALTH: "Health",
    VariableKind.PROGRESS: "Progress",
    VariableKind.POSITION: "Your position",
    VariableKind.SPEED: "Speed",
    VariableKind.COUNTER: "Timer",
    VariableKind.FLAG: "Something",
    VariableKind.INPUT: "The controller",
    VariableKind.UNKNOWN: "Something",
}
_ORDINALS = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th")


def plain_names(variables: Iterable[Variable]) -> dict[str, str]:
    """Names a kid can read, by variable id (T-858).

    A name a person typed is used as typed. An unnamed guess is called by its role
    ("Your score", "a life", "Progress"); when several unnamed guesses of one kind
    are given, the later ones say which guess they are ("Progress (2nd guess)").
    The order of ``variables`` is the order of the seats.
    """
    out: dict[str, str] = {}
    seen: dict[tuple[VariableKind, bool], int] = {}
    for v in variables:
        if v.name:
            out[v.id] = v.name
            continue
        mover = is_mover(v)
        seat = (v.kind, mover)
        n = seen.get(seat, 0) + 1
        seen[seat] = n
        word = MOVING_WORD if mover else ROLE_WORDS.get(v.kind, "Something")
        ordinal = _ORDINALS[n - 1] if n <= len(_ORDINALS) else f"{n}th"
        out[v.id] = word if n == 1 else f"{word} ({ordinal} guess)"
    return out


class Knowledge:
    """Everything learned about one game. Mutable; persisted as JSON."""

    def __init__(self, game_id: str, *, title: str = "") -> None:
        if not game_id or not isinstance(game_id, str):
            raise ConfigurationError("Knowledge needs a game id.")
        self.game_id = game_id
        self.title = title
        self._variables: dict[str, Variable] = {}
        self._relationships: dict[str, Relationship] = {}
        self._version = 0
        self._confident_version = 0  # a confident pair appeared, went or changed level (T-948)
        # Snapshots are observed on the play clock's worker and on the GUI thread
        # (T-818), so the finder may rewrite relationships while a handler on the
        # other thread walks them (T-847). Every walk and every edit takes this.
        self._lock = threading.RLock()
        # The button the AI found gets this game going when it stops listening
        # (a title screen, game over); worked out by experiment, never assumed (T-844).
        self.wake_button: str | None = None

    # ------------------------------------------------------------ access

    @property
    def version(self) -> int:
        """Goes up on every change to the variables (T-841).

        Anything that keeps a list derived from the variables (the event detector's
        watch list, the long look's) rebuilds it only when this moves. Relationship
        changes don't count: the finder rewrites pairs' rates on most events;
        :attr:`confident_relationships_version` is the number for those.
        """
        return self._version

    @property
    def confident_relationships_version(self) -> int:
        """Goes up when the confident relationships change (T-948).

        That is when a relationship becomes at least LOW or stops being, changes
        level, is confirmed, rejected or removed. A write that only moves a
        pair's counts within its level leaves it alone, so the suggestions can
        ask "did anything I read change?" in one comparison instead of walking
        every relationship twice a step, which cost the user 56 ms of each step.
        """
        return self._confident_version

    def __len__(self) -> int:
        return len(self._variables)

    def __iter__(self) -> Iterator[Variable]:
        with self._lock:
            return iter(tuple(self._variables.values()))

    def __contains__(self, var_id: str) -> bool:
        return var_id in self._variables

    def get(self, var_id: str) -> Variable:
        try:
            return self._variables[var_id]
        except KeyError:
            raise ConfigurationError(f"No variable {var_id!r} in the knowledge.") from None

    def at(self, region: str, offset: int) -> Variable | None:
        """The variable that starts at this location, if any."""
        with self._lock:
            for v in self._variables.values():
                if v.region == region and v.offset == offset:
                    return v
        return None

    def by_kind(self, kind: VariableKind) -> tuple[Variable, ...]:
        with self._lock:
            return tuple(v for v in self._variables.values() if v.kind is kind and not v.rejected)

    def best(self, kind: VariableKind) -> Variable | None:
        """The most trusted non-rejected variable of a kind."""
        candidates = self.by_kind(kind)
        if not candidates:
            return None
        return max(
            candidates, key=lambda v: (v.confidence.value, sum(e.weight for e in v.evidence))
        )

    def confident(self, at_least: Confidence = Confidence.MEDIUM) -> tuple[Variable, ...]:
        with self._lock:
            return tuple(
                v for v in self._variables.values() if not v.rejected and at_least <= v.confidence
            )

    def believed(
        self,
        at_least: Confidence = Confidence.LOW,
        *,
        regions: Iterable[str] | None = None,
        per_kind: int = BELIEVED_PER_KIND,
    ) -> tuple[Variable, ...]:
        """The guesses the AI acts on: the strongest few of each kind (T-850).

        A game has one lives counter, a score, a handful of progress values and a
        few flags. On the user's Super Mario Bros. the knowledge held 1,600 guesses,
        nearly all wrong, and watching, rewarding, pairing and learning from all of
        them cost most of every step and left the agent never seeing the same state
        twice. Every guess stays in the knowledge; this is the subset that the
        detector, the long look, the rewards, the goals and the state read.
        Seats go to confirmed variables first, then by confidence, then by evidence
        weight, then by address; with ``regions`` only variables in those regions
        compete (memory the emulator no longer shares must not take a seat). The
        result keeps the knowledge's own order.
        """
        if per_kind < 1:
            raise ConfigurationError("per_kind must be at least 1.")
        wanted = None if regions is None else frozenset(regions)
        with self._lock:
            candidates = [
                (i, v)
                for i, v in enumerate(self._variables.values())
                if not v.rejected
                and at_least <= v.confidence
                and (wanted is None or v.region in wanted)
            ]
        candidates.sort(key=lambda item: belief_rank(item[1]))
        # A moving thing is a role of its own (T-928): on the user's game sixteen plain
        # positions outranked every enemy sprite, and nothing moving was ever believed.
        seats: dict[tuple[VariableKind, bool], int] = {}
        chosen: list[tuple[int, Variable]] = []
        for item in candidates:
            role = (item[1].kind, is_mover(item[1]))
            taken = seats.get(role, 0)
            if taken < per_kind:
                seats[role] = taken + 1
                chosen.append(item)
        chosen.sort()
        return tuple(v for _, v in chosen)

    def offered(self, at_least: Confidence = Confidence.LOW) -> tuple[Variable, ...]:
        """The best guess of each role, the ones rewards and steps are offered for (T-858).

        A kid reading the Rewards panel should see one score, one lives counter and
        a few progress values in plain words, not seven look-alike rows by address.
        Seats are given as :meth:`believed` gives them (confirmed first, then by
        confidence and evidence), one per kind, three for progress and flags.
        """
        few = self.believed(at_least, per_kind=OFFERED_MANY)
        one = {v.id for v in self.believed(at_least, per_kind=OFFERED_PER_KIND)}
        # Nobody puts a reward on "Something moving" (T-928): the offers are the roles.
        return tuple(
            v for v in few if (v.kind in OFFERED_MANY_KINDS or v.id in one) and not is_mover(v)
        )

    def needs_help(self) -> tuple[Variable, ...]:
        """Beliefs that are worth asking a person about (DESIGN.md §3.4)."""
        with self._lock:
            return tuple(
                v
                for v in self._variables.values()
                if not v.rejected
                and not v.confirmed
                and v.confidence in (Confidence.LOW, Confidence.MEDIUM)
            )

    # ----------------------------------------------------------- editing

    def add(self, variable: Variable) -> Variable:
        with self._lock:
            if variable.id in self._variables:
                raise ConfigurationError(f"Variable {variable.id!r} already exists.")
            clash = self.at(variable.region, variable.offset)
            if clash is not None and not clash.rejected:
                raise ConfigurationError(
                    f"{variable.region}[{variable.offset}] is already {clash.id!r}; "
                    "update it instead."
                )
            self._variables[variable.id] = variable
            self._version += 1
        return variable

    def new_id(self) -> str:
        """The next free ``vN`` id (nothing is reserved until ``add`` succeeds)."""
        n = 1
        while f"v{n}" in self._variables:
            n += 1
        return f"v{n}"

    def propose(
        self,
        region: str,
        offset: int,
        *,
        encoding: Encoding = Encoding.U8,
        kind: VariableKind = VariableKind.UNKNOWN,
        name: str = "",
        evidence: Iterable[Evidence] = (),
        judged: int = 0,
        length: int = 0,
    ) -> Variable:
        """Add a new belief with a fresh id (what the discovery engine calls)."""
        return self.add(
            Variable(
                self.new_id(),
                region,
                offset,
                encoding,
                kind,
                name,
                tuple(evidence),
                judged=judged,
                length=length,
            )
        )

    def update(self, var_id: str, **changes: Any) -> Variable:
        with self._lock:
            current = self.get(var_id)
            if "id" in changes:
                raise ConfigurationError("A variable's id can't change.")
            updated = replace(current, **changes)
            self._variables[var_id] = updated
            self._version += 1
        return updated

    def forget_judged(self) -> int:
        """Set every guess's ``judged`` count to zero; returns how many had one.

        The count says how many changes of its byte a guess was last judged on
        (T-917), and those changes are a run's: the tracker starts from zero at
        every connection, so a count saved from an earlier run could never be
        reached again and nothing was rethought (T-926).
        """
        with self._lock:
            stale = [v for v in self._variables.values() if v.judged]
            for v in stale:
                self._variables[v.id] = replace(v, judged=0)
            if stale:
                self._version += 1
            return len(stale)

    def support(self, var_id: str, *evidence: Evidence) -> Variable:
        with self._lock:
            return self.update(var_id, evidence=self.get(var_id).evidence + tuple(evidence))

    def confirm(self, var_id: str, *, by: str = "user") -> Variable:
        """A person says the belief is right."""
        proof = Evidence("user", f"Confirmed by {by}.", weight=0.0)
        current = self.get(var_id)
        return self.update(
            var_id, confirmed=True, rejected=False, evidence=(*current.evidence, proof)
        )

    def correct(
        self,
        var_id: str,
        *,
        name: str | None = None,
        kind: VariableKind | None = None,
        encoding: Encoding | None = None,
        length: int | None = None,
        by: str = "user",
    ) -> Variable:
        """A person fixes what the belief means; the fix counts as confirmation."""
        changes: dict[str, Any] = {}
        if name is not None:
            changes["name"] = name
        if kind is not None:
            changes["kind"] = kind
        if encoding is not None:
            changes["encoding"] = encoding
        if length is not None:
            changes["length"] = length
        if not changes:
            raise ConfigurationError("correct() needs something to change.")
        self.update(var_id, **changes)
        return self.confirm(var_id, by=by)

    def reject(self, var_id: str, *, by: str = "user") -> Variable:
        """A person says the belief is wrong; it stays as a record so it is not re-proposed."""
        proof = Evidence("user", f"Rejected by {by}.", weight=0.0)
        current = self.get(var_id)
        return self.update(
            var_id, rejected=True, confirmed=False, evidence=(*current.evidence, proof)
        )

    def remove(self, var_id: str) -> None:
        with self._lock:
            self.get(var_id)
            del self._variables[var_id]
            self._version += 1

    # ----------------------------------------------------- relationships

    def relationships(self) -> tuple[Relationship, ...]:
        with self._lock:
            return tuple(self._relationships.values())

    def relationship(self, rel_id: str) -> Relationship:
        try:
            return self._relationships[rel_id]
        except KeyError:
            raise ConfigurationError(f"No relationship {rel_id!r} in the knowledge.") from None

    def set_relationship(self, rel: Relationship) -> Relationship:
        """Add or replace the counts of a relationship, keeping a person's verdict."""
        with self._lock:
            current = self._relationships.get(rel.id)
            if current is not None and (current.confirmed or current.rejected):
                rel = replace(
                    rel, confirmed=current.confirmed, rejected=current.rejected, notes=current.notes
                )
            self._relationships[rel.id] = rel
            self._note_confident(current, rel)
        return rel

    def confirm_relationship(self, rel_id: str) -> Relationship:
        with self._lock:
            current = self.relationship(rel_id)
            rel = replace(current, confirmed=True, rejected=False)
            self._relationships[rel_id] = rel
            self._note_confident(current, rel)
        return rel

    def reject_relationship(self, rel_id: str) -> Relationship:
        with self._lock:
            current = self.relationship(rel_id)
            rel = replace(current, rejected=True, confirmed=False)
            self._relationships[rel_id] = rel
            self._note_confident(current, rel)
        return rel

    def remove_relationship(self, rel_id: str) -> None:
        with self._lock:
            current = self.relationship(rel_id)
            del self._relationships[rel_id]
            self._note_confident(current, None)

    @staticmethod
    def _confident_level(rel: Relationship | None) -> Confidence | None:
        """The level the suggestions read, or None when they would not see the pair."""
        if rel is None or rel.rejected:
            return None
        level = rel.confidence
        return level if level >= Confidence.LOW else None

    def _note_confident(self, before: Relationship | None, after: Relationship | None) -> None:
        if self._confident_level(before) != self._confident_level(after):
            self._confident_version += 1

    def confident_relationships(
        self, at_least: Confidence = Confidence.LOW
    ) -> tuple[Relationship, ...]:
        with self._lock:
            return tuple(
                r
                for r in self._relationships.values()
                if not r.rejected and at_least <= r.confidence
            )

    # ------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": SCHEMA_VERSION,
                "game_id": self.game_id,
                "title": self.title,
                "variables": [v.to_dict() for v in self._variables.values()],
                "relationships": [r.to_dict() for r in self._relationships.values()],
                "wake_button": self.wake_button,
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Knowledge:
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(
                    f"This knowledge file is from a newer version (schema {schema})."
                )
            k = cls(data["game_id"], title=data.get("title", ""))
            for item in data.get("variables", []):
                k.add(Variable.from_dict(item))
            for item in data.get("relationships", []):
                k.set_relationship(Relationship.from_dict(item))
            wake = data.get("wake_button")
            if wake is not None and (not isinstance(wake, str) or not wake):
                raise ConfigurationError("wake_button must be a button name or null.")
            k.wake_button = wake
        except (KeyError, TypeError, AttributeError) as exc:
            raise ConfigurationError(f"Bad knowledge data: {exc}") from exc
        return k


__all__ = [
    "AI",
    "BELIEVED_PER_KIND",
    "INPUT_LINKED",
    "MAX_BINARY",
    "MAX_DIGITS",
    "MOVER",
    "MOVING_WORD",
    "OFFERED_MANY",
    "OFFERED_PER_KIND",
    "ROLE_WORDS",
    "SCHEMA_VERSION",
    "THRESHOLDS",
    "Confidence",
    "Encoding",
    "Evidence",
    "Knowledge",
    "Relationship",
    "Variable",
    "VariableKind",
    "address_text",
    "belief_rank",
    "confidence_from_evidence",
    "decode",
    "is_mover",
    "plain_names",
    "rejected_by_ai",
]
