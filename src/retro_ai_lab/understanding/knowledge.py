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

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot

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


class Encoding(Enum):
    U8 = "u8"
    S8 = "s8"
    U16LE = "u16le"
    U16BE = "u16be"
    BCD = "bcd"  # one decimal digit per nibble, little-endian bytes
    FLAG = "flag"  # 0 / non-zero

    @property
    def width(self) -> int:
        return 2 if self in (Encoding.U16LE, Encoding.U16BE) else 1


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
        if self.rejected:
            confidence = Confidence.UNKNOWN
        elif self.confirmed:
            confidence = Confidence.CONFIRMED
        else:
            confidence = confidence_from_evidence(e.weight for e in self.evidence)
        object.__setattr__(self, "confidence", confidence)

    @property
    def width(self) -> int:
        return self.encoding.width

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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad variable: {exc}") from exc


def decode(raw: bytes, encoding: Encoding) -> int:
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

    # ------------------------------------------------------------ access

    @property
    def version(self) -> int:
        """Goes up on every change to the variables or relationships (T-841).

        Anything that keeps a list derived from the knowledge (the event detector's
        watch list, the long look's) rebuilds it only when this moves.
        """
        return self._version

    def __len__(self) -> int:
        return len(self._variables)

    def __iter__(self) -> Iterator[Variable]:
        return iter(self._variables.values())

    def __contains__(self, var_id: str) -> bool:
        return var_id in self._variables

    def get(self, var_id: str) -> Variable:
        try:
            return self._variables[var_id]
        except KeyError:
            raise ConfigurationError(f"No variable {var_id!r} in the knowledge.") from None

    def at(self, region: str, offset: int) -> Variable | None:
        """The variable that starts at this location, if any."""
        for v in self._variables.values():
            if v.region == region and v.offset == offset:
                return v
        return None

    def by_kind(self, kind: VariableKind) -> tuple[Variable, ...]:
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
        return tuple(
            v for v in self._variables.values() if not v.rejected and at_least <= v.confidence
        )

    def needs_help(self) -> tuple[Variable, ...]:
        """Beliefs that are worth asking a person about (DESIGN.md §3.4)."""
        return tuple(
            v
            for v in self._variables.values()
            if not v.rejected
            and not v.confirmed
            and v.confidence in (Confidence.LOW, Confidence.MEDIUM)
        )

    # ----------------------------------------------------------- editing

    def add(self, variable: Variable) -> Variable:
        if variable.id in self._variables:
            raise ConfigurationError(f"Variable {variable.id!r} already exists.")
        clash = self.at(variable.region, variable.offset)
        if clash is not None and not clash.rejected:
            raise ConfigurationError(
                f"{variable.region}[{variable.offset}] is already {clash.id!r}; update it instead."
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
    ) -> Variable:
        """Add a new belief with a fresh id (what the discovery engine calls)."""
        return self.add(
            Variable(self.new_id(), region, offset, encoding, kind, name, tuple(evidence))
        )

    def update(self, var_id: str, **changes: Any) -> Variable:
        current = self.get(var_id)
        if "id" in changes:
            raise ConfigurationError("A variable's id can't change.")
        updated = replace(current, **changes)
        self._variables[var_id] = updated
        self._version += 1
        return updated

    def support(self, var_id: str, *evidence: Evidence) -> Variable:
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
        self.get(var_id)
        del self._variables[var_id]
        self._version += 1

    # ----------------------------------------------------- relationships

    def relationships(self) -> tuple[Relationship, ...]:
        return tuple(self._relationships.values())

    def relationship(self, rel_id: str) -> Relationship:
        try:
            return self._relationships[rel_id]
        except KeyError:
            raise ConfigurationError(f"No relationship {rel_id!r} in the knowledge.") from None

    def set_relationship(self, rel: Relationship) -> Relationship:
        """Add or replace the counts of a relationship, keeping a person's verdict."""
        current = self._relationships.get(rel.id)
        if current is not None and (current.confirmed or current.rejected):
            rel = replace(
                rel, confirmed=current.confirmed, rejected=current.rejected, notes=current.notes
            )
        self._relationships[rel.id] = rel
        self._version += 1
        return rel

    def confirm_relationship(self, rel_id: str) -> Relationship:
        rel = replace(self.relationship(rel_id), confirmed=True, rejected=False)
        self._relationships[rel_id] = rel
        self._version += 1
        return rel

    def reject_relationship(self, rel_id: str) -> Relationship:
        rel = replace(self.relationship(rel_id), rejected=True, confirmed=False)
        self._relationships[rel_id] = rel
        self._version += 1
        return rel

    def remove_relationship(self, rel_id: str) -> None:
        self.relationship(rel_id)
        del self._relationships[rel_id]
        self._version += 1

    def confident_relationships(
        self, at_least: Confidence = Confidence.LOW
    ) -> tuple[Relationship, ...]:
        return tuple(
            r for r in self._relationships.values() if not r.rejected and at_least <= r.confidence
        )

    # ------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "game_id": self.game_id,
            "title": self.title,
            "variables": [v.to_dict() for v in self._variables.values()],
            "relationships": [r.to_dict() for r in self._relationships.values()],
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
        except (KeyError, TypeError, AttributeError) as exc:
            raise ConfigurationError(f"Bad knowledge data: {exc}") from exc
        return k


__all__ = [
    "SCHEMA_VERSION",
    "THRESHOLDS",
    "Confidence",
    "Encoding",
    "Evidence",
    "Knowledge",
    "Relationship",
    "Variable",
    "VariableKind",
    "confidence_from_evidence",
    "decode",
]
