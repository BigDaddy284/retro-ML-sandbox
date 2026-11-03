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
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot
from retro_ai_lab.understanding.knowledge import Confidence, Knowledge, Variable, VariableKind

DEFAULT_BINS = 8
IGNORED_KINDS = frozenset({VariableKind.UNKNOWN, VariableKind.INPUT, VariableKind.COUNTER})


def bucket(value: int, kind: VariableKind, bins: int = DEFAULT_BINS) -> int:
    """The bin a value falls in (see the module docstring)."""
    if kind is VariableKind.FLAG:
        return 1 if value else 0
    if value < 0:
        return -1 - bucket(-value, kind, bins)
    if value < bins:
        return value
    return bins + (value // bins).bit_length() - 1


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

    def describe(self, knowledge: Knowledge | None = None) -> str:
        if not self.parts:
            return "nothing known yet"
        words = []
        for vid, b in self.parts:
            name = knowledge.get(vid).display_name if knowledge and vid in knowledge else vid
            words.append(f"{name}={b}")
        return ", ".join(words)


class StateEncoder:
    def __init__(
        self,
        knowledge: Knowledge,
        *,
        at_least: Confidence = Confidence.LOW,
        bins: int = DEFAULT_BINS,
    ) -> None:
        if not 2 <= bins <= 64:
            raise ConfigurationError("bins must be between 2 and 64.")
        self._knowledge = knowledge
        self._at_least = at_least
        self._bins = bins
        self._variables: tuple[Variable, ...] = ()
        self.refresh()

    @property
    def bins(self) -> int:
        return self._bins

    @property
    def variables(self) -> tuple[Variable, ...]:
        """The believed variables the state is built from, lowest address first."""
        return self._variables

    def refresh(self) -> tuple[Variable, ...]:
        """Re-read the knowledge; call after discoveries change."""
        self._variables = tuple(
            sorted(
                (
                    v
                    for v in self._knowledge
                    if not v.rejected
                    and v.confidence >= self._at_least
                    and v.kind not in IGNORED_KINDS
                ),
                key=lambda v: (v.region, v.offset),
            )
        )
        return self._variables

    def encode(self, snapshot: MemorySnapshot) -> State:
        parts: list[tuple[str, int]] = []
        for v in self._variables:
            try:
                value = v.decode(snapshot)
            except ConfigurationError:
                continue  # the snapshot does not cover this variable
            parts.append((v.id, bucket(value, v.kind, self._bins)))
        return State(tuple(parts))


__all__ = ["DEFAULT_BINS", "IGNORED_KINDS", "State", "StateEncoder", "bucket"]
