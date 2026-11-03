"""Relationship discovery: which events tend to follow which.

DESIGN.md §3.3 wants the app to see that "reach a checkpoint" leads to
"score jumps" and that "lives hit zero" leads to "game over", rather than
treating them as unrelated memory changes. The finder watches the event
stream and counts, for every pair of event keys, how often the second
happened within ``window`` frames after the first (events on the same
frame count both ways, since their order is an accident of memory layout). The counts live in the
knowledge as :class:`Relationship` records so a person can confirm or
reject them and they survive restarts.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.understanding.events import EventKind, GameEvent
from retro_ai_lab.understanding.knowledge import Knowledge, Relationship

DEFAULT_WINDOW = 5


def event_key(event: GameEvent) -> str:
    return f"{event.kind.value}:{event.variable_id or '*'}"


def describe_key(key: str, knowledge: Knowledge | None = None) -> str:
    """Plain words for an event key, e.g. ``"Progress of Checkpoint"``."""
    kind_value, _, var_id = key.partition(":")
    try:
        kind = EventKind(kind_value).label
    except ValueError:
        kind = kind_value
    if var_id == "*" or knowledge is None or var_id not in knowledge:
        return kind if var_id == "*" else f"{kind} of {var_id}"
    return f"{kind} of {knowledge.get(var_id).display_name}"


class RelationshipFinder:
    """Count cause→effect pairs in an event stream and write them to the knowledge."""

    def __init__(self, knowledge: Knowledge, *, window: int = DEFAULT_WINDOW) -> None:
        if window < 0:
            raise ConfigurationError("window must not be negative.")
        self._knowledge = knowledge
        self._window = window
        self._recent: deque[GameEvent] = deque()
        self._counts: dict[str, int] = {}
        self._pairs: dict[tuple[str, str], int] = {}
        self._last_pair_frame: dict[tuple[str, str], int] = {}

    @property
    def window(self) -> int:
        return self._window

    @property
    def seen(self) -> int:
        return sum(self._counts.values())

    def reset(self) -> None:
        self._recent.clear()
        self._counts.clear()
        self._pairs.clear()
        self._last_pair_frame.clear()

    def observe(self, events: Iterable[GameEvent]) -> tuple[Relationship, ...]:
        """Feed the events of one frame (in order). Returns relationships that changed."""
        changed: dict[str, Relationship] = {}
        for event in events:
            if not isinstance(event, GameEvent):
                raise ConfigurationError("RelationshipFinder.observe needs GameEvents.")
            key = event_key(event)
            self._counts[key] = self._counts.get(key, 0) + 1
            while self._recent and event.frame - self._recent[0].frame > self._window:
                self._recent.popleft()
            for earlier in self._recent:
                cause = event_key(earlier)
                if cause == key:
                    continue
                pairs = [(cause, key)]
                if earlier.frame == event.frame:
                    # Same frame: order is arbitrary, so the link goes both ways.
                    pairs.append((key, cause))
                for pair in pairs:
                    # One effect occurrence can only confirm a cause once.
                    if self._last_pair_frame.get(pair) == event.frame:
                        continue
                    self._last_pair_frame[pair] = event.frame
                    self._pairs[pair] = self._pairs.get(pair, 0) + 1
                    rel = self._knowledge.set_relationship(self._relationship(pair))
                    changed[rel.id] = rel
            self._recent.append(event)
        # Counts of causes/effects moved too, so refresh every stored pair's rates.
        for pair in self._pairs:
            rel_id = f"{pair[0]}->{pair[1]}"
            if rel_id not in changed:
                self._knowledge.set_relationship(self._relationship(pair))
        return tuple(changed.values())

    def _relationship(self, pair: tuple[str, str]) -> Relationship:
        cause, effect = pair
        support = self._pairs[pair]
        return Relationship(
            cause,
            effect,
            min(support, self._counts[cause], self._counts[effect]),
            self._counts[cause],
            self._counts[effect],
            self._window,
        )


__all__ = ["DEFAULT_WINDOW", "RelationshipFinder", "describe_key", "event_key"]
