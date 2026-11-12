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
REFRESH_EVERY = 15  # steps between flushes: a second of play at four frames a step


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
    """Count cause→effect pairs in an event stream and write them to the knowledge.

    A pair's record carries its support and how often its cause and its
    effect happened at all. The pairs whose support moved this call are
    written at once, with the counts as they stand at the end of the call.
    A pair whose counts moved but whose support did not (its cause or its
    effect happened without the other) is written by :meth:`flush`, which
    runs every ``refresh_every`` calls and before a save, so a rate in the
    knowledge is at most that many steps old (T-946).
    """

    def __init__(
        self,
        knowledge: Knowledge,
        *,
        window: int = DEFAULT_WINDOW,
        refresh_every: int = REFRESH_EVERY,
    ) -> None:
        if window < 0:
            raise ConfigurationError("window must not be negative.")
        if refresh_every < 1:
            raise ConfigurationError("refresh_every must be at least 1.")
        self._knowledge = knowledge
        self._window = window
        self._refresh_every = refresh_every
        self._recent: deque[GameEvent] = deque()
        self._counts: dict[str, int] = {}
        self._pairs: dict[tuple[str, str], int] = {}
        self._last_pair_frame: dict[tuple[str, str], int] = {}
        self._by_key: dict[str, set[tuple[str, str]]] = {}
        self._stale: set[str] = set()  # keys whose count moved since the last flush
        self._calls = 0

    @property
    def window(self) -> int:
        return self._window

    @property
    def refresh_every(self) -> int:
        return self._refresh_every

    @property
    def seen(self) -> int:
        return sum(self._counts.values())

    def forget_recent(self) -> None:
        """Nothing that happens from here on followed what happened before.

        For the app putting the game back to an earlier moment of a go (T-1004).
        Everything counted so far is kept.
        """
        self._recent.clear()

    def reset(self) -> None:
        self._recent.clear()
        self._counts.clear()
        self._pairs.clear()
        self._last_pair_frame.clear()
        self._by_key.clear()
        self._stale.clear()
        self._calls = 0

    def observe(self, events: Iterable[GameEvent]) -> tuple[Relationship, ...]:
        """Feed the events of one frame (in order). Returns relationships whose support grew."""
        grown: dict[tuple[str, str], None] = {}  # insertion order: the order the pairs were seen
        for event in events:
            if not isinstance(event, GameEvent):
                raise ConfigurationError("RelationshipFinder.observe needs GameEvents.")
            key = event_key(event)
            self._counts[key] = self._counts.get(key, 0) + 1
            self._stale.add(key)
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
                    if pair not in self._pairs:
                        self._by_key.setdefault(pair[0], set()).add(pair)
                        self._by_key.setdefault(pair[1], set()).add(pair)
                    self._pairs[pair] = self._pairs.get(pair, 0) + 1
                    grown[pair] = None
            self._recent.append(event)
        # Written once each, with the counts as they stand after the whole frame.
        changed = tuple(self._knowledge.set_relationship(self._relationship(p)) for p in grown)
        self._calls += 1
        if self._calls % self._refresh_every == 0:
            self.flush()
        return changed

    def flush(self) -> int:
        """Write every pair whose cause or effect count moved since the last flush.

        Returns how many relationships were written.
        """
        stale: set[tuple[str, str]] = set()
        for key in self._stale:
            stale.update(self._by_key.get(key, ()))
        self._stale.clear()
        for pair in stale:
            self._knowledge.set_relationship(self._relationship(pair))
        return len(stale)

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


__all__ = ["DEFAULT_WINDOW", "REFRESH_EVERY", "RelationshipFinder", "describe_key", "event_key"]
