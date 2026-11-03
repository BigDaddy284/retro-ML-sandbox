"""A running count of the rewards earned while playing."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards.model import Category, RewardProfile


@dataclass(frozen=True, slots=True)
class RewardEntry:
    frame: int
    amount: float
    behaviour_ids: tuple[str, ...]


@dataclass(slots=True)
class BehaviourTotal:
    count: int = 0
    amount: float = 0.0


class RewardTally:
    """Sums rewards overall, per behaviour and per category; remembers recent ones."""

    def __init__(self, capacity: int = 1000) -> None:
        if capacity < 1:
            raise ConfigurationError("capacity must be positive.")
        self._recent: deque[RewardEntry] = deque(maxlen=capacity)
        self._by_behaviour: dict[str, BehaviourTotal] = {}
        self._total = 0.0
        self._count = 0

    def __len__(self) -> int:
        return len(self._recent)

    def __iter__(self) -> Iterator[RewardEntry]:
        return iter(tuple(self._recent))

    @property
    def total(self) -> float:
        return self._total

    @property
    def count(self) -> int:
        """Rewards ever recorded, including ones that scrolled off ``recent``."""
        return self._count

    def record(
        self,
        frame: int,
        amount: float,
        behaviour_ids: Iterable[str],
        *,
        shares: Iterable[float] = (),
    ) -> RewardEntry:
        """Record one reward. ``shares`` (same order as ids) says how much each behaviour gave."""
        ids = tuple(behaviour_ids)
        parts = tuple(shares)
        if parts and len(parts) != len(ids):
            raise ConfigurationError("shares must match behaviour_ids.")
        entry = RewardEntry(int(frame), float(amount), ids)
        self._recent.append(entry)
        self._total += entry.amount
        self._count += 1
        for i, bid in enumerate(ids):
            slot = self._by_behaviour.setdefault(bid, BehaviourTotal())
            slot.count += 1
            slot.amount += parts[i] if parts else entry.amount / len(ids)
        return entry

    def by_behaviour(self) -> dict[str, BehaviourTotal]:
        return {k: BehaviourTotal(v.count, v.amount) for k, v in self._by_behaviour.items()}

    def by_category(self, profile: RewardProfile) -> dict[Category, float]:
        out = {c: 0.0 for c in Category}
        for bid, slot in self._by_behaviour.items():
            b = profile.find(bid)
            if b is not None:
                out[b.category] += slot.amount
        return out

    def recent(self, n: int) -> tuple[RewardEntry, ...]:
        if n < 0:
            raise ConfigurationError("n must not be negative.")
        return tuple(list(self._recent)[-n:]) if n else ()

    def since(self, frame: int) -> tuple[RewardEntry, ...]:
        return tuple(e for e in self._recent if e.frame >= frame)

    def reset(self) -> None:
        self._recent.clear()
        self._by_behaviour.clear()
        self._total = 0.0
        self._count = 0


__all__ = ["BehaviourTotal", "RewardEntry", "RewardTally"]
