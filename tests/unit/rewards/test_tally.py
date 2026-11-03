"""T-404: the running reward tally."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards import (
    Behaviour,
    BehaviourTotal,
    Category,
    RewardProfile,
    RewardTally,
    Trigger,
)
from retro_ai_lab.understanding import EventKind


def test_tally_sums_counts_and_groups() -> None:
    t = RewardTally(capacity=3)
    assert t.total == 0.0 and t.count == 0 and len(t) == 0
    t.record(1, 5.0, ["a"])
    t.record(2, -4.0, ["b", "c"], shares=[-3.0, -1.0])
    t.record(3, 2.0, ["a", "b"])  # no shares: split evenly
    t.record(4, 1.0, [])  # a reward with no behaviour still counts
    assert t.total == 4.0 and t.count == 4 and len(t) == 3  # capacity trims recent only
    assert t.by_behaviour() == {
        "a": BehaviourTotal(2, 6.0),
        "b": BehaviourTotal(2, -2.0),
        "c": BehaviourTotal(1, -1.0),
    }
    assert [e.frame for e in t] == [2, 3, 4]
    assert [e.frame for e in t.recent(2)] == [3, 4] and t.recent(0) == ()
    assert [e.frame for e in t.since(3)] == [3, 4]
    p = RewardProfile("g")
    p.add(Behaviour("a", "A", Category.COLLECTION, Trigger(EventKind.SCORE_GAIN), 1.0, 1.0))
    p.add(Behaviour("b", "B", Category.SURVIVAL, Trigger(EventKind.SETBACK), -1.0, -1.0))
    groups = t.by_category(p)
    assert groups[Category.COLLECTION] == 6.0 and groups[Category.SURVIVAL] == -2.0
    assert groups[Category.COMBAT] == 0.0  # "c" is not in the profile: ignored
    t.reset()
    assert t.total == 0.0 and t.count == 0 and t.by_behaviour() == {}


def test_tally_rejects_bad_input() -> None:
    with pytest.raises(ConfigurationError):
        RewardTally(0)
    t = RewardTally()
    with pytest.raises(ConfigurationError):
        t.record(1, 1.0, ["a", "b"], shares=[1.0])
    with pytest.raises(ConfigurationError):
        t.recent(-1)
