"""T-702: a bounded memory of past steps the AI can learn from again."""

from __future__ import annotations

import dataclasses
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import ReplayBuffer, State, Transition

A = State((("v1", 1),))
B = State((("v1", 2),))


def t(n: int, source: str = "ai") -> Transition:
    return Transition(A, n % 3, float(n), B, n % 5 == 0, source)


def test_buffer_keeps_the_newest_and_counts_everything() -> None:
    buf = ReplayBuffer(3)
    assert len(buf) == 0 and buf.capacity == 3 and buf.added == 0
    for i in range(5):
        buf.append(t(i))
    assert len(buf) == 3 and buf.added == 5
    assert [x.reward for x in buf] == [2.0, 3.0, 4.0]
    assert [x.reward for x in buf.latest(2)] == [3.0, 4.0]
    assert buf.latest(10) == tuple(buf)
    buf.extend([t(5, "child"), t(6)])
    assert [x.reward for x in buf] == [4.0, 5.0, 6.0] and buf.added == 7
    assert tuple(buf)[1].source == "child"
    buf.clear()
    assert len(buf) == 0 and buf.added == 7


def test_sampling_is_distinct_seeded_and_checked() -> None:
    buf = ReplayBuffer(10, seed=3)
    with pytest.raises(ConfigurationError, match="empty"):
        buf.sample(1)
    buf.extend(t(i) for i in range(10))
    batch = buf.sample(4)
    assert len(batch) == 4 and len({x.reward for x in batch}) == 4
    assert all(x in tuple(buf) for x in batch)
    twin = ReplayBuffer(10, seed=3)
    twin.extend(t(i) for i in range(10))
    assert [x.reward for x in twin.sample(4)] == [x.reward for x in batch]
    assert len(buf.sample(10)) == 10
    with pytest.raises(ConfigurationError, match="only holds"):
        buf.sample(11)
    with pytest.raises(ConfigurationError, match="at least one"):
        buf.sample(0)


def test_validation_and_immutability() -> None:
    with pytest.raises(ConfigurationError):
        ReplayBuffer(0)
    with pytest.raises(ConfigurationError, match="finite"):
        Transition(A, 0, float("nan"), B, False)
    with pytest.raises(ConfigurationError, match="finite"):
        Transition(A, 0, float("inf"), B, False)
    with pytest.raises(ConfigurationError, match="action"):
        Transition(A, -1, 0.0, B, False)
    with pytest.raises(ConfigurationError, match="source"):
        Transition(A, 0, 0.0, B, False, "")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t(1).reward = 2.0  # type: ignore[misc]
    buf = ReplayBuffer(2)
    with pytest.raises(ConfigurationError):
        buf.append("not a transition")  # type: ignore[arg-type]


def test_round_trip_keeps_order_counts_and_random_state() -> None:
    buf = ReplayBuffer(4, seed=11)
    buf.extend(t(i, "child" if i == 2 else "ai") for i in range(6))
    buf.sample(2)  # moves the random state along
    data = json.loads(json.dumps(buf.to_dict()))
    again = ReplayBuffer.from_dict(data)
    assert again.capacity == 4 and again.added == 6 and tuple(again) == tuple(buf)
    assert [x.reward for x in again.sample(3)] == [x.reward for x in buf.sample(3)]
    assert next(iter(again)).source == "child"
    # More saved than fits: the newest are kept.
    big = dict(data, capacity=2)
    small = ReplayBuffer.from_dict(big)
    assert [x.reward for x in small] == [4.0, 5.0] and small.added == 6
    for bad in ({}, {"capacity": 0}, {"capacity": 2, "transitions": [{"action": 1}]}, "x"):
        with pytest.raises(ConfigurationError):
            ReplayBuffer.from_dict(bad)  # type: ignore[arg-type]


@given(st.integers(1, 6), st.lists(st.integers(0, 50), max_size=30))
def test_buffer_holds_exactly_the_tail(capacity: int, rewards: list[int]) -> None:
    buf = ReplayBuffer(capacity)
    for r in rewards:
        buf.append(Transition(A, 0, float(r), B, False))
    assert len(buf) == min(len(rewards), capacity) and buf.added == len(rewards)
    assert [x.reward for x in buf] == [float(r) for r in rewards[-capacity:]]


def test_a_transition_carries_the_steps_its_return_spans() -> None:
    """T-963: an n-step return remembers how many steps it covers; 1 when not said."""
    one = Transition(A, 0, 1.0, B, False)
    assert one.steps == 1 and one.to_dict()["steps"] == 1
    three = Transition(A, 0, 2.5, B, False, steps=3)
    assert Transition.from_dict(json.loads(json.dumps(three.to_dict()))) == three
    saved_before = {k: v for k, v in one.to_dict().items() if k != "steps"}
    assert Transition.from_dict(saved_before) == one  # a memory saved before T-963
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ConfigurationError):
            Transition(A, 0, 1.0, B, False, steps=bad)  # type: ignore[arg-type]
