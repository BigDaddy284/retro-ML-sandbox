"""T-703: a situation as numbers a neural network can read."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import DEFAULT_BINS, FeatureMap, State, bucket
from retro_ai_lab.learning.features import DEFAULT_WIDTH, SCHEMA_VERSION
from retro_ai_lab.understanding import VariableKind

A = State((("lane", 1), ("speed", 3)))
B = State((("lane", 2), ("speed", 3)))
C = State((("lane", 2), ("lives", 2)))


def test_slots_follow_first_sight_and_never_move() -> None:
    fm = FeatureMap(width=4)
    assert fm.size == 12 and fm.width == 4
    assert FeatureMap().width == DEFAULT_WIDTH and FeatureMap().size == 3 * DEFAULT_WIDTH
    assert fm.ids == ()
    fm.vector(A)
    assert fm.ids == ("lane", "speed")
    fm.vector(C)
    assert fm.ids == ("lane", "speed", "lives")
    assert [fm.slot_of(i) for i in fm.ids] == [0, 1, 2]
    fm.vector(B)
    assert fm.ids == ("lane", "speed", "lives") and fm.size == 12
    with pytest.raises(ConfigurationError, match="score"):
        fm.slot_of("score")


def test_vector_values_present_absent_and_negative() -> None:
    fm = FeatureMap(width=3, bins=DEFAULT_BINS)
    span = DEFAULT_BINS + 16
    fine = DEFAULT_BINS
    v = fm.vector(A)
    assert v.dtype == np.float32 and v.shape == (9,)
    assert v.tolist() == pytest.approx([1, 1 / fine, 1 / span, 1, 3 / fine, 3 / span, 0, 0, 0])
    v = fm.vector(C)  # speed is absent now, lives is new
    assert v.tolist() == pytest.approx([1, 2 / fine, 2 / span, 0, 0, 0, 1, 2 / fine, 2 / span])
    v = fm.vector(State((("speed", -2),)))
    assert v.tolist() == pytest.approx([0, 0, 0, 1, -2 / fine, -2 / span, 0, 0, 0])
    assert fm.vector(State((("lane", 20),))).tolist() == pytest.approx([1, 1, 20 / span] + [0] * 6)
    assert fm.vector(State()).tolist() == [0.0] * 9


@given(
    value=st.integers(min_value=-65536, max_value=65535),
    bins=st.integers(min_value=2, max_value=64),
    kind=st.sampled_from([VariableKind.POSITION, VariableKind.SCORE, VariableKind.FLAG]),
)
def test_every_feature_lies_between_minus_one_and_one(
    value: int, bins: int, kind: VariableKind
) -> None:
    fm = FeatureMap(width=1, bins=bins)
    v = fm.vector(State((("x", bucket(value, kind, bins)),)))
    assert v[0] == 1.0 and -1.0 <= float(v[1]) <= 1.0 and -1.0 <= float(v[2]) <= 1.0
    positive = value > 0 if kind is not VariableKind.FLAG else bool(value)
    assert (v[1] > 0) == positive and (v[2] > 0) == positive


def test_batch_matches_single_vectors_and_tells_states_apart() -> None:
    fm = FeatureMap(width=4)
    rows = fm.batch([A, B, C, State()])
    assert rows.shape == (4, 12) and rows.dtype == np.float32
    for row, state in zip(rows, [A, B, C, State()], strict=True):
        assert np.array_equal(row, fm.vector(state))
    assert not np.array_equal(rows[0], rows[1])
    assert not np.array_equal(rows[1], rows[2])
    assert not rows[3].any()
    assert fm.batch([]).shape == (0, 12)


def test_validation_and_overflow() -> None:
    for bad in ({"width": 0}, {"bins": 1}, {"bins": 65}):
        with pytest.raises(ConfigurationError):
            FeatureMap(**bad)  # type: ignore[arg-type]
    fm = FeatureMap(width=2)
    fm.vector(C)  # lane, lives
    # lives is absent from this state, so score takes its slot (T-908); timer finds
    # both holders present and is dropped.
    v = fm.vector(State((("lane", 1), ("score", 9), ("timer", 1))))
    assert fm.ids == ("lane", "score") and fm.dropped == 1 and fm.evicted == 1
    assert fm.slot_of("score") == 1
    assert v.tolist() == pytest.approx([1, 1 / 8, 1 / 24, 1, 1, 9 / 24])
    assert fm.vector(State((("score", 3),))).tolist() == pytest.approx([0, 0, 0, 1, 3 / 8, 3 / 24])
    assert fm.dropped == 1


def test_json_round_trip_and_bad_data() -> None:
    fm = FeatureMap(width=3, bins=4)
    fm.vector(C)
    fm.vector(State((("a", 1), ("b", 1), ("c", 1))))
    data = fm.to_dict()
    # a took the free slot; lane and lives were absent, so b and c took theirs (T-908).
    assert data["schema"] == SCHEMA_VERSION and data["ids"] == ["b", "c", "a"]
    assert fm.evicted == 2 and fm.dropped == 0
    again = FeatureMap.from_dict(data)
    assert again.ids == fm.ids and again.width == 3 and again.bins == 4 and again.dropped == 0
    assert again.to_dict() == data
    for state in (A, B, C):  # both maps hand the absent holders' slots over alike
        assert np.array_equal(again.vector(state), fm.vector(state))
    assert again.to_dict() == fm.to_dict() and again.ids == ("lane", "speed", "lives")
    for bad in (
        "nope",
        {**data, "ids": ["lane", "lane"]},
        {**data, "ids": ["a", "b", "c", "d"]},
        {**data, "width": 0},
        {**data, "schema": SCHEMA_VERSION + 1},
    ):
        with pytest.raises(ConfigurationError):
            FeatureMap.from_dict(bad)  # type: ignore[arg-type]


def test_a_full_map_gives_an_absent_slot_to_a_newcomer() -> None:
    """T-908: the user's brain never saw a position found after its slots had filled."""
    fm = FeatureMap(width=3)
    fm.vector(State((("flag1", 1), ("flag2", 0), ("flag3", 1))))
    assert fm.ids == ("flag1", "flag2", "flag3")
    fm.vector(State((("flag1", 1), ("flag3", 1))))  # flag2 unseen since fill 1
    fm.vector(State((("flag1", 1),)))  # flag3 unseen since fill 2
    # A position arrives with flag1 only: flag2, unseen longest, gives up its slot.
    fm.vector(State((("flag1", 1), ("position", 5))))
    assert fm.slot_of("position") == 1 and fm.evicted == 1 and fm.dropped == 0
    assert set(fm.ids) == {"flag1", "flag3", "position"}
    with pytest.raises(ConfigurationError):
        fm.slot_of("flag2")
    # Every holder present: the newcomer is dropped, and taken in when a holder is away.
    fm.vector(State((("flag1", 1), ("flag3", 1), ("position", 5), ("lives", 2))))
    assert fm.dropped == 1 and "lives" not in fm.ids
    fm.vector(State((("flag1", 1), ("position", 5), ("lives", 2))))
    assert "lives" in fm.ids and fm.dropped == 0 and fm.evicted == 2
    # The map round-trips with its new holders.
    again = FeatureMap.from_dict(fm.to_dict())
    assert again.ids == fm.ids and [again.slot_of(i) for i in fm.ids] == [
        fm.slot_of(i) for i in fm.ids
    ]
