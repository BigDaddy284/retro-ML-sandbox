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
    assert fm.size == 8 and fm.width == 4
    assert FeatureMap().width == DEFAULT_WIDTH and FeatureMap().size == 2 * DEFAULT_WIDTH
    assert fm.ids == ()
    fm.vector(A)
    assert fm.ids == ("lane", "speed")
    fm.vector(C)
    assert fm.ids == ("lane", "speed", "lives")
    assert [fm.slot_of(i) for i in fm.ids] == [0, 1, 2]
    fm.vector(B)
    assert fm.ids == ("lane", "speed", "lives") and fm.size == 8
    with pytest.raises(ConfigurationError, match="score"):
        fm.slot_of("score")


def test_vector_values_present_absent_and_negative() -> None:
    fm = FeatureMap(width=3, bins=DEFAULT_BINS)
    span = DEFAULT_BINS + 16
    v = fm.vector(A)
    assert v.dtype == np.float32 and v.shape == (6,)
    assert v.tolist() == pytest.approx([1, 1 / span, 1, 3 / span, 0, 0])
    v = fm.vector(C)  # speed is absent now, lives is new
    assert v.tolist() == pytest.approx([1, 2 / span, 0, 0, 1, 2 / span])
    v = fm.vector(State((("speed", -2),)))
    assert v.tolist() == pytest.approx([0, 0, 1, -2 / span, 0, 0])
    assert fm.vector(State()).tolist() == [0.0] * 6


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
    assert v[0] == 1.0 and -1.0 <= float(v[1]) <= 1.0
    assert (v[1] > 0) == (value > 0 if kind is not VariableKind.FLAG else bool(value))


def test_batch_matches_single_vectors_and_tells_states_apart() -> None:
    fm = FeatureMap(width=4)
    rows = fm.batch([A, B, C, State()])
    assert rows.shape == (4, 8) and rows.dtype == np.float32
    for row, state in zip(rows, [A, B, C, State()], strict=True):
        assert np.array_equal(row, fm.vector(state))
    assert not np.array_equal(rows[0], rows[1])
    assert not np.array_equal(rows[1], rows[2])
    assert not rows[3].any()
    assert fm.batch([]).shape == (0, 8)


def test_validation_and_overflow() -> None:
    for bad in ({"width": 0}, {"bins": 1}, {"bins": 65}):
        with pytest.raises(ConfigurationError):
            FeatureMap(**bad)  # type: ignore[arg-type]
    fm = FeatureMap(width=2)
    fm.vector(C)  # lane, lives
    v = fm.vector(State((("lane", 1), ("score", 9), ("timer", 1))))
    assert fm.ids == ("lane", "lives") and fm.dropped == 2
    assert v.tolist() == pytest.approx([1, 1 / 24, 0, 0])
    assert fm.vector(State((("score", 3),))).tolist() == [0.0] * 4 and fm.dropped == 2


def test_json_round_trip_and_bad_data() -> None:
    fm = FeatureMap(width=3, bins=4)
    fm.vector(C)
    fm.vector(State((("a", 1), ("b", 1), ("c", 1))))
    data = fm.to_dict()
    assert data["schema"] == SCHEMA_VERSION and data["ids"] == ["lane", "lives", "a"]
    again = FeatureMap.from_dict(data)
    assert again.ids == fm.ids and again.width == 3 and again.bins == 4 and again.dropped == 2
    assert again.to_dict() == data
    for state in (A, B, C):  # A and B bring "speed", dropped by both maps alike
        assert np.array_equal(again.vector(state), fm.vector(state))
    assert again.to_dict() == fm.to_dict() and again.dropped == 3
    for bad in (
        "nope",
        {**data, "ids": ["lane", "lane"]},
        {**data, "ids": ["a", "b", "c", "d"]},
        {**data, "width": 0},
        {**data, "schema": SCHEMA_VERSION + 1},
    ):
        with pytest.raises(ConfigurationError):
            FeatureMap.from_dict(bad)  # type: ignore[arg-type]
