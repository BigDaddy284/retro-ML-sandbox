"""T-502: the learner's action menu."""

from __future__ import annotations

from random import Random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retro_ai_lab.core import Button, ConfigurationError
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.learning import ActionSpace
from retro_ai_lab.learning.actions import NOTHING


def test_toy_racer_menu() -> None:
    space = ActionSpace.from_buttons(BUTTONS)
    assert space.labels == (
        NOTHING,
        "up",
        "down",
        "left",
        "right",
        "a",
        "b",
        "up + a",
        "up + b",
        "down + a",
        "down + b",
        "left + a",
        "left + b",
        "right + a",
        "right + b",
    )
    assert len(space) == 15 and space[0].buttons == frozenset()
    assert space[7].buttons == frozenset({"up", "a"}) and space[7].index == 7
    assert all("start" not in a.buttons for a in space)


def test_options_and_validation() -> None:
    plain = ActionSpace.from_buttons(BUTTONS, combos=False)
    assert plain.labels == (NOTHING, "up", "down", "left", "right", "a", "b")
    with_start = ActionSpace.from_buttons(BUTTONS, exclude=())
    assert "start" in with_start.labels and "up + start" in with_start.labels
    nothing_left = ActionSpace.from_buttons(["start"], exclude=["start"])
    assert nothing_left.labels == (NOTHING,)
    assert ActionSpace.from_buttons(["x", "y"]).labels == (NOTHING, "x", "y")
    with pytest.raises(ConfigurationError):
        ActionSpace.from_buttons([Button("a", "A"), Button("a", "A again")])
    with pytest.raises(ConfigurationError):
        ActionSpace([])
    with pytest.raises(ConfigurationError):
        ActionSpace([frozenset(), frozenset()])
    with pytest.raises(ConfigurationError):
        ActionSpace([frozenset({""})])
    space = ActionSpace.from_buttons(BUTTONS)
    for bad in (-1, 15, True, "1"):
        with pytest.raises(ConfigurationError):
            space[bad]  # type: ignore[index]
    with pytest.raises(ConfigurationError):
        space.index_of(["start"])


def test_round_trip_and_lookups() -> None:
    space = ActionSpace.from_buttons(BUTTONS)
    again = ActionSpace.from_dict(space.to_dict())
    assert again == space and again.labels == space.labels and hash(again) == hash(space)
    for a in space:
        assert space.index_of(a.buttons) == a.index
        assert space.buttons_for(a.index) == a.buttons
    assert ActionSpace.from_dict({"actions": [[], ["b"]]}).labels == (NOTHING, "b")
    with pytest.raises(ConfigurationError):
        ActionSpace.from_dict({"nope": []})
    assert space != ActionSpace.from_buttons(BUTTONS, combos=False)


@given(st.integers(min_value=0, max_value=10_000))
def test_sample_is_always_a_valid_index(seed: int) -> None:
    space = ActionSpace.from_buttons(BUTTONS)
    i = space.sample(Random(seed))
    assert 0 <= i < len(space) and space[i].index == i


def test_matches_picks_the_exact_or_largest_contained_action() -> None:
    space = ActionSpace.from_buttons(BUTTONS)
    assert space.matches(["a", "up"]).label == "up + a"
    assert space.matches(["up", "left", "a"]).label in ("up + a", "left + a")
    assert space.matches(["start"]).label == NOTHING
    assert space.matches(["up", "start"]).label == "up"
    assert space.matches([]).label == NOTHING
