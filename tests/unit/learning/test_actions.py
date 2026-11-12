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


# ---------------------------------------------------------------- T-998

PAD = ("up", "down", "left", "right", "a", "b", "start")


def test_near_is_one_button_away() -> None:
    space = ActionSpace.from_buttons(PAD)

    def near(*buttons: str) -> set[frozenset[str]]:
        return {space.buttons_for(i) for i in space.near(space.index_of(buttons))}

    # From holding right: keep holding it and add one of the buttons that go with it.
    # Letting go of it is stopping, not a small change.
    assert near("right") == {frozenset({"a", "right"}), frozenset({"b", "right"})}
    # From nothing there is nothing to keep doing.
    assert near() == set()
    # From right and A: let go of either.
    assert near("a", "right") == {frozenset({"a"}), frozenset({"right"})}
    assert near("a") == {frozenset({"a", d}) for d in ("up", "down", "left", "right")}
    # Each way round, never the move itself, one button apart and one still held.
    for i in range(len(space)):
        assert i not in space.near(i)
        for j in space.near(i):
            assert i in space.near(j)
            assert len(space.buttons_for(i) ^ space.buttons_for(j)) == 1
            assert space.buttons_for(i) & space.buttons_for(j)
    # Asked twice it is the same answer; a move that is not there is refused.
    assert space.near(3) is space.near(3)
    with pytest.raises(ConfigurationError):
        space.near(len(space))
    # Moves with nothing one button away from them have none, and nor have controls
    # where buttons are never held together.
    apart = ActionSpace([frozenset({"left", "a"}), frozenset({"right", "b"})])
    assert apart.near(0) == () and apart.near(1) == ()
    singles = ActionSpace.from_buttons(PAD, combos=False)
    assert all(singles.near(i) == () for i in range(len(singles)))


def test_sample_near_is_half_near_and_half_anything() -> None:
    space = ActionSpace.from_buttons(PAD)
    right = space.index_of(["right"])
    near = set(space.near(right))
    rng = Random(4)
    picks = [space.sample_near(rng, right) for _ in range(6000)]
    share = sum(p in near for p in picks) / len(picks)
    # Half are near outright, and 2 of the 15 moves are near in the other half: 57%.
    assert 0.54 <= share <= 0.60
    assert set(picks) == set(range(len(space)))  # every move is still reached
    # Each near move is as likely as the next.
    counts = [picks.count(i) for i in near]
    assert max(counts) - min(counts) < 0.15 * max(counts)
    # Nothing near: the plain pick, draw for draw.
    singles = ActionSpace.from_buttons(PAD, combos=False)
    one, other = Random(9), Random(9)
    assert [singles.sample_near(one, i % 7) for i in range(200)] == [
        singles.sample(other) for _ in range(200)
    ]
    assert one.random() == other.random()
    with pytest.raises(ConfigurationError):
        space.sample_near(rng, -1)
