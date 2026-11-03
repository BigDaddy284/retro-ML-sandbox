"""T-401: the reward model (behaviours, categories, simple controls)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards import (
    MAX_VALUE,
    SCHEMA_VERSION,
    STEP,
    Behaviour,
    Category,
    RewardProfile,
    Trigger,
)
from retro_ai_lab.understanding import EventKind, GameEvent


def ev(kind: EventKind, variable_id: str | None = "v1", frame: int = 0) -> GameEvent:
    return GameEvent(kind, frame, variable_id, "test")


def beh(
    id: str = "b1",
    kind: EventKind = EventKind.SCORE_GAIN,
    variable_id: str | None = "v1",
    *,
    suggested: float = 5.0,
    value: float | None = None,
    category: Category = Category.COLLECTION,
    enabled: bool = True,
) -> Behaviour:
    return Behaviour(
        id,
        id.capitalize(),
        category,
        Trigger(kind, variable_id),
        suggested=suggested,
        value=suggested if value is None else value,
        enabled=enabled,
    )


# ------------------------------------------------------------ categories


def test_categories_follow_the_design_order_and_have_labels() -> None:
    assert [c.label for c in Category] == [
        "Exploration",
        "Collection",
        "Combat",
        "Survival",
        "Progression",
        "Quests and Story",
        "Efficiency",
        "Completion",
    ]
    assert Category("quests") is Category.QUESTS
    with pytest.raises(ValueError):
        Category("fighting")


# ------------------------------------------------------------ triggers


def test_trigger_matches_kind_and_variable() -> None:
    t = Trigger(EventKind.SCORE_GAIN, "v1")
    assert t.matches(ev(EventKind.SCORE_GAIN, "v1"))
    assert not t.matches(ev(EventKind.SCORE_GAIN, "v2"))
    assert not t.matches(ev(EventKind.PROGRESS, "v1"))
    assert t.key == "score_gain:v1"
    assert "Score gain" in t.describe()


def test_trigger_without_variable_matches_any_variable_of_that_kind() -> None:
    t = Trigger(EventKind.SETBACK)
    assert t.matches(ev(EventKind.SETBACK, "v1"))
    assert t.matches(ev(EventKind.SETBACK, None))
    assert not t.matches(ev(EventKind.RECOVERY, "v1"))
    assert t.key == "setback:*"
    assert Trigger.from_dict(t.to_dict()) == t
    with pytest.raises(ConfigurationError):
        Trigger.from_dict({"kind": "explode"})


# ------------------------------------------------------------ behaviours


def test_behaviour_validates_its_fields() -> None:
    b = beh()
    assert b.display_name == "B1" and not b.is_penalty and not b.custom
    with pytest.raises(ConfigurationError):
        beh(id="")
    with pytest.raises(ConfigurationError):
        Behaviour("x", "", Category.COMBAT, Trigger(EventKind.RESET))
    with pytest.raises(ConfigurationError):
        beh(value=math.inf)
    with pytest.raises(ConfigurationError):
        beh(suggested=math.nan)
    with pytest.raises(ConfigurationError):
        beh(value=MAX_VALUE + 1)
    with pytest.raises(ConfigurationError):
        beh(value=-MAX_VALUE - 0.5)
    with pytest.raises(ConfigurationError):
        Behaviour("x", "X", "combat", Trigger(EventKind.RESET))  # type: ignore[arg-type]


def test_behaviour_round_trips_and_rejects_bad_data() -> None:
    b = beh(value=-3.0, enabled=False).replace(custom=True, notes="ouch")
    assert b.is_penalty
    again = Behaviour.from_dict(b.to_dict())
    assert again == b
    with pytest.raises(ConfigurationError):
        Behaviour.from_dict({"id": "x"})
    with pytest.raises(ConfigurationError):
        Behaviour.from_dict({**b.to_dict(), "category": "fighting"})


# ------------------------------------------------------------ profile


def test_profile_add_get_remove_and_duplicates() -> None:
    p = RewardProfile("fake:toy")
    assert p.game_id == "fake:toy" and len(p) == 0 and p.ids() == ()
    b = beh()
    p.add(b)
    assert p.get("b1") == b and "b1" in p and len(p) == 1
    with pytest.raises(ConfigurationError, match="already"):
        p.add(beh())
    with pytest.raises(ConfigurationError):
        p.get("nope")
    assert p.find("nope") is None
    p.remove("b1")
    assert "b1" not in p
    with pytest.raises(ConfigurationError):
        p.remove("b1")
    with pytest.raises(ConfigurationError):
        RewardProfile("")


def test_plus_and_minus_move_one_step_clamp_and_mark_custom() -> None:
    p = RewardProfile("g")
    p.add(beh(suggested=5.0))
    assert p.increase("b1").value == 5.0 + STEP
    assert p.get("b1").custom
    assert p.decrease("b1").value == 5.0
    p.set_value("b1", MAX_VALUE - 0.25)
    assert p.increase("b1").value == MAX_VALUE
    assert p.increase("b1").value == MAX_VALUE  # stays clamped
    p.set_value("b1", -MAX_VALUE + 0.25)
    assert p.decrease("b1").value == -MAX_VALUE
    assert p.decrease("b1").value == -MAX_VALUE
    reset = p.reset_to_suggested("b1")
    assert reset is not None and reset.value == 5.0
    assert not p.get("b1").custom
    with pytest.raises(ConfigurationError):
        p.increase("b1", step=0)
    with pytest.raises(ConfigurationError):
        p.set_value("b1", math.nan)


def test_set_value_and_enabled() -> None:
    p = RewardProfile("g")
    p.add(beh())
    assert p.set_value("b1", -2.5).value == -2.5
    assert p.get("b1").custom and p.get("b1").is_penalty
    assert p.set_value("b1", 500).value == MAX_VALUE  # clamped, not an error
    assert not p.set_enabled("b1", False).enabled
    assert p.set_enabled("b1", True).enabled
    assert p.enabled_count() == 1


def test_suggest_moves_the_value_only_until_a_person_sets_it() -> None:
    p = RewardProfile("g")
    p.add(beh(suggested=5.0))
    assert p.suggest("b1", 8.0).value == 8.0
    assert p.get("b1").suggested == 8.0
    p.set_value("b1", 2.0)
    assert p.suggest("b1", 20.0).value == 2.0
    assert p.get("b1").suggested == 20.0
    reset = p.reset_to_suggested("b1")
    assert reset is not None and reset.value == 20.0
    with pytest.raises(ConfigurationError):
        p.suggest("b1", math.inf)


def test_reset_all_to_suggested() -> None:
    p = RewardProfile("g")
    p.add(beh("a", suggested=1.0))
    p.add(beh("b", suggested=2.0))
    p.set_value("a", 9.0)
    p.set_value("b", 9.0)
    p.reset_to_suggested()
    assert [b.value for b in p] == [1.0, 2.0]
    assert not any(b.custom for b in p)


def test_reward_for_sums_enabled_matching_behaviours() -> None:
    p = RewardProfile("g")
    p.add(beh("score", EventKind.SCORE_GAIN, "v1", suggested=3.0))
    p.add(beh("any_score", EventKind.SCORE_GAIN, None, suggested=1.0))
    p.add(beh("off", EventKind.SCORE_GAIN, "v1", suggested=50.0, enabled=False))
    p.add(beh("hit", EventKind.SETBACK, "lives", suggested=-4.0, category=Category.SURVIVAL))
    assert p.reward_for(ev(EventKind.SCORE_GAIN, "v1")) == 4.0
    assert p.reward_for(ev(EventKind.SCORE_GAIN, "v9")) == 1.0
    assert p.reward_for(ev(EventKind.SETBACK, "lives")) == -4.0
    assert p.reward_for(ev(EventKind.RESET, None)) == 0.0
    events = (ev(EventKind.SCORE_GAIN, "v1"), ev(EventKind.SETBACK, "lives"))
    assert p.reward_for_events(events) == 0.0
    assert p.matching(ev(EventKind.SCORE_GAIN, "v1")) == (p.get("score"), p.get("any_score"))


def test_by_category_lists_every_category_in_order() -> None:
    p = RewardProfile("g")
    p.add(beh("s", category=Category.SURVIVAL))
    p.add(beh("c1", category=Category.COLLECTION))
    p.add(beh("c2", category=Category.COLLECTION))
    groups = p.by_category()
    assert list(groups) == list(Category)
    assert groups[Category.COLLECTION] == (p.get("c1"), p.get("c2"))
    assert groups[Category.SURVIVAL] == (p.get("s"),)
    assert groups[Category.COMBAT] == ()


def test_profile_round_trips_and_rejects_bad_or_newer_data() -> None:
    p = RewardProfile("g", personality="explorer")
    p.add(beh("a", suggested=1.0))
    p.add(beh("b", EventKind.SETBACK, None, suggested=-2.0, category=Category.SURVIVAL))
    p.increase("a")
    p.set_enabled("b", False)
    data = p.to_dict()
    assert data["schema"] == SCHEMA_VERSION and data["personality"] == "explorer"
    again = RewardProfile.from_dict(data)
    assert again.game_id == "g" and again.personality == "explorer"
    assert list(again) == list(p)
    with pytest.raises(ConfigurationError, match="newer"):
        RewardProfile.from_dict({**data, "schema": SCHEMA_VERSION + 1})
    with pytest.raises(ConfigurationError):
        RewardProfile.from_dict({"schema": 1})
    with pytest.raises(ConfigurationError):
        RewardProfile.from_dict({**data, "behaviours": [{"id": "x"}]})
    p.personality = "custom"
    assert p.personality == "custom"
    with pytest.raises(ConfigurationError):
        p.personality = ""


@given(
    start=st.floats(min_value=-MAX_VALUE, max_value=MAX_VALUE),
    clicks=st.lists(st.sampled_from(["+", "-"]), max_size=60),
)
def test_clicking_never_leaves_the_allowed_range(start: float, clicks: list[str]) -> None:
    p = RewardProfile("g")
    p.add(beh(suggested=0.0, value=start))
    for c in clicks:
        b = p.increase("b1") if c == "+" else p.decrease("b1")
        assert -MAX_VALUE <= b.value <= MAX_VALUE
