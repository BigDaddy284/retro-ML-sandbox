"""T-403: AI personalities as reward configurations."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards import (
    MAX_VALUE,
    PERSONALITIES,
    Behaviour,
    Category,
    Personality,
    RewardProfile,
    Trigger,
    apply_personality,
    apply_suggestions,
    personality,
)
from retro_ai_lab.understanding import (
    Encoding,
    EventKind,
    Evidence,
    GameEvent,
    Knowledge,
    VariableKind,
)

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]


def beh(id: str, category: Category, suggested: float) -> Behaviour:
    return Behaviour(id, id, category, Trigger(EventKind.PROGRESS, id), suggested, suggested)


def profile() -> RewardProfile:
    p = RewardProfile("g")
    p.add(beh("walk", Category.EXPLORATION, 4.0))
    p.add(beh("coin", Category.COLLECTION, 5.0))
    p.add(beh("hit", Category.SURVIVAL, -10.0))
    p.add(beh("fast", Category.EFFICIENCY, 2.0))
    return p


def test_the_seven_presets() -> None:
    ids = [p.id for p in PERSONALITIES]
    assert ids == [
        "explorer",
        "collector",
        "completionist",
        "speedrunner",
        "battler",
        "survivor",
        "balanced",
    ]
    assert len({p.name for p in PERSONALITIES}) == 7
    for p in PERSONALITIES:
        assert p.name and p.description.endswith(".")
        assert set(p.weights) == set(Category)
        assert all(w > 0 for w in p.weights.values())
    assert personality("balanced").weights == {c: 1.0 for c in Category}
    assert personality("explorer").weights[Category.EXPLORATION] > 1.0
    assert personality("survivor").weights[Category.SURVIVAL] > 1.0
    assert personality("speedrunner").weights[Category.EFFICIENCY] > 1.0
    assert personality("Battler") is personality("battler")
    with pytest.raises(ConfigurationError):
        personality("ghost")
    with pytest.raises(ConfigurationError):
        Personality("x", "X", "x.", {Category.COMBAT: 0.0})
    with pytest.raises(ConfigurationError):
        Personality("", "X", "x.", {})


def test_apply_rescales_untouched_values_and_keeps_a_persons() -> None:
    p = profile()
    p.set_value("coin", 1.0)
    p.set_enabled("fast", False)
    survivor = apply_personality(p, "survivor")
    assert survivor.id == "survivor" and p.personality == "survivor"
    assert p.weights == survivor.weights
    assert p.get("hit").value == -10.0 * survivor.weights[Category.SURVIVAL]
    assert p.get("walk").value == 4.0 * survivor.weights[Category.EXPLORATION]
    assert p.get("coin").value == 1.0 and p.get("coin").custom
    assert not p.get("fast").enabled
    apply_personality(p, personality("balanced"))
    assert p.get("hit").value == -10.0 and p.get("walk").value == 4.0
    assert p.get("coin").value == 1.0
    assert p.personality == "balanced"


def test_apply_clamps_scaled_values() -> None:
    p = RewardProfile("g")
    p.add(beh("big", Category.SURVIVAL, -90.0))
    apply_personality(p, "survivor")
    assert p.get("big").value == -MAX_VALUE


def test_suggest_and_reset_honour_the_weights() -> None:
    p = profile()
    apply_personality(p, "collector")
    weight = p.weights[Category.COLLECTION]
    assert p.suggest("coin", 6.0).value == 6.0 * weight
    p.set_value("coin", 1.0)
    assert p.suggest("coin", 8.0).value == 1.0
    reset = p.reset_to_suggested("coin")
    assert reset is not None and reset.value == 8.0 * weight
    with pytest.raises(ConfigurationError):
        p.set_weights({Category.COMBAT: -1.0})
    with pytest.raises(ConfigurationError):
        p.set_weights({"combat": 2.0})  # type: ignore[dict-item]
    p.set_weights({Category.COMBAT: 2.0})  # partial: the rest default to 1.0
    assert p.weights[Category.COMBAT] == 2.0 and p.weights[Category.COLLECTION] == 1.0


def test_weights_round_trip_and_default_for_older_files() -> None:
    p = profile()
    apply_personality(p, "explorer")
    data = p.to_dict()
    again = RewardProfile.from_dict(data)
    assert again.weights == p.weights and again.personality == "explorer"
    assert [b.value for b in again] == [b.value for b in p]
    del data["weights"]
    assert RewardProfile.from_dict(data).weights == {c: 1.0 for c in Category}
    with pytest.raises(ConfigurationError):
        RewardProfile.from_dict({**data, "weights": {"combat": "lots"}})


def test_personalities_change_what_the_toy_racer_earns() -> None:
    k = Knowledge("toyracer")
    k.propose(
        RAM,
        0,
        encoding=Encoding.U16LE,
        kind=VariableKind.SCORE,
        name="Score",
        evidence=SURE,
    )
    lives = k.propose(RAM, 2, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
    lap = k.propose(RAM, 3, kind=VariableKind.PROGRESS, name="Lap", evidence=SURE)
    crash = GameEvent(EventKind.SETBACK, 10, lives.id, "", 3, 2)
    checkpoint = GameEvent(EventKind.PROGRESS, 20, lap.id, "", 0, 1)

    def earns(pid: str) -> tuple[float, float]:
        p = RewardProfile("toyracer")
        apply_suggestions(p, k)
        apply_personality(p, pid)
        return p.reward_for(crash), p.reward_for(checkpoint)

    assert earns("survivor")[0] < earns("explorer")[0] < 0
    assert earns("speedrunner")[1] > earns("collector")[1] > 0
    # Later suggestions keep the personality in effect.
    p = RewardProfile("toyracer")
    apply_personality(p, "survivor")
    apply_suggestions(p, k)
    assert p.reward_for(crash) == earns("survivor")[0]
