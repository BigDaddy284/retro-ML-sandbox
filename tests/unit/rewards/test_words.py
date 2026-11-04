"""T-859: rewards explain themselves in plain words."""

from __future__ import annotations

from retro_ai_lab.rewards import (
    CATEGORY_WORDS,
    Behaviour,
    Category,
    Trigger,
    describe_category,
    explain_behaviour,
)
from retro_ai_lab.understanding import EventKind


def behaviour(**kw: object) -> Behaviour:
    base: dict[str, object] = {
        "id": "setback:v1",
        "name": "Losing a life",
        "category": Category.SURVIVAL,
        "trigger": Trigger(EventKind.SETBACK, "v1"),
        "suggested": -7.5,
        "value": -7.5,
    }
    base.update(kw)
    return Behaviour(**base)  # type: ignore[arg-type]


def test_a_reward_explains_itself() -> None:
    assert explain_behaviour(behaviour()) == "Losing a life: the AI loses 7.5 points each time."
    one = explain_behaviour(behaviour(value=1.0))
    assert one == "Losing a life: the AI gets 1 point each time."
    assert explain_behaviour(behaviour(value=0.0)) == "Losing a life: the AI gets nothing for it."
    assert explain_behaviour(
        behaviour(
            name="Your score jumps up",
            value=5.625,
            suggested=3.75,
            custom=True,
            enabled=False,
            notes="The AI is not sure yet; the value is smaller until it is.",
        )
    ) == (
        "Your score jumps up: the AI gets 5.625 points each time. "
        "The AI is not sure yet; the value is smaller until it is. "
        "You set the value; the AI suggested +3.75. "
        "It is switched off, so it counts for nothing right now."
    )
    assert set(CATEGORY_WORDS) == set(Category)  # every group has its words
    assert describe_category(Category.SURVIVAL) == "Survival: staying alive"
