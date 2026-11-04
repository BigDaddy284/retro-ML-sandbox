"""T-860: a reward a person adds on any value going up or down."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards import (
    OWN_NOTE,
    Category,
    RewardProfile,
    Trigger,
    explain_behaviour,
    is_own,
    own_behaviour,
    own_id,
)
from retro_ai_lab.rewards.suggestions import apply_suggestions, is_automatic
from retro_ai_lab.understanding import EventKind, Evidence, GameEvent, Knowledge, VariableKind

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]


def test_an_own_reward_is_built_and_kept() -> None:
    k = Knowledge("g")
    x = k.propose(RAM, 4, kind=VariableKind.POSITION, evidence=SURE)
    up = own_behaviour(x, "Your position", True, 2.0)
    down = own_behaviour(x, "Your position", False, -2.0)
    assert up.id == own_id(x.id, True) == f"mine:up:{x.id}" and is_own(up.id)
    assert not is_automatic(up.id) and not is_own("progress:v1")
    assert up.name == "Your position goes up" and down.name == "Your position goes down"
    assert up.trigger == Trigger(EventKind.UP, x.id)
    assert down.trigger == Trigger(EventKind.DOWN, x.id)
    assert up.category is Category.PROGRESSION and up.custom and up.notes == OWN_NOTE
    assert up.value == up.suggested == 2.0 and down.value == -2.0
    assert explain_behaviour(up) == (
        "Your position goes up: the AI gets 2 points each time. You added this. "
        "You set the value; the AI suggested +2."
    )
    # It earns on the matching move only, and the AI's own suggestions leave it alone.
    profile = RewardProfile("g")
    profile.add(up)
    profile.add(down)
    assert profile.reward_for(GameEvent(EventKind.UP, 1, x.id, "", 3, 5)) == 2.0
    assert profile.reward_for(GameEvent(EventKind.DOWN, 2, x.id, "", 5, 4)) == -2.0
    assert profile.reward_for(GameEvent(EventKind.UP, 3, "other", "", 0, 1)) == 0.0
    report = apply_suggestions(profile, k)
    assert up.id not in report.removed and profile.find(up.id) == up and profile.find(down.id)
    # Misuse.
    with pytest.raises(ConfigurationError):
        own_behaviour(x, "   ", True, 1.0)
    k.reject(x.id)
    with pytest.raises(ConfigurationError):
        own_behaviour(k.get(x.id), "Your position", True, 1.0)
