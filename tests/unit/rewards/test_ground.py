"""T-993: the reward row for getting somewhere new, measured off the screen."""

from __future__ import annotations

from retro_ai_lab.rewards import (
    GROUND_ID,
    GROUND_NAME,
    GROUND_POINTS,
    Category,
    RewardProfile,
    ensure_ground_row,
    ground_behaviour,
    is_button,
)
from retro_ai_lab.rewards.exploitation import PROGRESS_CATEGORIES
from retro_ai_lab.rewards.own import is_own
from retro_ai_lab.rewards.suggestions import is_automatic
from retro_ai_lab.understanding.events import EventKind, GameEvent


def test_the_row() -> None:
    row = ground_behaviour()
    assert (row.id, row.name) == (GROUND_ID, GROUND_NAME) == ("screen:new_ground", row.name)
    assert row.name == "Getting somewhere new" and "off the screen" in row.notes
    assert row.category is Category.PROGRESSION and row.category in PROGRESS_CATEGORIES
    assert row.suggested == row.value == GROUND_POINTS == 1.0 and row.enabled and not row.custom
    assert row.trigger.matches(GameEvent(EventKind.NEW_GROUND, 1, None, ""))
    assert not row.trigger.matches(GameEvent(EventKind.PROGRESS, 1, None, ""))
    # Not a guess from memory, not a button, not a person's own: suggestions leave it be.
    assert not is_automatic(row.id) and not is_button(row.id) and not is_own(row.id)


def test_the_row_is_added_once_and_not_when_dismissed() -> None:
    profile = RewardProfile("g")
    assert ensure_ground_row(profile) and GROUND_ID in profile
    assert profile.get(GROUND_ID).value == 1.0
    # A person's points and switch stay; nothing is added twice.
    profile.set_value(GROUND_ID, 3.0)
    profile.set_enabled(GROUND_ID, False)
    assert not ensure_ground_row(profile)
    assert profile.get(GROUND_ID).value == 3.0 and not profile.get(GROUND_ID).enabled
    # Removed by a person it stays out, until they ask for suggestions again.
    profile.dismiss(GROUND_ID)
    profile.remove(GROUND_ID)
    assert not ensure_ground_row(profile) and GROUND_ID not in profile
    profile.bring_back()
    assert ensure_ground_row(profile) and profile.get(GROUND_ID).value == 1.0
    # It is saved and loaded with the rest.
    again = RewardProfile.from_dict(profile.to_dict())
    assert again.get(GROUND_ID) == profile.get(GROUND_ID)
