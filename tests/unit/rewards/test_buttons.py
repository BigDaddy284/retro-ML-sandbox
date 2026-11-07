"""T-874: a reward row for every button of the connected game, and one for repeats."""

from __future__ import annotations

from retro_ai_lab.core.types import Button
from retro_ai_lab.rewards import (
    AGAIN_ID,
    AGAIN_NAME,
    Category,
    RewardProfile,
    button_behaviours,
    ensure_button_rows,
    is_button,
    press_id,
)
from retro_ai_lab.rewards.suggestions import is_automatic
from retro_ai_lab.understanding.events import EventKind, GameEvent

PAD = (Button("a", "A"), Button("up", "Up"))


def test_button_rows_follow_the_controller() -> None:
    rows = button_behaviours(PAD)
    assert [b.id for b in rows] == [press_id("a"), press_id("up"), AGAIN_ID]
    assert [b.name for b in rows] == ["Pressing A", "Pressing Up", AGAIN_NAME]
    assert all(b.category is Category.CONTROLS and b.value == 0 == b.suggested for b in rows)
    assert all(is_button(b.id) and not is_automatic(b.id) for b in rows)  # suggestions leave them
    assert rows[0].trigger.matches(GameEvent(EventKind.PRESS, 1, "a", ""))
    assert not rows[0].trigger.matches(GameEvent(EventKind.PRESS, 1, "up", ""))
    assert rows[2].trigger.matches(GameEvent(EventKind.REPEAT, 1, "up", ""))  # any button
    profile = RewardProfile("g")
    added, removed = ensure_button_rows(profile, PAD)
    assert added == (press_id("a"), press_id("up"), AGAIN_ID) and removed == ()
    profile.set_value(press_id("up"), 2.5)
    profile.set_enabled(AGAIN_ID, False)
    # A smaller pad next time: Up's points and the switch stay, A's row goes, nothing is re-added.
    added, removed = ensure_button_rows(profile, (Button("up", "Up"), Button("b", "B")))
    assert added == (press_id("b"),) and removed == (press_id("a"),)
    assert profile.get(press_id("up")).value == 2.5 and not profile.get(AGAIN_ID).enabled
    assert ensure_button_rows(profile, (Button("up", "Up"), Button("b", "B"))) == ((), ())


def test_a_dismissed_button_row_is_not_added_back() -> None:
    """T-943: a button row a person removed stays out when the game connects again."""
    profile = RewardProfile("g")
    ensure_button_rows(profile, PAD)
    profile.dismiss(press_id("up"))
    profile.remove(press_id("up"))
    added, removed = ensure_button_rows(profile, PAD)
    assert added == () and removed == () and press_id("up") not in profile
    profile.bring_back()
    added, _ = ensure_button_rows(profile, PAD)
    assert added == (press_id("up"),)
