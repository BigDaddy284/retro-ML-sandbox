"""Rewards on the buttons themselves (T-874): a row per button, and one for repeats.

The AI has no opinion about buttons, so every row suggests 0; a person types
points above zero to encourage a button and below zero to put the AI off it.
The rows are scored every step from what the runner held, so "Pressing A"
pays each step A is held and "Pressing the same button again" each step a
button is held that was held the step before.
"""

from __future__ import annotations

from collections.abc import Iterable

from retro_ai_lab.core.types import Button
from retro_ai_lab.rewards.model import Behaviour, Category, RewardProfile, Trigger
from retro_ai_lab.understanding.events import EventKind

BUTTON = "button"  # the id prefix; not an event kind, so suggestions leave the rows alone
PRESS_NOTE = "Each step the button is held. Points below zero put the AI off it."
AGAIN_ID = f"{BUTTON}:again"
AGAIN_NAME = "Pressing the same button again"
AGAIN_NOTE = "Each step a button is held that was held the step before."


def press_id(name: str) -> str:
    return f"{BUTTON}:press:{name}"


def is_button(behaviour_id: str) -> bool:
    return behaviour_id.startswith(BUTTON + ":")


def press_behaviour(button: Button) -> Behaviour:
    return Behaviour(
        press_id(button.name),
        f"Pressing {button.label}",
        Category.CONTROLS,
        Trigger(EventKind.PRESS, button.name),
        notes=PRESS_NOTE,
    )


def again_behaviour() -> Behaviour:
    return Behaviour(
        AGAIN_ID, AGAIN_NAME, Category.CONTROLS, Trigger(EventKind.REPEAT, None), notes=AGAIN_NOTE
    )


def button_behaviours(buttons: Iterable[Button]) -> tuple[Behaviour, ...]:
    """The rows a game with these buttons gets: one per button, then the repeat row."""
    return (*(press_behaviour(b) for b in buttons), again_behaviour())


def ensure_button_rows(
    profile: RewardProfile, buttons: Iterable[Button]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Add the rows the game's controller calls for and drop those for buttons it lacks.

    A row already there keeps its points and its switch; a row a person removed
    (``profile.dismissed``, T-943) is not added back. Returns (added, removed).
    """
    wanted = {b.id: b for b in button_behaviours(buttons) if b.id not in profile.dismissed}
    added = tuple(bid for bid in wanted if profile.find(bid) is None)
    for bid in added:
        profile.add(wanted[bid])
    removed = tuple(bid for bid in profile.ids() if is_button(bid) and bid not in wanted)
    for bid in removed:
        profile.remove(bid)
    return added, removed


__all__ = [
    "AGAIN_ID",
    "AGAIN_NAME",
    "AGAIN_NOTE",
    "BUTTON",
    "PRESS_NOTE",
    "again_behaviour",
    "button_behaviours",
    "ensure_button_rows",
    "is_button",
    "press_behaviour",
    "press_id",
]
