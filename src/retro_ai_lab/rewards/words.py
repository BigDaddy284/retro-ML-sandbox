"""Plain words for rewards (T-859): what a behaviour means and what a category is."""

from __future__ import annotations

from retro_ai_lab.rewards.model import Behaviour, Category

CATEGORY_WORDS: dict[Category, str] = {
    Category.EXPLORATION: "finding new places",
    Category.COLLECTION: "picking up points and things",
    Category.COMBAT: "fighting",
    Category.SURVIVAL: "staying alive",
    Category.PROGRESSION: "getting further",
    Category.QUESTS: "finishing tasks",
    Category.EFFICIENCY: "doing it quickly",
    Category.COMPLETION: "finishing the game",
}

PICK_ONE = "Pick a reward to see what it means."


def describe_category(category: Category) -> str:
    return f"{category.label}: {CATEGORY_WORDS[category]}"


def _points(value: float) -> str:
    amount = f"{abs(value):g}"
    return f"{amount} point{'' if amount == '1' else 's'}"


def explain_behaviour(b: Behaviour) -> str:
    """One sentence a child can read: what it is, what it is worth, and any caveats."""
    if b.value > 0:
        worth = f"the AI gets {_points(b.value)} each time"
    elif b.value < 0:
        worth = f"the AI loses {_points(b.value)} each time"
    else:
        worth = "the AI gets nothing for it"
    parts = [f"{b.name}: {worth}."]
    if b.notes:
        parts.append(b.notes)
    if b.custom:
        parts.append(f"You set the value; the AI suggested {b.suggested:+g}.")
    if not b.enabled:
        parts.append("It is switched off, so it counts for nothing right now.")
    return " ".join(parts)


__all__ = ["CATEGORY_WORDS", "PICK_ONE", "describe_category", "explain_behaviour"]
