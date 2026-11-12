"""The reward for getting somewhere new, measured off the screen (T-993).

The AI's other suggested rewards come from guesses about the game's memory.
This one needs no guess: while it plays, the AI's look at the screen counts
each step of the player's own size past the furthest it has been this go
(:mod:`retro_ai_lab.observation.ground`), and every step pays this row. The
row is added the first time ground is reported, with a suggested point a
step; a person can change it, switch it off or remove it like any other.
"""

from __future__ import annotations

from retro_ai_lab.rewards.model import Behaviour, Category, RewardProfile, Trigger
from retro_ai_lab.understanding.events import EventKind

# Not named after an event kind, so the suggestions made from memory leave the row alone.
GROUND_ID = "screen:new_ground"
GROUND_NAME = "Getting somewhere new"
GROUND_NOTE = (
    "Each step of its own size past the furthest it has been this go, measured off the screen."
)
GROUND_POINTS = 1.0


def ground_behaviour() -> Behaviour:
    return Behaviour(
        GROUND_ID,
        GROUND_NAME,
        Category.PROGRESSION,
        Trigger(EventKind.NEW_GROUND, None),
        suggested=GROUND_POINTS,
        value=GROUND_POINTS,
        notes=GROUND_NOTE,
    )


def ensure_ground_row(profile: RewardProfile) -> bool:
    """Add the row unless it is there or a person removed it (``profile.dismissed``,
    T-943). Says whether it was added. A row already there keeps its points and its
    switch."""
    if profile.find(GROUND_ID) is not None or GROUND_ID in profile.dismissed:
        return False
    profile.add(ground_behaviour())
    profile.reset_to_suggested(GROUND_ID)  # the profile's personality weighs it
    return True


__all__ = [
    "GROUND_ID",
    "GROUND_NAME",
    "GROUND_NOTE",
    "GROUND_POINTS",
    "ensure_ground_row",
    "ground_behaviour",
]
