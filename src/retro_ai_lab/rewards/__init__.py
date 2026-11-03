"""Reward system (Phase 4): how children tell the AI what matters."""

from retro_ai_lab.rewards.model import (
    MAX_VALUE,
    SCHEMA_VERSION,
    STEP,
    Behaviour,
    Category,
    RewardProfile,
    Trigger,
)
from retro_ai_lab.rewards.personalities import (
    PERSONALITIES,
    Personality,
    apply_personality,
    personality,
)
from retro_ai_lab.rewards.suggestions import (
    SuggestionReport,
    apply_suggestions,
    is_automatic,
    suggest_behaviours,
)

__all__ = [
    "MAX_VALUE",
    "PERSONALITIES",
    "SCHEMA_VERSION",
    "STEP",
    "Behaviour",
    "Category",
    "Personality",
    "RewardProfile",
    "SuggestionReport",
    "Trigger",
    "apply_personality",
    "apply_suggestions",
    "is_automatic",
    "personality",
    "suggest_behaviours",
]
