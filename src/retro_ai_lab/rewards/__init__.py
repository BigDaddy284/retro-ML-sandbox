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
from retro_ai_lab.rewards.suggestions import (
    SuggestionReport,
    apply_suggestions,
    is_automatic,
    suggest_behaviours,
)

__all__ = [
    "MAX_VALUE",
    "SCHEMA_VERSION",
    "STEP",
    "Behaviour",
    "Category",
    "RewardProfile",
    "SuggestionReport",
    "Trigger",
    "apply_suggestions",
    "is_automatic",
    "suggest_behaviours",
]
