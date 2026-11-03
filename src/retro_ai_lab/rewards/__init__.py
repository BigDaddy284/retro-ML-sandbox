"""Reward system (Phase 4): how children tell the AI what matters."""

from retro_ai_lab.rewards.exploitation import ExploitDetector, ExploitWarning
from retro_ai_lab.rewards.goals import (
    GOAL_IDEAS,
    Goal,
    GoalOutcome,
    Objective,
    merge_suggestions,
    suggest_goal,
)
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
from retro_ai_lab.rewards.tally import BehaviourTotal, RewardEntry, RewardTally

__all__ = [
    "GOAL_IDEAS",
    "MAX_VALUE",
    "PERSONALITIES",
    "SCHEMA_VERSION",
    "STEP",
    "Behaviour",
    "BehaviourTotal",
    "Category",
    "ExploitDetector",
    "ExploitWarning",
    "Goal",
    "GoalOutcome",
    "Objective",
    "Personality",
    "RewardEntry",
    "RewardProfile",
    "RewardTally",
    "SuggestionReport",
    "Trigger",
    "apply_personality",
    "apply_suggestions",
    "is_automatic",
    "merge_suggestions",
    "personality",
    "suggest_behaviours",
    "suggest_goal",
]
