"""Reward system (Phase 4): how children tell the AI what matters."""

from retro_ai_lab.rewards.buttons import (
    AGAIN_ID,
    AGAIN_NAME,
    button_behaviours,
    ensure_button_rows,
    is_button,
    press_id,
)
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
from retro_ai_lab.rewards.own import MINE, OWN_NOTE, is_own, own_behaviour, own_id
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
from retro_ai_lab.rewards.words import (
    CATEGORY_WORDS,
    PICK_ONE,
    describe_category,
    explain_behaviour,
)

__all__ = [
    "AGAIN_ID",
    "AGAIN_NAME",
    "CATEGORY_WORDS",
    "GOAL_IDEAS",
    "MAX_VALUE",
    "MINE",
    "OWN_NOTE",
    "PERSONALITIES",
    "PICK_ONE",
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
    "button_behaviours",
    "describe_category",
    "ensure_button_rows",
    "explain_behaviour",
    "is_automatic",
    "is_button",
    "is_own",
    "merge_suggestions",
    "own_behaviour",
    "own_id",
    "personality",
    "press_id",
    "suggest_behaviours",
    "suggest_goal",
]
