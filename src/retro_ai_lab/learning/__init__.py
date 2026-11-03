"""Learning (Phase 5+): how the AI gets better at the game."""

from retro_ai_lab.learning.actions import Action, ActionSpace
from retro_ai_lab.learning.features import FeatureMap
from retro_ai_lab.learning.learner import (
    Q_TABLE,
    Decision,
    Learner,
    algorithms,
    learner_from_dict,
    make_learner,
    register_learner,
)
from retro_ai_lab.learning.qlearning import QLearningAgent, QTable, Settings
from retro_ai_lab.learning.replay import ReplayBuffer, Transition
from retro_ai_lab.learning.state import DEFAULT_BINS, State, StateEncoder, bucket

__all__ = [
    "DEFAULT_BINS",
    "Q_TABLE",
    "Action",
    "ActionSpace",
    "Decision",
    "FeatureMap",
    "Learner",
    "QLearningAgent",
    "QTable",
    "ReplayBuffer",
    "Settings",
    "State",
    "StateEncoder",
    "Transition",
    "algorithms",
    "bucket",
    "learner_from_dict",
    "make_learner",
    "register_learner",
]
