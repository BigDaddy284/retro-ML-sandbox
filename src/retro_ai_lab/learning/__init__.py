"""Learning (Phase 5+): how the AI gets better at the game."""

from retro_ai_lab.learning.actions import Action, ActionSpace
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
from retro_ai_lab.learning.state import State, StateEncoder, bucket

__all__ = [
    "Q_TABLE",
    "Action",
    "ActionSpace",
    "Decision",
    "Learner",
    "QLearningAgent",
    "QTable",
    "Settings",
    "State",
    "StateEncoder",
    "algorithms",
    "bucket",
    "learner_from_dict",
    "make_learner",
    "register_learner",
]
