"""Learning (Phase 5+): how the AI gets better at the game."""

from retro_ai_lab.learning.actions import Action, ActionSpace
from retro_ai_lab.learning.demonstrations import Demonstrations
from retro_ai_lab.learning.dqn import DQN, DQNConfig, DQNLearner
from retro_ai_lab.learning.features import FeatureMap
from retro_ai_lab.learning.learner import (
    AUTO,
    Q_TABLE,
    Decision,
    Learner,
    algorithm_choices,
    algorithms,
    describe_algorithm,
    learner_from_dict,
    make_learner,
    register_learner,
)
from retro_ai_lab.learning.network import Network
from retro_ai_lab.learning.picture import (
    NO_MOVE,
    PICTURE_HEIGHT,
    PICTURE_SIZE,
    PICTURE_WIDTH,
    Sight,
    shrink,
)
from retro_ai_lab.learning.promotion import carries_over, convert, demote, promote
from retro_ai_lab.learning.qlearning import QLearningAgent, QTable, Settings
from retro_ai_lab.learning.replay import ReplayBuffer, Transition
from retro_ai_lab.learning.screen import SCREEN, ScreenConfig, ScreenLearner
from retro_ai_lab.learning.skills import SKILLS, SkillsLearner
from retro_ai_lab.learning.state import DEFAULT_BINS, State, StateEncoder, bucket

__all__ = [
    "AUTO",
    "DEFAULT_BINS",
    "DQN",
    "NO_MOVE",
    "PICTURE_HEIGHT",
    "PICTURE_SIZE",
    "PICTURE_WIDTH",
    "Q_TABLE",
    "SCREEN",
    "SKILLS",
    "Action",
    "ActionSpace",
    "DQNConfig",
    "DQNLearner",
    "Decision",
    "Demonstrations",
    "FeatureMap",
    "Learner",
    "Network",
    "QLearningAgent",
    "QTable",
    "ReplayBuffer",
    "ScreenConfig",
    "ScreenLearner",
    "Settings",
    "Sight",
    "SkillsLearner",
    "State",
    "StateEncoder",
    "Transition",
    "algorithm_choices",
    "algorithms",
    "bucket",
    "carries_over",
    "convert",
    "demote",
    "describe_algorithm",
    "learner_from_dict",
    "make_learner",
    "promote",
    "register_learner",
    "shrink",
]
