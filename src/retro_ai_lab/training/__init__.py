"""Training (Phase 5+): running the learner against the game."""

from retro_ai_lab.training.brain import Brain
from retro_ai_lab.training.curve import CurvePoint, LearningCurve
from retro_ai_lab.training.episode import EpisodeResult, EpisodeRunner, EpisodeSignals
from retro_ai_lab.training.records import EpisodeRecord, Mode, Speed, Status
from retro_ai_lab.training.service import (
    EpisodeFinished,
    TrainingService,
    TrainingStateChanged,
    TrainingStepped,
)

__all__ = [
    "Brain",
    "CurvePoint",
    "EpisodeFinished",
    "EpisodeRecord",
    "EpisodeResult",
    "EpisodeRunner",
    "EpisodeSignals",
    "LearningCurve",
    "Mode",
    "Speed",
    "Status",
    "TrainingService",
    "TrainingStateChanged",
    "TrainingStepped",
]
