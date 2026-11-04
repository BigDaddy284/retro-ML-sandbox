"""Session orchestration: connect, observe, train. No Qt below this line."""

from retro_ai_lab.services.connection import (
    LINK_LOST,
    ConnectionFailed,
    ConnectionService,
    ConnectionState,
    ConnectionStateChanged,
    EmulatorConnected,
    EmulatorDisconnected,
    Session,
)
from retro_ai_lab.services.discovery import DiscoveryProgress, DiscoveryService, DiscoveryUpdated
from retro_ai_lab.services.experiments import Comparison, ExperimentHarness, Trial, constant_plan
from retro_ai_lab.services.goals import (
    GoalChanged,
    GoalService,
    ObjectiveCompleted,
    ObjectiveSetback,
    TriggerOption,
)
from retro_ai_lab.services.observer import (
    ButtonsHeld,
    InputRecorded,
    MemoryObserved,
    ObservationService,
    ScreenObserved,
)
from retro_ai_lab.services.rewards import (
    ExploitSuspected,
    GoStarted,
    RewardEarned,
    RewardProfileChanged,
    RewardService,
)

__all__ = [
    "LINK_LOST",
    "ButtonsHeld",
    "Comparison",
    "ConnectionFailed",
    "ConnectionService",
    "ConnectionState",
    "ConnectionStateChanged",
    "DiscoveryProgress",
    "DiscoveryService",
    "DiscoveryUpdated",
    "EmulatorConnected",
    "EmulatorDisconnected",
    "ExperimentHarness",
    "ExploitSuspected",
    "GoStarted",
    "GoalChanged",
    "GoalService",
    "InputRecorded",
    "MemoryObserved",
    "ObjectiveCompleted",
    "ObjectiveSetback",
    "ObservationService",
    "RewardEarned",
    "RewardProfileChanged",
    "RewardService",
    "ScreenObserved",
    "Session",
    "Trial",
    "TriggerOption",
    "constant_plan",
]
