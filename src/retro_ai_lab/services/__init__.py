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
from retro_ai_lab.services.observer import (
    InputRecorded,
    MemoryObserved,
    ObservationService,
    ScreenObserved,
)
from retro_ai_lab.services.rewards import RewardEarned, RewardProfileChanged, RewardService

__all__ = [
    "LINK_LOST",
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
    "InputRecorded",
    "MemoryObserved",
    "ObservationService",
    "RewardEarned",
    "RewardProfileChanged",
    "RewardService",
    "ScreenObserved",
    "Session",
    "Trial",
    "constant_plan",
]
