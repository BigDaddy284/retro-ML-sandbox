"""Everything the UI needs from the layers below, bundled once."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from retro_ai_lab.comparison import SavedAiService
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.core.paths import default_data_dir
from retro_ai_lab.emulator.registry import AdapterRegistry, default_registry
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    GoalService,
    ObservationService,
    RewardService,
)
from retro_ai_lab.storage import BrainStore, GoalStore, KnowledgeStore, RewardStore, SavedAiStore
from retro_ai_lab.training import StepLog, TrainingService
from retro_ai_lab.ui.bridge import EventBridge


@dataclass
class AppContext:
    """``data_dir`` is where knowledge and archives live (default: the user's app data)."""

    bus: EventBus = field(default_factory=EventBus)
    registry: AdapterRegistry = field(default_factory=default_registry)
    data_dir: Path = field(default_factory=default_data_dir)
    knowledge_store: KnowledgeStore = field(init=False)
    reward_store: RewardStore = field(init=False)
    goal_store: GoalStore = field(init=False)
    brain_store: BrainStore = field(init=False)
    saved_ai_store: SavedAiStore = field(init=False)
    connection: ConnectionService = field(init=False)
    observation: ObservationService = field(init=False)
    discovery: DiscoveryService = field(init=False)
    rewards: RewardService = field(init=False)
    goals: GoalService = field(init=False)
    training: TrainingService = field(init=False)
    saved_ais: SavedAiService = field(init=False)
    step_log: StepLog = field(init=False)
    bridge: EventBridge = field(init=False)

    def __post_init__(self) -> None:
        self.connection = ConnectionService(self.registry, self.bus)
        self.observation = ObservationService(self.connection, self.bus)
        self.knowledge_store = KnowledgeStore(self.data_dir / "knowledge")
        self.discovery = DiscoveryService(
            self.connection, self.observation, self.bus, store=self.knowledge_store
        )
        self.reward_store = RewardStore(self.data_dir / "rewards")
        self.rewards = RewardService(
            self.connection, self.discovery, self.bus, store=self.reward_store
        )
        self.goal_store = GoalStore(self.data_dir / "goals")
        self.goals = GoalService(self.connection, self.discovery, self.bus, store=self.goal_store)
        self.brain_store = BrainStore(self.data_dir / "brains")
        self.training = TrainingService(
            self.connection,
            self.observation,
            self.discovery,
            self.bus,
            store=self.brain_store,
            goals=self.goals,
            rewards=self.rewards,
        )
        self.saved_ai_store = SavedAiStore(self.data_dir / "saved_ais")
        self.saved_ais = SavedAiService(
            self.rewards, self.goals, self.training, self.bus, store=self.saved_ai_store
        )
        self.step_log = StepLog(self.training, self.bus)
        self.bridge = EventBridge(self.bus)


__all__ = ["AppContext"]
