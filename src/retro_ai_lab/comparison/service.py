"""Save the connected AI under a name, load a saved one back, delete one.

The three parts of an AI belong to three services; this one only asks them
for their current state and hands a saved state back through their own
``replace`` entry points, so every invariant stays where it is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_ai_lab.comparison.saved_ai import SavedAi, check_name
from retro_ai_lab.core import ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.rewards.goals import Goal
from retro_ai_lab.rewards.model import RewardProfile
from retro_ai_lab.services.goals import GoalService
from retro_ai_lab.services.rewards import RewardService
from retro_ai_lab.storage.saved_ais import SavedAiStore
from retro_ai_lab.training.records import Status
from retro_ai_lab.training.service import TrainingService


@dataclass(frozen=True, slots=True)
class SavedAiChanged:
    game_id: str
    name: str
    reason: str  # "saved", "loaded", "deleted"


class SavedAiService:
    def __init__(
        self,
        rewards: RewardService,
        goals: GoalService,
        training: TrainingService,
        bus: EventBus,
        *,
        store: SavedAiStore,
    ) -> None:
        self._rewards = rewards
        self._goals = goals
        self._training = training
        self._bus = bus
        self._store = store

    # ------------------------------------------------------------ queries

    @property
    def store(self) -> SavedAiStore:
        return self._store

    @property
    def enabled(self) -> bool:
        """True while a game is connected and has rewards, a goal and a brain."""
        return (
            self._rewards.enabled
            and self._goals.enabled
            and self._training.enabled
            and self._training.game_id is not None
        )

    @property
    def game_id(self) -> str | None:
        return self._training.game_id if self.enabled else None

    def names(self) -> tuple[str, ...]:
        game_id = self.game_id
        return self._store.names(game_id) if game_id is not None else ()

    def get(self, name: str) -> SavedAi:
        return self._load(self._require(), name)

    # ------------------------------------------------------------ actions

    def save_as(self, name: str, notes: str = "") -> SavedAi:
        """Keep the connected AI under ``name`` (replacing a saved AI of that name)."""
        game_id = self._require()
        name = check_name(name)
        profile = self._rewards.profile
        goal = self._goals.goal
        brain = self._training.brain()
        if profile is None or goal is None or brain is None:  # pragma: no cover - guarded
            raise ConfigurationError("There is no AI to save yet.")
        saved = SavedAi(
            name, game_id, profile.to_dict(), goal.to_dict(), brain.to_dict(), notes=notes
        )
        self._store.save(saved.to_dict())
        self._bus.publish(SavedAiChanged(game_id, name, "saved"))
        return saved

    def load(self, name: str) -> SavedAi:
        """Make a saved AI the connected one: rewards, goal and brain together."""
        game_id = self._require()
        if self._training.status is not Status.IDLE:
            raise ConfigurationError("Stop training before loading another AI.")
        saved = self._load(game_id, name)
        profile = RewardProfile.from_dict(saved.rewards)
        goal = Goal.from_dict(saved.goal)
        # The brain is the part most likely to be refused (other controls), so
        # it goes first: a refusal then leaves rewards and goal untouched.
        self._training.replace_brain(saved.brain)
        self._rewards.replace(profile)
        self._goals.replace(goal)
        self._bus.publish(SavedAiChanged(game_id, saved.name, "loaded"))
        return saved

    def delete(self, name: str) -> bool:
        game_id = self._require()
        if not self._store.delete(game_id, name):
            return False
        self._bus.publish(SavedAiChanged(game_id, name, "deleted"))
        return True

    # ------------------------------------------------------------ helpers

    def _load(self, game_id: str, name: str) -> SavedAi:
        saved = SavedAi.from_dict(self._store.load(game_id, name))
        if saved.game_id != game_id:
            raise ConfigurationError(f"The saved AI {name!r} is for another game.")
        return saved

    def _require(self) -> str:
        game_id = self.game_id
        if game_id is None:
            raise UnsupportedCapabilityError(
                "memory_read", "Connect to a game the AI can learn first."
            )
        return game_id


__all__ = ["SavedAiChanged", "SavedAiService"]
