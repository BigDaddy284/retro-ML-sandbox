"""Goal service: the live goal and objectives for the connected game.

Keeps one :class:`Goal` per game (loaded from the :class:`GoalStore` or
started fresh), lets the AI add suggested objectives as it learns
(DESIGN.md §4.1), ticks objectives off as the game's events show them done
and applies setback penalties (§4.2), and exposes the edits a child makes
to the objective structure (§4.3) so they are saved and announced.

Objective rewards are announced as :class:`ObjectiveCompleted` and
:class:`ObjectiveSetback`, separately from behaviour rewards
(:class:`RewardEarned`), so the learner can weigh both.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_ai_lab.core import ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.rewards.goals import DEFAULT_REWARD, Goal, Objective, merge_suggestions
from retro_ai_lab.rewards.model import Trigger
from retro_ai_lab.rewards.suggestions import RESET_KEY, RULES, RelationshipWatch
from retro_ai_lab.services.connection import (
    ConnectionService,
    EmulatorConnected,
    EmulatorDisconnected,
)
from retro_ai_lab.services.discovery import DiscoveryService, DiscoveryUpdated, game_id_for
from retro_ai_lab.storage.goals import GoalStore
from retro_ai_lab.understanding.events import EventKind, GameEvent
from retro_ai_lab.understanding.knowledge import Confidence


@dataclass(frozen=True, slots=True)
class GoalChanged:
    reason: str  # "loaded", "suggestions", "edit", "progress"


@dataclass(frozen=True, slots=True)
class ObjectiveCompleted:
    frame: int
    objective: Objective
    reward: float


@dataclass(frozen=True, slots=True)
class ObjectiveSetback:
    frame: int
    objective: Objective
    penalty: float


@dataclass(frozen=True, slots=True)
class TriggerOption:
    """A detection method a child can pick, in plain words."""

    trigger: Trigger
    label: str

    @property
    def key(self) -> str:
        return self.trigger.key


class GoalService:
    def __init__(
        self,
        connection: ConnectionService,
        discovery: DiscoveryService,
        bus: EventBus,
        *,
        store: GoalStore | None = None,
    ) -> None:
        self._connection = connection
        self._discovery = discovery
        self._bus = bus
        self._store = store
        self._goal: Goal | None = None
        self._enabled = False
        self._saves = 0
        self._watch = RelationshipWatch()
        bus.subscribe(EmulatorConnected, self._on_connected)
        bus.subscribe(EmulatorDisconnected, self._on_disconnected)
        bus.subscribe(DiscoveryUpdated, self._on_discovery)

    # ------------------------------------------------------------ queries

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def goal(self) -> Goal | None:
        """The connected game's goal (kept readable after disconnect)."""
        return self._goal

    @property
    def store(self) -> GoalStore | None:
        return self._store

    @property
    def saves(self) -> int:
        return self._saves

    def summary(self) -> str:
        if self._goal is None or not self._enabled:
            return "No goal yet."
        done, total = self._goal.progress()
        if total == 0:
            return f"{self._goal.title}: no steps yet."
        head = f"{self._goal.title}: {done} of {total} step{'s' if total != 1 else ''} done."
        if self._goal.is_complete:
            return head + " All done!"
        nxt = self._goal.next()
        return head + (f" Next: {nxt.description}." if nxt else " Nothing to do next yet.")

    def trigger_options(self) -> tuple[TriggerOption, ...]:
        """Detection methods the AI can offer, from what it believes about the game."""
        knowledge = self._discovery.knowledge
        options: list[TriggerOption] = []
        if knowledge is not None:
            for var in knowledge.believed(Confidence.LOW):
                for (event_kind, var_kind), (_category, _value, template) in RULES.items():
                    if var_kind is var.kind:
                        options.append(
                            TriggerOption(
                                Trigger(event_kind, var.id),
                                template.format(name=var.display_name),
                            )
                        )
        options.append(TriggerOption(Trigger(EventKind.RESET), "Starting over"))
        assert options[-1].key == RESET_KEY
        return tuple(options)

    # ------------------------------------------------------------ edits

    def set_title(self, title: str) -> None:
        goal = self._require()
        if not title or not title.strip():
            raise ConfigurationError("The goal needs a name.")
        goal.title = title.strip()
        self._announce("edit")

    def add_objective(
        self,
        description: str,
        trigger: Trigger | None = None,
        *,
        reward: float = DEFAULT_REWARD,
        penalty: float = 0.0,
        setback_trigger: Trigger | None = None,
        depends_on: Iterable[str] = (),
    ) -> Objective:
        goal = self._require()
        objective = Objective(
            goal.new_id(),
            description,
            trigger,
            reward=reward,
            penalty=penalty,
            setback_trigger=setback_trigger,
            depends_on=tuple(depends_on),
        )
        goal.add(objective)
        self._announce("edit")
        return objective

    def update_objective(self, objective_id: str, **changes: Any) -> Objective:
        updated = self._require().update(objective_id, **changes)
        self._announce("edit")
        return updated

    def remove_objective(self, objective_id: str) -> None:
        self._require().remove(objective_id)
        self._announce("edit")

    def move_objective(self, objective_id: str, position: int) -> None:
        self._require().move(objective_id, position)
        self._announce("edit")

    def set_dependencies(self, objective_id: str, depends_on: Iterable[str]) -> Objective:
        updated = self._require().set_dependencies(objective_id, depends_on)
        self._announce("edit")
        return updated

    def mark_complete(self, objective_id: str) -> Objective:
        done = self._require().complete(objective_id)
        self._announce("progress")
        return done

    def mark_not_complete(self, objective_id: str) -> Objective:
        undone = self._require().uncomplete(objective_id)
        self._announce("progress")
        return undone

    def reset_progress(self) -> None:
        self._require().reset_progress()
        self._announce("progress")

    def refresh_suggestions(self) -> tuple[str, ...]:
        goal = self._require()
        knowledge = self._discovery.knowledge
        if knowledge is None or knowledge.game_id != goal.game_id:
            return ()
        added = merge_suggestions(goal, knowledge)
        if added:
            self._announce("suggestions")
        return added

    def replace(self, goal: Goal) -> None:
        """Swap in another goal for the same game (loading a saved AI)."""
        current = self._require()
        if goal.game_id != current.game_id:
            raise ConfigurationError(f"That goal is for {goal.game_id!r}, not the connected game.")
        self._goal = goal
        self._announce("loaded")

    def save(self) -> Path | None:
        if self._store is None or self._goal is None:
            return None
        path = self._store.save(self._goal)
        self._saves += 1
        return path

    # ------------------------------------------------------------- events

    def _on_connected(self, event: EmulatorConnected) -> None:
        info = event.game_info
        game_id = game_id_for(info.system, info.title, info.rom_hash)
        self._enabled = event.capabilities.memory_read
        if not self._enabled:
            self._goal = None
            return
        title = f"Get as far as you can in {info.title}"
        if self._store is not None:
            try:
                self._goal = self._store.load_or_new(game_id, title=title)
            except ConfigurationError:
                self._goal = Goal(game_id, title=title)  # a damaged file: start over
        else:
            self._goal = Goal(game_id, title=title)
        self._watch.reset()
        self._bus.publish(GoalChanged("loaded"))
        self.refresh_suggestions()

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        if self._enabled:
            self.save()
        self._enabled = False

    def _on_discovery(self, update: DiscoveryUpdated) -> None:
        if not self._enabled or self._goal is None:
            return
        if update.added or update.changed or self._relationships_moved(update):
            self.refresh_suggestions()
        progressed = False
        for game_event in update.events:
            progressed = self._observe(game_event) or progressed
        if progressed:
            self._announce("progress")

    def _relationships_moved(self, update: DiscoveryUpdated) -> bool:
        """Only a change among the confident relationships is worth a rebuild (T-842)."""
        knowledge = self._discovery.knowledge
        if not update.relationships or knowledge is None:
            return False
        return self._watch.changed(knowledge)

    def _observe(self, game_event: GameEvent) -> bool:
        assert self._goal is not None
        outcome = self._goal.observe(game_event)
        for objective in outcome.penalised:
            self._bus.publish(ObjectiveSetback(game_event.frame, objective, objective.penalty))
        for objective in outcome.completed:
            self._bus.publish(ObjectiveCompleted(game_event.frame, objective, objective.reward))
        return not outcome.is_empty

    # ------------------------------------------------------------ helpers

    def _require(self) -> Goal:
        if self._goal is None or not self._enabled:
            raise UnsupportedCapabilityError(
                "memory_read", "Connect to a game that shares its memory first."
            )
        return self._goal

    def _announce(self, reason: str) -> None:
        self.save()
        self._bus.publish(GoalChanged(reason))


__all__ = [
    "GoalChanged",
    "GoalService",
    "ObjectiveCompleted",
    "ObjectiveSetback",
    "TriggerOption",
]
