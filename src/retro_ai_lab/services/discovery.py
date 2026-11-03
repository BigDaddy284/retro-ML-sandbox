"""Discovery service: the automatic observe → think → experiment loop.

This is the Game Understanding Engine's conductor (DESIGN.md §3.4). While
a game is connected it:

* feeds every memory snapshot to the event detector and relationship
  finder, so the knowledge's story grows as the child plays;
* every ``review_every`` frames, turns the change statistics into new
  variable proposals (``find_candidates`` → ``propose_candidates``);
* when the emulator supports save states, every ``experiment_every``
  frames runs the button-linking experiments and folds them in, from
  whatever situation the game is in at that moment.

Everything it learns goes into :attr:`knowledge`, and every step is
announced on the bus (:class:`DiscoveryProgress`, :class:`DiscoveryUpdated`)
so the UI can show what the AI is doing.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_ai_lab.core import EmulatorError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.services.connection import (
    ConnectionService,
    EmulatorConnected,
    EmulatorDisconnected,
)
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.services.observer import MemoryObserved, ObservationService
from retro_ai_lab.understanding.events import EventDetector, EventLog, GameEvent
from retro_ai_lab.understanding.knowledge import Knowledge, Relationship, Variable
from retro_ai_lab.understanding.linking import apply_input_links, find_input_links
from retro_ai_lab.understanding.relationships import RelationshipFinder
from retro_ai_lab.understanding.variables import find_candidates, propose_candidates

DEFAULT_REVIEW_EVERY = 120
DEFAULT_EXPERIMENT_EVERY = 600


@dataclass(frozen=True, slots=True)
class DiscoveryProgress:
    """What the engine is doing right now, in plain words."""

    stage: str  # "idle", "watching", "thinking", "experimenting"
    detail: str


@dataclass(frozen=True, slots=True)
class DiscoveryUpdated:
    """Something in the knowledge changed."""

    added: tuple[Variable, ...]
    changed: tuple[Variable, ...]
    events: tuple[GameEvent, ...]
    relationships: tuple[Relationship, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.changed or self.events or self.relationships)


def game_id_for(system: str, title: str, rom_hash: str | None) -> str:
    """A stable id for a game: its hash when the emulator gives one."""
    return rom_hash or f"{system}:{title}"


class DiscoveryService:
    def __init__(
        self,
        connection: ConnectionService,
        observation: ObservationService,
        bus: EventBus,
        *,
        review_every: int = DEFAULT_REVIEW_EVERY,
        experiment_every: int | None = DEFAULT_EXPERIMENT_EVERY,
        event_window: int = 5,
    ) -> None:
        if review_every < 1:
            raise ValueError("review_every must be positive")
        if experiment_every is not None and experiment_every < 1:
            raise ValueError("experiment_every must be positive or None")
        self._connection = connection
        self._observation = observation
        self._bus = bus
        self._review_every = review_every
        self._experiment_every = experiment_every
        self._event_window = event_window
        self._knowledge: Knowledge | None = None
        self._detector: EventDetector | None = None
        self._finder: RelationshipFinder | None = None
        self._events = EventLog()
        self._harness: ExperimentHarness | None = None
        self._buttons: tuple[str, ...] = ()
        self._since_review = 0
        self._since_experiment = 0
        self._experiments = 0
        self._reviews = 0
        self._stage = "idle"
        self._enabled = False
        bus.subscribe(EmulatorConnected, self._on_connected)
        bus.subscribe(EmulatorDisconnected, self._on_disconnected)
        bus.subscribe(MemoryObserved, self._on_memory)

    # ------------------------------------------------------------ queries

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def knowledge(self) -> Knowledge | None:
        """What has been learned about the connected game (None when idle)."""
        return self._knowledge

    @property
    def events(self) -> EventLog:
        return self._events

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def reviews(self) -> int:
        return self._reviews

    @property
    def experiments(self) -> int:
        return self._experiments

    def can_experiment(self) -> bool:
        return self._harness is not None and self._harness.available()

    def summary(self) -> str:
        """One line for the status bar / panel header."""
        k = self._knowledge
        if k is None or not self._enabled:
            return "Not watching a game."
        beliefs = sum(1 for v in k if not v.rejected)
        confident = len(k.confident())
        asks = len(k.needs_help())
        parts = [f"Watching {beliefs} value{'s' if beliefs != 1 else ''}"]
        if confident:
            parts.append(f"{confident} fairly sure")
        if asks:
            parts.append(f"{asks} could use your help")
        return ", ".join(parts) + "."

    # ------------------------------------------------------------ actions

    def review(self) -> DiscoveryUpdated:
        """Turn the statistics gathered so far into proposals, right now."""
        k = self._require()
        self._set_stage(
            "thinking", "Looking for values that behave like scores, lives or counters."
        )
        added = propose_candidates(k, find_candidates(self._observation.tracker))
        self._reviews += 1
        self._since_review = 0
        update = DiscoveryUpdated(added, (), (), ())
        self._announce(update)
        self._set_stage("watching", self.summary())
        return update

    def experiment(self) -> DiscoveryUpdated:
        """Try every button from the current situation and learn what each does."""
        k = self._require()
        if self._harness is None or not self._harness.available():
            raise UnsupportedCapabilityError(
                "save_states", "This emulator can't run experiments (no save states)."
            )
        self._set_stage("experimenting", "Trying each button to see what it changes.")
        self._harness.checkpoint()
        try:
            links = find_input_links(self._harness, self._buttons)
        finally:
            self._since_experiment = 0
        touched = apply_input_links(k, links)
        self._experiments += 1
        added = tuple(v for v in touched if not any(e.kind != "input-linked" for e in v.evidence))
        changed = tuple(v for v in touched if v not in added)
        update = DiscoveryUpdated(added, changed, (), ())
        self._announce(update)
        self._set_stage("watching", self.summary())
        return update

    # ------------------------------------------------------------- events

    def _on_connected(self, event: EmulatorConnected) -> None:
        info = event.game_info
        self._knowledge = Knowledge(
            game_id_for(info.system, info.title, info.rom_hash), title=info.title
        )
        self._detector = EventDetector(self._knowledge)
        self._finder = RelationshipFinder(self._knowledge, window=self._event_window)
        self._events.clear()
        self._buttons = tuple(b.name for b in event.buttons)
        self._harness = ExperimentHarness(self._connection)
        self._since_review = 0
        self._since_experiment = 0
        self._experiments = 0
        self._reviews = 0
        self._enabled = event.capabilities.memory_read
        if self._enabled:
            self._set_stage("watching", f"Watching {info.title} to learn how it works.")
        else:
            self._set_stage(
                "idle", "This emulator can't share its memory, so there is nothing to learn from."
            )

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._enabled = False
        self._harness = None
        self._set_stage("idle", "Not watching a game.")

    def _on_memory(self, event: MemoryObserved) -> None:
        if not self._enabled or self._knowledge is None:
            return
        assert self._detector is not None and self._finder is not None
        events = self._detector.observe(event.snapshot)
        relationships: tuple[Relationship, ...] = ()
        if events:
            self._events.extend(events)
            relationships = self._finder.observe(events)
        if events or relationships:
            self._announce(DiscoveryUpdated((), (), events, relationships))
        self._since_review += 1
        self._since_experiment += 1
        if self._since_review >= self._review_every:
            self.review()
        if (
            self._experiment_every is not None
            and self._since_experiment >= self._experiment_every
            and self.can_experiment()
        ):
            try:
                self.experiment()
            except EmulatorError:
                # Link loss is announced by the connection service; carry on.
                self._since_experiment = 0

    # ------------------------------------------------------------ helpers

    def _require(self) -> Knowledge:
        if self._knowledge is None or not self._enabled:
            raise UnsupportedCapabilityError(
                "memory_read", "Connect to a game that shares its memory first."
            )
        return self._knowledge

    def _set_stage(self, stage: str, detail: str) -> None:
        self._stage = stage
        self._bus.publish(DiscoveryProgress(stage, detail))

    def _announce(self, update: DiscoveryUpdated) -> None:
        if not update.is_empty:
            self._bus.publish(update)


__all__ = [
    "DEFAULT_EXPERIMENT_EVERY",
    "DEFAULT_REVIEW_EVERY",
    "DiscoveryProgress",
    "DiscoveryService",
    "DiscoveryUpdated",
    "game_id_for",
]
