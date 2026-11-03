"""Reward service: the live reward profile for the connected game.

While a game is connected it keeps one :class:`RewardProfile` (loaded from
the :class:`RewardStore` or started fresh), fills it from the knowledge as
discoveries arrive (DESIGN.md §5.4), turns every detected event into a
reward (:class:`RewardEarned`) that the learner and the UI can use, keeps
a :class:`RewardTally`, and offers the child's controls, which save the
profile and announce :class:`RewardProfileChanged`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from retro_ai_lab.core import ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.rewards.exploitation import ExploitDetector, ExploitWarning
from retro_ai_lab.rewards.model import Behaviour, RewardProfile
from retro_ai_lab.rewards.personalities import Personality, apply_personality
from retro_ai_lab.rewards.suggestions import SuggestionReport, apply_suggestions
from retro_ai_lab.rewards.tally import RewardTally
from retro_ai_lab.services.connection import (
    ConnectionService,
    EmulatorConnected,
    EmulatorDisconnected,
)
from retro_ai_lab.services.discovery import DiscoveryService, DiscoveryUpdated, game_id_for
from retro_ai_lab.storage.rewards import RewardStore
from retro_ai_lab.understanding.events import GameEvent


@dataclass(frozen=True, slots=True)
class RewardEarned:
    frame: int
    amount: float
    behaviour_ids: tuple[str, ...]
    event: GameEvent


@dataclass(frozen=True, slots=True)
class RewardProfileChanged:
    """The profile changed: new suggestions, or a person used a control."""

    added: tuple[str, ...]
    updated: tuple[str, ...]
    removed: tuple[str, ...]
    reason: str  # "suggestions", "control", "personality", "loaded"


@dataclass(frozen=True, slots=True)
class ExploitSuspected:
    """The AI looks like it is farming rewards instead of progressing."""

    frame: int
    warning: ExploitWarning


class RewardService:
    def __init__(
        self,
        connection: ConnectionService,
        discovery: DiscoveryService,
        bus: EventBus,
        *,
        store: RewardStore | None = None,
        tally_capacity: int = 1000,
        exploit_window: int = 3600,
        exploit_min_reward: float = 20.0,
    ) -> None:
        self._connection = connection
        self._discovery = discovery
        self._bus = bus
        self._store = store
        self._profile: RewardProfile | None = None
        self._tally = RewardTally(tally_capacity)
        self._enabled = False
        self._saves = 0
        self._exploit_window = exploit_window
        self._exploit_min_reward = exploit_min_reward
        self._detector: ExploitDetector | None = None
        self._warnings: list[ExploitSuspected] = []
        self._current: ExploitSuspected | None = None
        bus.subscribe(EmulatorConnected, self._on_connected)
        bus.subscribe(EmulatorDisconnected, self._on_disconnected)
        bus.subscribe(DiscoveryUpdated, self._on_discovery)

    # ------------------------------------------------------------ queries

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def profile(self) -> RewardProfile | None:
        """The connected game's profile (kept readable after disconnect)."""
        return self._profile

    @property
    def tally(self) -> RewardTally:
        return self._tally

    @property
    def store(self) -> RewardStore | None:
        return self._store

    @property
    def saves(self) -> int:
        return self._saves

    @property
    def warnings(self) -> tuple[ExploitSuspected, ...]:
        """Every exploitation warning raised since connect."""
        return tuple(self._warnings)

    @property
    def current_warning(self) -> ExploitSuspected | None:
        """The warning still in force, if play has not moved on since."""
        return self._current

    def summary(self) -> str:
        if self._profile is None or not self._enabled:
            return "No game to reward."
        n = self._profile.enabled_count()
        return f"{n} reward{'s' if n != 1 else ''} switched on; {self._tally.total:+.1f} earned."

    # ------------------------------------------------------------ controls

    def increase(self, behaviour_id: str) -> Behaviour:
        return self._control(self._require().increase(behaviour_id))

    def decrease(self, behaviour_id: str) -> Behaviour:
        return self._control(self._require().decrease(behaviour_id))

    def set_value(self, behaviour_id: str, value: float) -> Behaviour:
        return self._control(self._require().set_value(behaviour_id, value))

    def set_enabled(self, behaviour_id: str, enabled: bool) -> Behaviour:
        return self._control(self._require().set_enabled(behaviour_id, enabled))

    def reset_to_suggested(self, behaviour_id: str | None = None) -> None:
        profile = self._require()
        profile.reset_to_suggested(behaviour_id)
        ids = (behaviour_id,) if behaviour_id is not None else profile.ids()
        self._announce(RewardProfileChanged((), ids, (), "control"))

    def choose_personality(self, which: str | Personality) -> Personality:
        profile = self._require()
        chosen = apply_personality(profile, which)
        self._announce(RewardProfileChanged((), profile.ids(), (), "personality"))
        return chosen

    def refresh_suggestions(self) -> SuggestionReport:
        """Bring the profile in line with the knowledge now."""
        profile = self._require()
        knowledge = self._discovery.knowledge
        if knowledge is None or knowledge.game_id != profile.game_id:
            return SuggestionReport((), (), ())
        report = apply_suggestions(profile, knowledge)
        if not report.is_empty:
            self._announce(
                RewardProfileChanged(report.added, report.updated, report.removed, "suggestions")
            )
        return report

    def save(self) -> Path | None:
        if self._store is None or self._profile is None:
            return None
        path = self._store.save(self._profile)
        self._saves += 1
        return path

    # ------------------------------------------------------------- events

    def _on_connected(self, event: EmulatorConnected) -> None:
        info = event.game_info
        game_id = game_id_for(info.system, info.title, info.rom_hash)
        self._tally.reset()
        self._enabled = event.capabilities.memory_read
        if not self._enabled:
            self._profile = None
            return
        if self._store is not None:
            try:
                self._profile = self._store.load_or_new(game_id)
            except ConfigurationError:
                self._profile = RewardProfile(game_id)  # a damaged file: start over
        else:
            self._profile = RewardProfile(game_id)
        self._detector = ExploitDetector(
            self._profile,
            self._tally,
            window=self._exploit_window,
            min_reward=self._exploit_min_reward,
        )
        self._warnings.clear()
        self._current = None
        self._bus.publish(RewardProfileChanged((), self._profile.ids(), (), "loaded"))
        self.refresh_suggestions()

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        if self._enabled:
            self.save()
        self._enabled = False

    def _on_discovery(self, update: DiscoveryUpdated) -> None:
        if not self._enabled or self._profile is None:
            return
        if update.added or update.changed or update.relationships:
            self.refresh_suggestions()
        for game_event in update.events:
            self._score(game_event)
        if update.events:
            self._watch_for_exploits(max(e.frame for e in update.events))

    # ------------------------------------------------------------ helpers

    def _score(self, game_event: GameEvent) -> None:
        assert self._profile is not None
        matched = self._profile.matching(game_event)
        if not matched:
            return
        amount = float(sum(b.value for b in matched))
        ids = tuple(b.id for b in matched)
        self._tally.record(game_event.frame, amount, ids, shares=[b.value for b in matched])
        if amount != 0.0:
            self._bus.publish(RewardEarned(game_event.frame, amount, ids, game_event))

    def _watch_for_exploits(self, now: int) -> None:
        if self._detector is None:
            return
        warning = self._detector.check(now)
        if warning is None:
            if self._current is not None and now - self._current.frame >= self._exploit_window:
                self._current = None  # play moved on: the warning no longer applies
            return
        current = self._current
        if (
            current is not None
            and current.warning.key == warning.key
            and now - current.frame < self._exploit_window
        ):
            return  # already said so for this window
        suspected = ExploitSuspected(now, warning)
        self._current = suspected
        self._warnings.append(suspected)
        self._bus.publish(suspected)

    def _require(self) -> RewardProfile:
        if self._profile is None or not self._enabled:
            raise UnsupportedCapabilityError(
                "memory_read", "Connect to a game that shares its memory first."
            )
        return self._profile

    def _control(self, behaviour: Behaviour) -> Behaviour:
        self._announce(RewardProfileChanged((), (behaviour.id,), (), "control"))
        return behaviour

    def _announce(self, change: RewardProfileChanged) -> None:
        self.save()
        self._bus.publish(change)


__all__ = ["ExploitSuspected", "RewardEarned", "RewardProfileChanged", "RewardService"]
