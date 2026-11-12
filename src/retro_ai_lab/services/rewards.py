"""Reward service: the live reward profile for the connected game.

While a game is connected it keeps one :class:`RewardProfile` (loaded from
the :class:`RewardStore` or started fresh), fills it from the knowledge as
discoveries arrive (DESIGN.md §5.4), turns every detected event into a
reward (:class:`RewardEarned`) that the learner and the UI can use, keeps
a :class:`RewardTally`, and offers the child's controls, which save the
profile and announce :class:`RewardProfileChanged`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from retro_ai_lab.core import ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.rewards.buttons import ensure_button_rows
from retro_ai_lab.rewards.exploitation import ExploitDetector, ExploitWarning
from retro_ai_lab.rewards.ground import GROUND_ID, ensure_ground_row
from retro_ai_lab.rewards.model import Behaviour, RewardProfile
from retro_ai_lab.rewards.own import is_own, own_behaviour, own_id
from retro_ai_lab.rewards.personalities import Personality, apply_personality
from retro_ai_lab.rewards.suggestions import (
    RelationshipWatch,
    SuggestionReport,
    apply_suggestions,
    rule_key,
)
from retro_ai_lab.rewards.tally import RewardTally
from retro_ai_lab.services.connection import (
    ConnectionService,
    EmulatorConnected,
    EmulatorDisconnected,
)
from retro_ai_lab.services.discovery import DiscoveryService, DiscoveryUpdated, game_id_for
from retro_ai_lab.services.observer import ButtonsHeld, GroundCovered
from retro_ai_lab.storage.rewards import RewardStore
from retro_ai_lab.understanding.events import EventKind, GameEvent
from retro_ai_lab.understanding.knowledge import VariableKind, plain_names


@dataclass(frozen=True, slots=True)
class RewardEarned:
    frame: int
    amount: float
    behaviour_ids: tuple[str, ...]
    event: GameEvent
    names: tuple[str, ...] = ()  # the behaviours' names, same order as the ids (T-857)
    shares: tuple[float, ...] = ()  # what each behaviour gave, same order as the ids


@dataclass(frozen=True, slots=True)
class GoStarted:
    """A go began (T-878): progress and score start earning from scratch."""

    frame: int


# Events that pay only past the go's best so far (T-878): a step back and forward
# again is not progress, and a value that resets on death is not progress either.
PAST_BEST_KINDS = frozenset({EventKind.PROGRESS, EventKind.SCORE_GAIN})


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
        self._watch = RelationshipWatch()
        self._current: ExploitSuspected | None = None
        self._held: frozenset[str] = frozenset()  # the step before's buttons (T-874)
        self._best: dict[str, int] = {}  # per go: the best each progress/score reached
        bus.subscribe(EmulatorConnected, self._on_connected)
        bus.subscribe(EmulatorDisconnected, self._on_disconnected)
        bus.subscribe(ButtonsHeld, self._on_buttons_held)
        bus.subscribe(GroundCovered, self._on_ground_covered)
        bus.subscribe(GoStarted, self._on_go_started)
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
        n = self._profile.active_count()
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

    def own(self) -> tuple[Behaviour, ...]:
        """The rewards a person added themselves (T-860)."""
        profile = self._profile
        if profile is None:
            return ()
        return tuple(b for b in profile if is_own(b.id))

    def penalised(self) -> frozenset[str]:
        """The values a person's own rewards penalise most: what a death is to them (T-964).

        An own reward with a value below zero on a value going down (or up) names
        a byte the person watches for harm. Of those, the life counts and health
        values (by the AI's kind guess) are the deaths; when none is life-like, the
        value with the heaviest penalty is (T-967: the user's "x position goes down"
        at -2 beside "Lifes goes down" at -20 made every step back a death). The
        survival count and the check of a start spot that loses a life by itself
        follow the answer before any guess.
        """
        penalties: dict[str, float] = {}
        for b in self.own():
            variable_id = b.trigger.variable_id
            if b.enabled and b.value < 0 and variable_id:
                penalties[variable_id] = min(penalties.get(variable_id, 0.0), b.value)
        if not penalties:
            return frozenset()
        knowledge = self._discovery.knowledge
        life_like = frozenset(
            i
            for i in penalties
            if knowledge is not None
            and i in knowledge
            and knowledge.get(i).kind in (VariableKind.LIVES, VariableKind.HEALTH)
        )
        if life_like:
            return life_like
        heaviest = min(penalties.values())
        return frozenset(i for i, value in penalties.items() if value == heaviest)

    def add_own(
        self, variable_id: str, *, goes_up: bool, value: float, name: str | None = None
    ) -> Behaviour:
        """Reward ``variable_id`` going up (or down) by ``value`` every step it does (T-860)."""
        profile = self._require()
        knowledge = self._discovery.knowledge
        if knowledge is None or variable_id not in knowledge:
            raise ConfigurationError("The AI does not know that value.")
        var = knowledge.get(variable_id)
        if profile.find(own_id(variable_id, goes_up)) is not None:
            raise ConfigurationError("That reward is already there.")
        if name is None:
            offered = knowledge.offered()
            name = plain_names(offered).get(variable_id, var.display_name)
        behaviour = own_behaviour(var, name, goes_up, value)
        profile.add(behaviour)
        self._ask_for_own()
        self._announce(RewardProfileChanged((behaviour.id,), (), (), "control"))
        return behaviour

    def remove_own(self, behaviour_id: str) -> None:
        """Take out a reward a person added (only those)."""
        profile = self._require()
        if not is_own(behaviour_id):
            raise ConfigurationError("Only a reward you added can be removed.")
        profile.remove(behaviour_id)
        self._ask_for_own()
        self._announce(RewardProfileChanged((), (), (behaviour_id,), "control"))

    def dismiss(self, behaviour_id: str) -> str:
        """Take a row out for good (T-943): the rule behind it is remembered, so the AI's
        next guess of the same kind brings no row of that name back. Returns the key."""
        profile = self._require()
        behaviour = profile.get(behaviour_id)
        if is_own(behaviour_id):
            raise ConfigurationError("A reward you added is removed, not dismissed.")
        key = behaviour_id
        var_id = behaviour.trigger.variable_id
        knowledge = self._discovery.knowledge
        if var_id is not None and knowledge is not None and var_id in knowledge:
            key = rule_key(behaviour.trigger.kind, knowledge.get(var_id).kind)
        profile.dismiss(key)
        profile.remove(behaviour_id)
        self._ask_for_own()
        self._announce(RewardProfileChanged((), (), (behaviour_id,), "control"))
        return key

    def suggest_again(self) -> SuggestionReport:
        """Forget every dismissal (T-943) and bring the suggested rows back."""
        profile = self._require()
        profile.bring_back()
        session = self._connection.session
        buttons: tuple[str, ...] = ()
        if session is not None:
            buttons, _ = ensure_button_rows(profile, session.buttons)
            if buttons:
                self._announce(RewardProfileChanged(buttons, (), (), "control"))
        report = self.refresh_suggestions()
        return replace(report, added=buttons + report.added) if buttons else report

    def _ask_for_own(self) -> None:
        """Tell the detector which values the person's own rewards watch."""
        profile = self._profile
        ids: set[str] = set()
        if profile is not None:
            for b in profile:
                if b.trigger.kind in (EventKind.UP, EventKind.DOWN) and b.trigger.variable_id:
                    ids.add(b.trigger.variable_id)
        self._discovery.ask_for(ids)

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

    def replace(self, profile: RewardProfile) -> None:
        """Swap in another profile for the same game (loading a saved AI)."""
        current = self._require()
        if profile.game_id != current.game_id:
            raise ConfigurationError(
                f"Those rewards are for {profile.game_id!r}, not the connected game."
            )
        self._profile = profile
        self._detector = ExploitDetector(
            profile,
            self._tally,
            window=self._exploit_window,
            min_reward=self._exploit_min_reward,
        )
        self._warnings.clear()
        self._current = None
        self._ask_for_own()
        self._announce(RewardProfileChanged((), profile.ids(), (), "loaded"))

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
        self._best.clear()
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
        self._watch.reset()
        self._held = frozenset()
        ensure_button_rows(self._profile, event.buttons)  # a row per button (T-874)
        self._ask_for_own()
        self._bus.publish(RewardProfileChanged((), self._profile.ids(), (), "loaded"))
        self.refresh_suggestions()

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        if self._enabled:
            self.save()
        self._enabled = False

    def _on_discovery(self, update: DiscoveryUpdated) -> None:
        if not self._enabled or self._profile is None:
            return
        if update.added or update.changed or self._relationships_moved(update):
            self.refresh_suggestions()
        for game_event in update.events:
            self._score(game_event)
        if update.events:
            self._watch_for_exploits(max(e.frame for e in update.events))

    # ------------------------------------------------------------ helpers

    def _relationships_moved(self, update: DiscoveryUpdated) -> bool:
        """Only a change among the confident relationships is worth a rebuild (T-842)."""
        knowledge = self._discovery.knowledge
        if not update.relationships or knowledge is None:
            return False
        return self._watch.changed(knowledge)

    def _on_go_started(self, event: GoStarted) -> None:
        self._best.clear()
        # A go starts from its checkpoint with nothing held the step before: its first
        # step is not a repeat of the last go's last buttons (T-983 turned this up).
        self._held = frozenset()

    def _beats_best(self, game_event: GameEvent) -> bool:
        """Remember the value and say whether it is the go's best so far (T-878)."""
        key, value = game_event.variable_id, game_event.after
        if key is None or value is None:
            return True
        best = self._best.get(key)
        if best is None or value > best:
            self._best[key] = value
            return True
        return False

    def _score(self, game_event: GameEvent) -> None:
        assert self._profile is not None
        if game_event.kind in PAST_BEST_KINDS and not self._beats_best(game_event):
            return
        matched = self._profile.matching(game_event)
        if not matched:
            return
        amount = float(sum(b.value for b in matched))
        ids = tuple(b.id for b in matched)
        shares = tuple(float(b.value) for b in matched)
        self._tally.record(game_event.frame, amount, ids, shares=shares)
        if amount != 0.0:
            names = tuple(b.name for b in matched)
            self._bus.publish(
                RewardEarned(game_event.frame, amount, ids, game_event, names, shares)
            )

    def _on_buttons_held(self, event: ButtonsHeld) -> None:
        """Score the buttons a step held (T-874): a press each, a repeat for those held before."""
        profile = self._profile
        if not self._enabled or profile is None:
            return
        before, self._held = self._held, event.buttons
        for name in sorted(event.buttons):
            press = GameEvent(EventKind.PRESS, event.frame, name, f"{name} held")
            if profile.reward_for(press) != 0.0:
                self._score(press)
            if name in before:
                again = GameEvent(EventKind.REPEAT, event.frame, name, f"{name} held again")
                if profile.reward_for(again) != 0.0:
                    self._score(again)

    def _on_ground_covered(self, event: GroundCovered) -> None:
        """Pay each step of new ground the AI's look at the screen found (T-993). The
        row is added the first time there is ground to pay for, unless a person removed
        it; from then on it is theirs to change like any other."""
        profile = self._profile
        if not self._enabled or profile is None or event.steps < 1:
            return
        if ensure_ground_row(profile):
            self._announce(RewardProfileChanged((GROUND_ID,), (), (), "suggestions"))
        step = GameEvent(EventKind.NEW_GROUND, event.frame, None, "a step of new ground")
        if profile.reward_for(step) == 0.0:
            return
        for _ in range(event.steps):
            self._score(step)

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
