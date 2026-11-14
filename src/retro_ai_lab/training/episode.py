"""One episode of play-and-learn (DESIGN.md Phase 5).

Each step the agent looks at the game (a :class:`State` from the believed
variables), picks an action, presses it as the AI, lets a few frames pass,
and learns from the points the reward and goal services announced in the
meantime. An episode ends when the game starts over (a reset event), after
``max_steps``, or if the emulator link drops. With save states, every
episode can start from the same checkpoint (DESIGN.md §9).
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

from retro_ai_lab.core import (
    Button,
    ConfigurationError,
    EmulatorError,
    Frame,
    UnsupportedCapabilityError,
)
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.core.worklog import app_work
from retro_ai_lab.learning.learner import Learner
from retro_ai_lab.learning.qlearning import Settings
from retro_ai_lab.learning.replay import Transition
from retro_ai_lab.learning.state import State, StateEncoder
from retro_ai_lab.services.connection import ConnectionService
from retro_ai_lab.services.discovery import DiscoveryUpdated
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.services.goals import ObjectiveCompleted, ObjectiveSetback
from retro_ai_lab.services.observer import ButtonsHeld, GroundCovered, ObservationService
from retro_ai_lab.services.rewards import GoStarted, RewardEarned, RewardProfileChanged
from retro_ai_lab.training.deaths import DeathWatch
from retro_ai_lab.training.moments import LEAD, Moment, Moments
from retro_ai_lab.training.route import Route
from retro_ai_lab.training.search import BUDGET, find_way
from retro_ai_lab.training.trial import Outcome
from retro_ai_lab.training.vision import DEFAULT_LOOK_EVERY, Mark, ScreenWatch
from retro_ai_lab.understanding.events import EventKind
from retro_ai_lab.understanding.knowledge import Knowledge, Variable, VariableKind

log = logging.getLogger(__name__)

DEFAULT_FRAMES_PER_STEP = 4
DEFAULT_MAX_STEPS = 600
DEFAULT_UNSTICK_AFTER = 25  # steps with nothing changing before the go counts as stuck
# No button is assumed (T-844): the training service hands the runner the wake
# button it found by experiment, and a stuck go with none known ends "stuck".
DEFAULT_UNSTICK_BUTTONS: tuple[str, ...] = ()
FINGERPRINT_STEPS = 64  # state keys a go remembers, for spotting two identical goes
# Each step is learned from the rewards of this many steps added up (discounted) and
# the situation after them, so a penalty reaches its cause this many steps a pass
# instead of one (n-step returns, T-963). Half a second at four frames a step.
LOOKAHEAD_STEPS = 8
DEFAULT_SETTLE_FRAMES = 5  # the understanding engine's event window
NOW_SLOT = "back_now"  # where the present is saved while a search plays its tries (T-1007)
STAND_AFTER = 60  # steps about one spot, no life lost, before a way on is sought (T-1018)
STAND_WITHIN = 64  # pixels either way that are still "about one spot": dithering, a jump
# A search at a standstill may make this many times the plays of one after a life
# lost. There most changes lose the life again and cost one blind play; here nearly
# all stay alive, and each that ends on a new screen costs a second, under the watch.
STAND_TRIES = 2
STAND_NEAR = 8  # pixels: it "came to the spot" at the first look this near where it now is
STAND_REACH = 4  # steps past that look which the way tried from before it still covers
PRESS_FOR = 16  # steps a go that was paid nothing holds its best-paid button for (T-1011)


@dataclass(frozen=True, slots=True)
class Earning:
    """What one behaviour (or goal step) earned over a go (T-857)."""

    id: str
    name: str
    count: int
    amount: float

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "count": self.count, "amount": self.amount}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Earning:
        try:
            return cls(
                str(data["id"]), str(data["name"]), int(data["count"]), float(data["amount"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad earning: {exc}") from exc


def describe_earned(earned: Iterable[Earning], most: int | None = 3) -> str:
    """Word the top earners: "Score +2,500.0 (120 times), Lap +40.0 (once), and 2 more".

    ``most=None`` words every earner (the log wants the whole breakdown, T-877).
    """
    if most is not None and most < 1:
        raise ConfigurationError("most must be positive.")
    items = list(earned)
    words = [
        f"{e.name} {e.amount:+,.1f} ({'once' if e.count == 1 else f'{e.count} times'})"
        for e in (items if most is None else items[:most])
    ]
    rest = len(items) - len(words)
    if rest > 0:
        words.append(f"and {rest} more")
    return ", ".join(words)


class EpisodeSignals:
    """Adds up the points announced on the bus and notices game restarts."""

    def __init__(self, bus: EventBus) -> None:
        self._reward = 0.0
        self._resets = 0
        self._completed = 0
        self._earned: dict[str, Earning] = {}
        self._listening = False
        self._last_reset = ""  # what the detector blamed, for the log (T-840)
        self._last_reset_ids: tuple[str, ...] = ()
        self._setbacks = 0  # life or health drops since the last take (T-934)
        self._watched: frozenset[str] | None = None  # whose setbacks count (T-940)
        self._score_ids: frozenset[str] = frozenset()  # whose rises are "score up" (T-976)
        self._scores = 0
        # What the reward of one step is made of, row by row (T-1027): gathered as the
        # payments come in, and handed over with the reward at each take().
        self._pending: dict[str, float] = {}
        self._parts: tuple[tuple[str, float], ...] = ()
        self._names: dict[str, str] = {}  # each row's name as it was last paid under
        self._gone: set[str] = set()  # rows removed from the rewards since last asked
        bus.subscribe(RewardEarned, self._on_reward)
        bus.subscribe(RewardProfileChanged, self._on_profile)
        bus.subscribe(ObjectiveCompleted, self._on_completed)
        bus.subscribe(ObjectiveSetback, self._on_setback)
        bus.subscribe(DiscoveryUpdated, self._on_discovery)

    @property
    def listening(self) -> bool:
        return self._listening

    @property
    def completed(self) -> int:
        """Objectives completed since :meth:`start` (not cleared by :meth:`take`)."""
        return self._completed

    @property
    def last_reset(self) -> str:
        """The detail and frame of the last reset counted since :meth:`start`."""
        return self._last_reset

    @property
    def last_reset_ids(self) -> tuple[str, ...]:
        """The values the last reset blamed (T-845)."""
        return self._last_reset_ids

    @property
    def earned(self) -> tuple[Earning, ...]:
        """What paid since :meth:`start`, biggest amount first (T-857)."""
        return tuple(sorted(self._earned.values(), key=lambda e: (-abs(e.amount), e.id)))

    @property
    def parts(self) -> tuple[tuple[str, float], ...]:
        """What the reward last taken was made of (T-1027): each reward row's id (a
        goal step's is ``step:`` and its id, a setback's penalty with it) and what it
        gave, in the ids' order."""
        return self._parts

    def name_of(self, key: str) -> str:
        """The name a row was last paid under; its id while it has never paid."""
        return self._names.get(key, key)

    def take_gone(self) -> frozenset[str]:
        """The rows removed from the rewards since the last call, whether a go was
        running or not (T-1027); clears them."""
        out = frozenset(self._gone)
        self._gone.clear()
        return out

    def _on_profile(self, event: RewardProfileChanged) -> None:
        self._gone.update(event.removed)

    def _credit(self, key: str, name: str, amount: float) -> None:
        self._pending[key] = self._pending.get(key, 0.0) + amount
        if name != key or key not in self._names:  # a payment sent with no name keeps the old
            self._names[key] = name
        had = self._earned.get(key)
        if had is None:
            self._earned[key] = Earning(key, name, 1, amount)
        else:
            self._earned[key] = Earning(key, name, had.count + 1, had.amount + amount)

    def start(self) -> None:
        self._reward = 0.0
        self._resets = 0
        self._completed = 0
        self._earned = {}
        self._last_reset = ""
        self._last_reset_ids = ()
        self._setbacks = 0
        self._scores = 0
        self._pending = {}
        self._parts = ()
        self._listening = True

    def stop(self) -> None:
        self._listening = False

    def take(self) -> tuple[float, int]:
        """(reward, resets) since the last take; clears both. What the reward was made
        of is in :attr:`parts` until the next take."""
        out = (self._reward, self._resets)
        self._reward = 0.0
        self._resets = 0
        self._parts = tuple(sorted(self._pending.items()))
        self._pending = {}
        return out

    def watch_scores(self, ids: frozenset[str]) -> None:
        """The values whose rises count as the score going up (T-976)."""
        self._score_ids = ids

    def take_scores(self) -> int:
        """Score rises since the last call; clears the count (T-976)."""
        out = self._scores
        self._scores = 0
        return out

    def take_setbacks(self) -> int:
        """Life or health drops since the last call; clears the count (T-934)."""
        out = self._setbacks
        self._setbacks = 0
        return out

    @property
    def watched(self) -> frozenset[str] | None:
        """Whose setbacks count; None for every value's (T-940)."""
        return self._watched

    def watch_setbacks(self, ids: frozenset[str] | None) -> None:
        """Count only the setbacks of these values; None counts every one (T-940).

        The user's 950-go log counted 45 lives lost a go because every life-count
        guess's drops were added up; the one life count and the one health the
        rewards follow are what a death looks like.
        """
        self._watched = ids

    def _on_reward(self, event: RewardEarned) -> None:
        if not self._listening:
            return
        self._reward += event.amount
        ids = event.behaviour_ids
        shares = event.shares if len(event.shares) == len(ids) else ()
        names = event.names if len(event.names) == len(ids) else ()
        for i, bid in enumerate(ids):
            share = shares[i] if shares else event.amount / len(ids)
            self._credit(bid, names[i] if names else bid, share)

    def _on_completed(self, event: ObjectiveCompleted) -> None:
        if self._listening:
            self._reward += event.reward
            self._completed += 1
            self._credit(f"step:{event.objective.id}", event.objective.description, event.reward)

    def _on_setback(self, event: ObjectiveSetback) -> None:
        if self._listening:
            self._reward += event.penalty
            self._credit(f"step:{event.objective.id}", event.objective.description, event.penalty)

    def _on_discovery(self, update: DiscoveryUpdated) -> None:
        if not self._listening:
            return
        dropped: set[str] = set()  # watched values that fell this update (T-967)
        if any(
            e.kind is EventKind.SCORE_GAIN
            or (e.kind is EventKind.UP and e.variable_id in self._score_ids)
            for e in update.events
        ):
            self._scores += 1  # T-976
        for event in update.events:
            if event.kind is EventKind.RESET:
                self._resets += 1
                self._last_reset = f"{event.detail} (frame {event.frame})"
                self._last_reset_ids = event.variables
            elif self._watched is None:
                if event.kind is EventKind.SETBACK:
                    self._setbacks += 1
            elif (
                event.kind in (EventKind.SETBACK, EventKind.DOWN)
                and event.variable_id is not None
                and event.variable_id in self._watched
            ):
                # A person's value drops as a DOWN event (its reward's trigger) and, when
                # the AI also takes it for a life count, as a SETBACK too: one death (T-967).
                dropped.add(event.variable_id)
        self._setbacks += len(dropped)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    steps: int
    frames: int
    reward: float
    ended_by: str  # "reset", "max_steps", "stuck", "disconnected", "stopped", "mode change"
    states_seen: int
    objectives: int = 0  # objectives completed during the go
    earned: tuple[Earning, ...] = ()  # what paid, biggest first (T-857)
    setbacks: int = 0  # lives or health lost during the go (T-934)
    first_setback_step: int | None = None  # the step the first was lost at, 1-based
    ways_sought: int = 0  # times it went back to look for a way past a life lost (T-1007)
    ways_found: int = 0  # times that found one
    pressed_on: int = 0  # times it was paid nothing for a while and pressed on (T-1011)
    remembered: int = 0  # steps it began with from the way it remembered (T-1014)
    ways_stood: int = 0  # of ways_sought, those where it stood still, no life lost (T-1018)

    @property
    def succeeded(self) -> bool:
        return self.objectives > 0

    def describe_pressing(self) -> str:
        """ ", pressed on 2 times", or "" when it never did (T-1011)."""
        if self.pressed_on == 0:
            return ""
        return f", pressed on {'once' if self.pressed_on == 1 else f'{self.pressed_on} times'}"

    def describe_remembered(self) -> str:
        """ ", began with the 187 steps it remembered", or "" when it began with none (T-1014)."""
        if self.remembered == 0:
            return ""
        steps = "step" if self.remembered == 1 else f"{self.remembered} steps"
        return f", began with the {steps} it remembered"

    def describe_ways(self) -> str:
        """ ", went back 2 times and found a way past once", or "" when it never looked
        (T-1007). Where some of those were at a standstill and not after a life lost
        (T-1018): ", went back 2 times (once where it stood still) and ..."."""
        if self.ways_sought == 0:
            return ""
        back = "once" if self.ways_sought == 1 else f"{self.ways_sought} times"
        if self.ways_stood and self.ways_sought == 1:
            back += " (where it stood still)"
        elif self.ways_stood == self.ways_sought:
            back += " (each time where it stood still)"
        elif self.ways_stood:
            stood = "once" if self.ways_stood == 1 else f"{self.ways_stood} times"
            back += f" ({stood} where it stood still)"
        if self.ways_found == 0:
            found = "found no way past"
        elif self.ways_found == self.ways_sought:
            found = "found a way past" if self.ways_found == 1 else "found a way past each time"
        else:
            found = (
                f"found a way past {'once' if self.ways_found == 1 else f'{self.ways_found} times'}"
            )
        return f", went back {back} and {found}"

    def describe_losses(self) -> str:
        """ ", lost a life 2 times, the first at step 3", or "" when none were lost (T-934)."""
        if self.setbacks == 0:
            return ""
        times = "once" if self.setbacks == 1 else f"{self.setbacks} times"
        first = (
            f", the first at step {self.first_setback_step}"
            if self.first_setback_step is not None
            else ""
        )
        return f", lost a life {times}{first}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "frames": self.frames,
            "reward": self.reward,
            "ended_by": self.ended_by,
            "states_seen": self.states_seen,
            "objectives": self.objectives,
            "earned": [e.to_dict() for e in self.earned],
            "setbacks": self.setbacks,
            "first_setback_step": self.first_setback_step,
            "ways_sought": self.ways_sought,
            "ways_found": self.ways_found,
            "pressed_on": self.pressed_on,
            "remembered": self.remembered,
            "ways_stood": self.ways_stood,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeResult:
        try:
            return cls(
                int(data["steps"]),
                int(data["frames"]),
                float(data["reward"]),
                str(data["ended_by"]),
                int(data["states_seen"]),
                int(data.get("objectives", 0)),
                tuple(Earning.from_dict(e) for e in data.get("earned", ())),
                int(data.get("setbacks", 0)),
                None if data.get("first_setback_step") is None else int(data["first_setback_step"]),
                int(data.get("ways_sought", 0)),
                int(data.get("ways_found", 0)),
                int(data.get("pressed_on", 0)),
                int(data.get("remembered", 0)),
                int(data.get("ways_stood", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad go result: {exc}") from exc


class EpisodeRunner:
    def __init__(
        self,
        connection: ConnectionService,
        observation: ObservationService,
        bus: EventBus,
        agent: Learner,
        encoder: StateEncoder,
        *,
        frames_per_step: int = DEFAULT_FRAMES_PER_STEP,
        max_steps: int = DEFAULT_MAX_STEPS,
        harness: ExperimentHarness | None = None,
        unstick_after: int | None = DEFAULT_UNSTICK_AFTER,
        unstick_buttons: tuple[str, ...] = DEFAULT_UNSTICK_BUTTONS,
        settle_frames: int = DEFAULT_SETTLE_FRAMES,
        on_learned: Callable[[Transition], None] | None = None,
        lookahead: int = LOOKAHEAD_STEPS,
        watched_setbacks: Callable[[], frozenset[str]] | None = None,
        watch: ScreenWatch | None = None,
        look_every: int = DEFAULT_LOOK_EVERY,
        deaths: DeathWatch | None = None,
        moments: Moments | None = None,
        ways_per_go: int = 0,
        way_tries: int = BUDGET,
        way_seconds: float | None = None,
        give_up_after: int | None = None,
        flat_out: Callable[[], AbstractContextManager[None]] | None = None,
        press_after: int = 0,
        press_for: int = PRESS_FOR,
        route: Route | None = None,
        stand_after: int = 0,
    ) -> None:
        if route is not None and not isinstance(route, Route):
            raise ConfigurationError("route must be a Route (or None).")
        for value, what, least in (
            (ways_per_go, "ways_per_go", 0),
            (way_tries, "way_tries", 1),
            (press_after, "press_after", 0),
            (stand_after, "stand_after", 0),
            (press_for, "press_for", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ConfigurationError(f"{what} must be a whole number, {least} or more.")
        if way_seconds is not None and (
            isinstance(way_seconds, bool)
            or not isinstance(way_seconds, (int, float))
            or not way_seconds > 0
        ):
            raise ConfigurationError("way_seconds must be a number above 0 (or None).")
        if give_up_after is not None and (
            isinstance(give_up_after, bool)
            or not isinstance(give_up_after, int)
            or give_up_after < 1
        ):
            raise ConfigurationError("give_up_after must be a whole number, 1 or more (or None).")
        if moments is not None and NOW_SLOT in moments.slots:
            raise ConfigurationError(
                f"The slot {NOW_SLOT!r} is where the present is kept during a search: "
                "saved moments need other names."
            )
        if look_every < 1:
            raise ConfigurationError("look_every must be at least 1.")
        if frames_per_step < 1:
            raise ConfigurationError("frames_per_step must be at least 1.")
        if lookahead < 1:
            raise ConfigurationError("lookahead must be at least 1.")
        if max_steps < 1:
            raise ConfigurationError("max_steps must be at least 1.")
        if unstick_after is not None and unstick_after < 1:
            raise ConfigurationError("unstick_after must be at least 1 (or None).")
        if settle_frames < 0:
            raise ConfigurationError("settle_frames must be >= 0.")
        self._connection = connection
        self._observation = observation
        self._agent = agent
        self._encoder = encoder
        self._frames_per_step = frames_per_step
        self._max_steps = max_steps
        self._harness = harness
        self._unstick_after = unstick_after
        self._unstick_buttons = unstick_buttons
        self._settle_frames = settle_frames
        self._on_learned = on_learned
        self._lookahead = lookahead
        self._pending: deque[tuple[State, int, float]] = deque()  # steps not yet learned
        self._watched_setbacks = watched_setbacks  # the values a person penalises (T-964)
        self._deaths = deaths  # what the AI saw before it died (T-968)
        self._watched_last: frozenset[str] | None = None
        self._watch = watch  # looks at the screen every few steps (T-976)
        self._look_every = look_every
        self._seen: Frame | None = None  # the screenshot of the step being settled
        # Lives lost and score rises on steps without a look, kept for the next look (T-988).
        self._unseen_lost = 0
        self._unseen_scored = 0
        # Saved moments of the go and what it did since each, to go back to (T-1004).
        self._moments = moments
        self._forced: deque[int] = deque()  # moves to play next without asking the learner
        self._times_back = 0
        self._cannot_save = False
        # At a life lost, go back and look for a way past (T-1007).
        self._ways_per_go = ways_per_go
        self._way_tries = way_tries
        self._way_seconds = way_seconds
        # Searches in a row that found nothing, across goes; at give_up_after the
        # runner stops going back for good: in this game it only costs time (T-1009).
        self._give_up_after = give_up_after
        self._missed = 0
        self._gave_up = False
        self._flat_out = flat_out
        self._ways_left = 0
        self._ways_sought = 0
        self._ways_found = 0
        self._ways_stood = 0
        # Paid nothing for press_after steps, a go that can still go back holds the
        # button that has paid best for press_for steps (T-1011). What each button has
        # been paid is kept over the goes; 0 never presses on.
        self._press_after = press_after
        self._press_for = press_for
        # The best way played, to begin a go with (T-1014, T-1015).
        self._route = route
        self._route_left = 0  # moves of it still waiting to be played this go
        self._from_memory = 0  # steps this go has played from it (T-1015)
        self._way_left = 0  # moves of a way a search found that are still to play (T-1018)
        self._remembered = 0  # steps of it this go began with
        self._paid: dict[str, float] = {}
        self._unpaid = 0  # steps since anything was paid
        self._press_next = 0  # which of the paid buttons the next press holds, best first
        self._pressing = False  # the moves being played are a press
        self._pressed_stuck = False  # this standstill has had its press
        # A standstill (T-1018): the spot the watch had it at when it last left one, the
        # steps since, whether this go has counted any new ground, and whether this
        # standstill has had its search.
        # About one spot for stand_after steps with no life lost, a go that can still
        # go back looks for a way on (T-1018). 0 is never; the training service passes
        # STAND_AFTER.
        self._stand_after = stand_after
        self._stood_at: tuple[int, int] | None = None
        self._stood_for = 0
        self._stood_from = 0  # the step of the story at which it came to that spot
        # Where it was at each look since then, with the story's step (T-1018).
        self._stood_spots: deque[tuple[int, tuple[int, int]]] = deque(maxlen=256)
        self._got_somewhere = False
        self._stood_sought = False
        self._stood_missed = 0  # such searches in a row that found nothing, go to go
        self._stood_gave_up = False
        self._held_reset = False  # a restart with a life lost is waiting for its search
        self._held_said = False
        self._pressed_on = 0
        self._press_said = False
        self._restored = False
        self._unsticks = 0
        self._same_state = 0
        self._last_buttons: frozenset[str] = frozenset()
        self._last_action: int | None = None
        self._last_state: State | None = None
        self._last_reward = 0.0
        self._bus = bus
        self._signals = EpisodeSignals(bus)
        self._buttons: dict[str, Button] = {}
        self._state: State | None = None
        self._steps = 0
        self._frames = 0
        self._reward = 0.0
        self._states: set[str] = set()
        self._ended_by: str | None = None
        self._fingerprint: list[str] = []
        self._setbacks = 0  # lives lost this go and the step of the first (T-934)
        self._first_setback_step: int | None = None
        self._actions_seen: set[int] = set()

    # ------------------------------------------------------------ queries

    @property
    def fingerprint(self) -> tuple[str, ...]:
        """The first state keys of this go, for spotting a game that ignores the AI."""
        return tuple(self._fingerprint)

    @property
    def actions_varied(self) -> bool:
        """True once the go has pressed more than one distinct action."""
        return len(self._actions_seen) > 1

    @property
    def restored(self) -> bool:
        """True when this go began by putting the game back to the checkpoint (T-961)."""
        return self._restored

    @property
    def stuck_from_start(self) -> bool:
        """True when the go ended "stuck" without the screen ever changing.

        Then the place the go starts from is what ignores the AI (a game-over or
        title screen), not something that happened during the go.
        """
        return self._ended_by == "stuck" and self._same_state >= self._steps

    @property
    def unstick_buttons(self) -> tuple[str, ...]:
        return self._unstick_buttons

    def set_unstick_buttons(self, buttons: tuple[str, ...]) -> None:
        """The button(s) to press when a screen stops changing (found by experiment)."""
        self._unstick_buttons = tuple(buttons)

    @property
    def running(self) -> bool:
        return self._state is not None and self._ended_by is None

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def state(self) -> State | None:
        return self._state

    @property
    def signals(self) -> EpisodeSignals:
        return self._signals

    @property
    def frames_per_step(self) -> int:
        return self._frames_per_step

    @property
    def last_buttons(self) -> frozenset[str]:
        """What was pressed on the last step (for the UI to show)."""
        return self._last_buttons

    @property
    def last_action(self) -> int | None:
        """The action behind the last step; None for an unstick press."""
        return self._last_action

    @property
    def last_state(self) -> State | None:
        """The situation the last step's move was chosen in."""
        return self._last_state

    @property
    def last_reward(self) -> float:
        """The reward the last step collected."""
        return self._last_reward

    @property
    def last_reset(self) -> str:
        """What the detector blamed for the reset that ended the go, if one did (T-840)."""
        return self._signals.last_reset

    @property
    def last_reset_ids(self) -> tuple[str, ...]:
        """The values the reset that ended this go blamed (T-845)."""
        return self._signals.last_reset_ids

    @property
    def watch(self) -> ScreenWatch | None:
        return self._watch

    def set_watch(self, watch: ScreenWatch | None) -> None:
        """Look at the screen while playing (T-976); None stops looking."""
        self._watch = watch

    @property
    def moments(self) -> Moments | None:
        """The saved moments this go keeps, if it keeps any (T-1004)."""
        return self._moments

    @property
    def times_back(self) -> int:
        """How often this go was put back to one of its saved moments."""
        return self._times_back

    @property
    def ways_sought(self) -> int:
        """Times this go went back to look for a way past a life lost (T-1007)."""
        return self._ways_sought

    @property
    def ways_found(self) -> int:
        """Times that found one."""
        return self._ways_found

    @property
    def ways_stood(self) -> int:
        """Of :attr:`ways_sought`, those made where it stood still with no life lost
        (T-1018)."""
        return self._ways_stood

    @property
    def stand_after(self) -> int:
        """Steps about one spot, with no life lost, before it looks for a way on
        (T-1018); 0 when it never does."""
        return self._stand_after

    @property
    def gave_up(self) -> bool:
        """Its searches kept finding nothing, so it no longer goes back (T-1009)."""
        return self._gave_up

    @property
    def stood_gave_up(self) -> bool:
        """Its searches at a standstill kept finding nothing, so it makes no more of
        them (T-1018). After a life lost it still goes back."""
        return self._stood_gave_up

    @property
    def press_after(self) -> int:
        """Steps paid nothing before it presses on; 0 when it never does (T-1011)."""
        return self._press_after

    @property
    def pressed_on(self) -> int:
        """Times this go pressed on."""
        return self._pressed_on

    @property
    def route(self) -> Route | None:
        """The way this runner remembers from go to go, if it keeps one (T-1014)."""
        return self._route

    @property
    def remembered(self) -> int:
        """Steps of the remembered way this go began with."""
        return self._remembered

    @property
    def own_steps(self) -> int:
        """Steps of this go that were not played from the way it remembers: what the
        go's length is counted in (T-1015)."""
        return self._steps - self._from_memory

    @property
    def paid_buttons(self) -> tuple[str, ...]:
        """The buttons that have been paid, over this runner's goes, the best-paid first."""
        return tuple(sorted(self._paid, key=lambda name: (-self._paid[name], name)))

    @property
    def unsticks(self) -> int:
        """Times Start was pressed because nothing was changing (a game-over screen)."""
        return self._unsticks

    # ------------------------------------------------------------ driving

    def begin(self, *, restore: bool = True) -> State:
        """Start an episode: restore the checkpoint if there is one, read the first state.

        ``restore=False`` carries on from wherever the game is, even with a checkpoint
        (when the child is the one playing).
        """
        if self.running:
            raise ConfigurationError("An episode is already running.")
        self._steps = 0
        self._frames = 0
        self._reward = 0.0
        self._states.clear()
        self._ended_by = None
        self._same_state = 0
        self._fingerprint = []
        self._setbacks = 0
        self._first_setback_step = None
        self._actions_seen = set()
        self._pending.clear()
        self._forced.clear()
        self._times_back = 0
        self._held_reset = False
        self._ways_left = 0 if self._gave_up else self._ways_per_go
        self._unpaid = 0
        self._press_next = 0
        self._pressing = False
        self._pressed_stuck = False
        self._stood_sought = False
        self._stood_at = None
        self._stood_for = 0
        self._stood_spots.clear()
        self._got_somewhere = False
        self._pressed_on = 0
        self._ways_sought = 0
        self._ways_found = 0
        self._ways_stood = 0
        if self._moments is not None:
            self._moments.forget()  # a new go is a new story
        self._buttons = {b.name: b for b in self._connection.call(lambda a: a.buttons())}
        self._restored = restore and self._harness is not None and self._harness.has_checkpoint
        if self._restored:
            assert self._harness is not None
            self._harness.restore()
        self._encoder.refresh()
        self._watch_deaths(self._encoder.knowledge)
        score = self._encoder.knowledge.best(VariableKind.SCORE)
        self._signals.watch_scores(frozenset({score.id}) if score is not None else frozenset())
        self._seen = None
        self._unseen_lost = 0
        self._unseen_scored = 0
        if self._watch is not None:
            self._watch.begin_go()
        # Observe before listening: putting the game back to the checkpoint looks like
        # a restart to the understanding engine, and that must not end the new episode.
        latest = self._observation.observe()
        snapshot = latest.snapshot if latest is not None else self._observation.latest
        if snapshot is None:
            raise ConfigurationError(
                "Can't see the game's memory, so there is nothing to learn from."
            )
        self._signals.start()
        self._bus.publish(GoStarted(self._observation.frame))  # T-878
        self._state = self._encoder.encode(snapshot)
        self._states.add(self._state.key)
        self._begin_route()
        return self._state

    def _loss_stands(self) -> None:
        """A life went and the go did not get it back. What the go remembers of itself
        ends ``LEAD`` steps before it (T-1014), and so does the way it began with, if
        the life went that close behind the way's end (T-1015): a way that ends within
        a few steps of a loss the AI finds no way past would bring every go to the
        same end. The next go arrives earlier, with room to do something else."""
        route = self._route
        if route is None:
            return
        had = len(route.moves)
        if route.failed(LEAD):
            log.info(
                "A life went at step %d, close behind the end of the way the AI remembers, "
                "and it found no way past: it now remembers %d steps of that way instead "
                "of %d, to arrive there earlier.",
                self._steps,
                len(route.moves),
                had,
            )
        route.lost(LEAD)

    def _rows_gone(self) -> None:
        """Reward rows have gone since this was last asked (the AI dropped the guess
        one went by, a person removed one): what they paid is taken out of the way the
        AI remembers and of this go's own account (T-1027), so that the way is not
        held to be worth points nothing pays any more."""
        rows = self._signals.take_gone()
        route = self._route
        if not rows or route is None:
            return
        had, worth = len(route.moves), route.paid
        found = route.gone(rows)
        if not found:
            return
        names = ", ".join(sorted(self._signals.name_of(row) for row in found))
        left = len(route.moves)
        if not left:
            log.info(
                "Gone from the rewards: %s. Nothing else had paid for the way the AI "
                "remembers (%.1f points), so it forgets that way, and its next go begins "
                "without one.",
                names,
                worth,
            )
        elif left < had:
            log.info(
                "Gone from the rewards: %s. What that paid is taken out of the way the AI "
                "remembers: it is worth %.1f points now, not %.1f, and the AI remembers "
                "%d step%s of it instead of %d, up to where it is paid the most.",
                names,
                route.paid,
                worth,
                left,
                "" if left == 1 else "s",
                had,
            )
        else:
            log.info(
                "Gone from the rewards: %s. What that paid is taken out of the way the AI "
                "remembers: it is worth %.1f points now, not %.1f.",
                names,
                route.paid,
                worth,
            )

    def _begin_route(self) -> None:
        """Queue the way remembered from this go's start, if there is one (T-1014).

        Only a go put back to the checkpoint starts where the way starts, and a way is
        remembered where the AI can go back, which takes its look at the screen: any
        other go plays and remembers nothing. The moves are played through the ordinary
        step, like a way found by going back, without asking the learner.
        """
        self._route_left = self._remembered = self._from_memory = 0
        self._way_left = 0
        route = self._route
        if route is None:
            return
        self._rows_gone()  # a row that went between goes: before the way is played
        start = None
        if self._restored and self._watch is not None and self._harness is not None:
            start = (self._harness.slot, self._harness.taken)
        had = len(route.moves)
        moves = route.begin(start)
        if had and start is not None and not moves:
            log.info(
                "The spot the goes start from has changed, so the way the AI remembered "
                "from the old one (%d step%s) is forgotten.",
                had,
                "" if had == 1 else "s",
            )
        size = len(self._agent.space)
        if any(move >= size for move in moves):
            route.forget()  # remembered with other controls
            moves = route.begin(start)
        self._forced.extend(moves)
        self._route_left = self._remembered = len(moves)
        if moves:
            log.debug("The go begins with the %d steps the AI remembers.", len(moves))

    def step(self) -> bool:
        """One act-advance-learn step. Returns False once the episode is over."""
        if self._state is None:
            raise ConfigurationError("Call begin() before step().")
        if self._ended_by is not None:
            return False
        state = self._state
        from_route = False  # this step's move is one of the way it remembers
        # On a screen where nothing changes (game over, a menu) the learner's menu has
        # no way out. Press the wake button the service found, if it has one; with
        # none known the go ends "stuck" and the service goes looking for it (T-844).
        stuck = self._unstick_after is not None and self._same_state >= self._unstick_after
        if stuck and (self._forced or self._press_on(stuck=True)):
            # A list of moves is played to its end: the way it remembers may have stood
            # about before it got on (T-1014). Otherwise nothing changing is getting
            # nowhere too: press on first, once.
            stuck = False
        if stuck and self._stands_still(stuck=True):
            # Nothing changing is a standstill whatever the count of steps says, and
            # the go is about to end or to press a wake button, which ends the story
            # the saved moments belong to: look for the way on first (T-1018).
            if not self._seek(stood=True):
                return False
            stuck = not self._forced
        if stuck:
            names = frozenset(n for n in self._unstick_buttons if n in self._buttons)
            if not names:
                self._ended_by = "stuck"
                self._signals.stop()
                return False
            action = None
            self._unsticks += 1
            self._same_state = 0
            self._forced.clear()
            self._route_left = 0
            self._pressing = False
        else:
            if not self._keep_moment():
                return False
            if self._stands_still() and not self._seek(stood=True):
                return False
            self._press_on()
            if self._forced:
                action = self._forced.popleft()  # a way worked out for it (T-1004)
                if self._way_left:
                    self._way_left -= 1
                if self._route_left:
                    self._route_left -= 1
                    self._from_memory += 1
                    from_route = True
            else:
                with app_work.timed("step: choose"):
                    action = self._agent.choose(state)
            names = self._agent.space.buttons_for(action)
        held = [self._buttons[n] for n in names]
        self._last_buttons = frozenset(names)
        self._last_action = action
        try:
            # One request per step (T-824): the buttons, the frames and the
            # memory snapshot after them travel together.
            self._observation.record_input(names, "ai")
            self._bus.publish(ButtonsHeld(self._observation.frame, frozenset(names), "ai"))
            frames = self._frames_per_step
            reads = self._observation.snapshot_reads()
            pressed = frozenset(held)
            look = self._watch is not None and self._steps % self._look_every == 0
            with app_work.timed("step: emulator"):
                result = self._connection.call(
                    lambda a: a.step(frames, pressed=pressed, reads=reads, screenshot=look)
                )
            self._seen = result.frame if look else None
            # The snapshot's readers (change tracker, discovery, rewards) run inside.
            with app_work.timed("step: observe"):
                self._observation.advance(frames, snapshot=result.reads if reads else None)
        except EmulatorError:
            self._ended_by = "disconnected"
            self._signals.stop()
            return False
        if self._moments is not None:
            self._moments.did(action)  # an unstick press is a hole: it forgets them
        lost_before = self._setbacks
        settled = self._settle(state, action, self._frames_per_step, "ai")
        route = self._route
        if self._setbacks > lost_before and route is not None:
            if from_route and route.failed(LEAD):
                log.info(
                    "A life went at step %d while the AI was playing the way it remembers, "
                    "so it no longer trusts that way from %d steps before there.",
                    self._steps,
                    LEAD,
                )
            if not settled:
                self._loss_stands()  # the go ends with the loss: its story ends before it
        if not settled:
            return False
        if self._setbacks > lost_before:
            self._forced.clear()  # whatever way was being played did not get past
            self._route_left = 0
            self._pressing = False  # a press that cost a life is not one that got nothing
            waiting, self._held_reset = self._held_reset, False
            back = self._times_back
            if not self._seek():
                return False
            if self._times_back == back:
                self._loss_stands()  # the story ends before it (T-1014)
            if waiting and self._times_back == back:
                self._ended_by = "reset"  # no way past: the restart stands (T-1013)
                self._signals.stop()
                return False
            if waiting:
                log.log(
                    logging.DEBUG if self._held_said else logging.INFO,
                    "The game put the player back to the start when that life went, which "
                    "used to end the go. The AI has gone back to before it and carries on.",
                )
                self._held_said = True
        return True

    def _paid_moves(self) -> list[tuple[str, int]]:
        """The buttons that have been paid, best first, each with its own move."""
        space = self._agent.space
        moves = []
        for name in self.paid_buttons:
            try:
                moves.append((name, space.index_of([name])))
            except ConfigurationError:
                continue  # no move of its own: it is only ever held with another
        return moves

    def _press_on(self, *, stuck: bool = False) -> bool:
        """Paid nothing for a while, hold the best-paid button for a while (T-1011).

        A brain that has come to rate waiting above going on stands still or dithers,
        loses no life, and so never looks for the way past what it is avoiding. Only
        where the go can still go back: there a life lost to the press is how the way
        is found. A press that was paid nothing (and cost no life, or the search would
        have followed) is followed by the next-best-paid button, round and round; any
        pay starts again from the best.

        ``stuck`` is the go about to end because nothing changes: that is getting
        nowhere whatever the count of unpaid steps says (the last pay can come a step
        or two after the last change), so it presses on first, once for each such
        standstill; if that changes nothing either, the go is stuck as before. True
        when a press was started.
        """
        if self._forced:
            return False
        if self._pressing:  # the last press ran its course, paid nothing
            self._pressing = False
            self._press_next += 1
        if not self._press_after or self._moments is None or self._ways_left < 1:
            return False
        if stuck:
            if self._pressed_stuck:
                return False
        elif self._unpaid < self._press_after:
            return False
        moves = self._paid_moves()
        if not moves:
            return False
        name, move = moves[self._press_next % len(moves)]
        self._forced.extend([move] * self._press_for)
        self._pressing = True
        self._pressed_stuck = stuck
        self._pressed_on += 1
        self._unpaid = 0
        log.log(
            logging.DEBUG if self._press_said else logging.INFO,
            "The AI has been paid nothing for %d steps, so it presses on with %s for %d steps. "
            "It does this where it can go back: if that costs a life, it looks for the way past.",
            self._press_after,
            name,
            self._press_for,
        )
        self._press_said = True
        return True

    def _stands_still(self, *, stuck: bool = False) -> bool:
        """The go got somewhere and has then stayed about one spot for ``stand_after``
        steps, with no life lost (T-1018): a wall, a ledge too high. Then there is a
        way on to look for, as after a life lost. Once for each standstill, and not
        while a list of moves is being played. With ``stuck`` nothing at all has
        changed for as long as ends a go, and the count of steps is not waited for.

        By where the watch has it and not by the pay: something is nearly always paid
        a little (a jump on the spot reaches a new height, a clock taken for progress
        ticks). And not by new ground alone: put back to the start after a life lost
        it walks over old ground, which is no standstill.
        """
        if not self._stand_after or self._forced or self._stood_sought or self._stood_gave_up:
            return False
        if self._moments is None or not self._ways_per_go or self._ways_left < 1:
            return False
        if self._watch is None or not self._watch.follows_camera:
            return False
        return self._got_somewhere and (stuck or self._stood_for >= self._stand_after)

    def _note_place(self, new_ground: int) -> None:
        """After a look: whether it is still about the spot it was at (T-1018)."""
        assert self._watch is not None
        if new_ground:
            self._got_somewhere = True
        x, y = self._watch.spot
        at = self._stood_at
        now = self._moments.steps if self._moments is not None else 0
        if at is None or max(abs(x - at[0]), abs(y - at[1])) > STAND_WITHIN:
            self._stood_at = (x, y)
            self._stood_for = 0
            self._stood_from = now
            self._stood_sought = False
            self._stood_spots.clear()
        self._stood_spots.append((now, (x, y)))

    def _left_the_spot(self, seen: Outcome) -> bool:
        """A try from where it stood still ended out of "about one spot": more than
        ``STAND_WITHIN`` pixels from where the go came to stand, by the same measure
        the standing was told by. New ground alone will not do: a hop of two steps
        and standing again counts some, and where the camera has not begun to move
        a run across half the screen counts next to none."""
        at = self._stood_at
        if at is None or seen.spot is None:
            return False
        x, y = seen.spot
        return max(abs(x - at[0]), abs(y - at[1])) > STAND_WITHIN

    def _came_to_the_spot(self) -> int:
        """The step of the story at which it was first seen where it now stands: the
        first look of this standstill within ``STAND_NEAR`` pixels of the last."""
        if not self._stood_spots:
            return self._stood_from
        x, y = self._stood_spots[-1][1]
        return next(
            step
            for step, (sx, sy) in self._stood_spots
            if max(abs(sx - x), abs(sy - y)) <= STAND_NEAR
        )

    def _keep_moment(self) -> bool:
        """Save the game when a moment is due (T-1004). False if the link dropped.

        Only while a life or health value is watched: without one nothing can be seen
        to be lost, so there is never a reason to go back.
        """
        moments = self._moments
        if moments is None or self._cannot_save or not moments.due():
            return True
        if not self._signals.watched:
            return True
        if self._ways_per_go and not self._ways_left:
            return True  # no search is left to go back for (T-1007)
        slot = moments.next_slot()
        try:
            with app_work.timed("step: save"):
                self._connection.call(lambda a: a.save_state(slot))
        except UnsupportedCapabilityError:
            self._cannot_save = True
            log.info("This console can't save the game, so the AI cannot go back and retry.")
            return True
        except EmulatorError:
            self._ended_by = "disconnected"
            self._signals.stop()
            return False
        moments.saved(self._watch.mark() if self._watch is not None else None)
        return True

    def _way_back(
        self, *, stood: bool = False
    ) -> tuple[Moment, tuple[int, ...], list[Variable]] | None:
        """Where a search for a way past a life lost now would start: the moment, what
        was done since, and the life values to watch. None where none can be made.

        For a standstill (``stood``) it is the moment before the go came to about the
        spot, and what it did up to ``STAND_REACH`` steps past the look at which it was
        first seen where it now stands: the changes tried, from the end of that
        backwards, then begin where the standing did, and next to none of the standing
        is in the way found.
        """
        moments = self._moments
        if moments is None or not self._ways_left or self._watch is None:
            return None
        found = moments.since(self._stood_from) if stood else moments.before_loss()
        if found is None or not isinstance(found[0].mark, Mark):
            return None
        if stood:
            moment, moves = found
            came = self._came_to_the_spot()
            found = moment, moves[: max(0, came - moment.step) + STAND_REACH]
            if not found[1]:
                return None
        knowledge = self._encoder.knowledge
        lives = [knowledge.get(i) for i in sorted(self._signals.watched or ()) if i in knowledge]
        return (*found, lives) if lives else None

    def _seek(self, *, stood: bool = False) -> bool:
        """A life just went: go back and look for a way past (T-1007). False if the link dropped.

        The loss itself stands for the learner and the count: it happened. What goes
        back is the game. From the saved moment before, the best-paid button held
        and the small changes to that, and then the small changes to what it did,
        are tried off the record (:func:`find_way`); a way found is played
        from that moment through the ordinary step, and with none the game is put
        back to where it was and the go carries on a life down. A loss that stands
        makes the moments before it useless (every try from them would lose that
        life again first), so they are forgotten.
        """
        moments = self._moments
        if moments is None or not self._ways_per_go:
            return True  # nothing here searches: the moments are the caller's (T-1005)
        if stood:
            self._stood_sought = True
        start = self._way_back(stood=stood)
        if start is None:
            if not stood:
                moments.forget()
            return True
        moment, trace, lives = start
        watch = self._watch
        mark = moment.mark
        assert watch is not None and isinstance(mark, Mark)
        self._ways_left -= 1
        self._ways_sought += 1
        if stood:
            self._ways_stood += 1
        space = self._agent.space
        at = self._steps
        gone = moments.steps - moment.step
        stayed = max(self._stood_for, self._same_state)
        paid = self._paid_moves()  # the best-paid button's move is the plain way (T-1012)
        try:
            fast = self._flat_out() if self._flat_out is not None else nullcontext()
            with fast, app_work.timed("step: search"):
                self._connection.call(lambda a: a.save_state(NOW_SLOT))
                search = find_way(
                    self._connection,
                    moment.slot,
                    trace,
                    space,
                    frames_per_step=self._frames_per_step,
                    lives=lives,
                    watch=lambda: watch.trial(mark),
                    budget=self._way_tries * (STAND_TRIES if stood else 1),
                    look_every=self._look_every,
                    seconds=self._way_seconds,
                    plain=paid[0][1] if paid else None,
                    stood=stood,
                    enough=self._left_the_spot if stood else None,
                )
                if search.way is None:
                    self._connection.call(lambda a: a.load_state(NOW_SLOT))
            if search.way is not None:
                self.go_back(moment, then=search.way.moves)
                self._way_left = len(search.way.moves)
        except EmulatorError:
            self._ended_by = "disconnected"
            self._signals.stop()
            return False
        back = f"{gone} step{'' if gone == 1 else 's'}"
        way = search.way
        if way is None and stood:
            log.info(
                "Stayed about one spot for %d steps, with no life lost. Went back %s and "
                "made %d tries in %.1f s%s: none got on.",
                stayed,
                back,
                search.tries,
                search.seconds,
                ", all the time a search may take" if search.out_of_time else "",
            )
            self._stood_missed += 1
            # As many in a row for nothing as stop the searches after a life lost
            # (``give_up_after``), counted apart: one kind can be of no use in a game
            # where the other is.
            if self._give_up_after is not None and self._stood_missed >= self._give_up_after:
                self._stood_gave_up = True
                log.info(
                    "Looking for a way on where it stands still found none %d times in a "
                    "row, so the AI stops doing that in this game: here it only costs time.",
                    self._stood_missed,
                )
            return True
        if way is None:
            moments.forget()
            log.info(
                "Lost a life at step %d. Went back %s and made %d tries in %.1f s%s: "
                "none got past.",
                at,
                back,
                search.tries,
                search.seconds,
                ", all the time a search may take" if search.out_of_time else "",
            )
            self._missed += 1
            if self._give_up_after is not None and self._missed >= self._give_up_after:
                self._gave_up = True
                self._ways_left = 0
                log.info(
                    "Going back found no way past %d time%s in a row, so the AI stops going "
                    "back in this game: here it only costs time.",
                    self._missed,
                    "" if self._missed == 1 else "s",
                )
            return True
        if stood:
            self._stood_missed = 0
        else:
            self._missed = 0
        self._ways_found += 1
        change = way.change
        lead = (
            f"Stayed about one spot for {stayed} steps, with no life lost. Went back "
            f"{back} and found a way on at try {way.tries}, after {search.seconds:.1f} s: "
            if stood
            else f"Lost a life at step {at}. Went back {back} and found a way past on try "
            f"{way.tries}, after {search.seconds:.1f} s: "
        )
        held = f"{space[change.move].label} for {change.hold} step{'' if change.hold == 1 else 's'}"
        if way.plain is None and stood:
            how = f"{held} from step {change.at + 1} of what it did from there."
        elif way.plain is None:
            before = len(trace) - change.at
            how = (
                f"{held}, starting {before} step{'' if before == 1 else 's'} before the life went."
            )
        else:
            # The plain way (T-1012): not what it did, but its best-paid button held.
            how = f"holding {space[way.plain].label} instead of what it did"
            if way.lost_at is None and change.move == way.plain:
                how += "."
            elif way.lost_at is None:
                how += f", with {held} from step {change.at + 1} of that."
            else:
                before = way.lost_at - change.at
                how += (
                    f", with {held}, starting {before} step{'' if before == 1 else 's'} "
                    "before that would lose the life."
                )
        log.info("%s%s", lead, how)
        return True

    def go_back(self, moment: Moment, then: Sequence[int] = ()) -> State:
        """Put the game back to a saved moment of this go and carry on from there.

        The jump is the app's doing, not the game's: nothing is paid, lost or taken
        for a restart across it. The steps still waiting for their lookahead are
        learned as final first, because the story they belonged to ends here. The
        moves in ``then`` are played next, one a step, through the ordinary step
        (looked at, paid and learned from) without asking the learner. Returns the
        state the go is in again.
        """
        if self._state is None or self._ended_by is not None:
            raise ConfigurationError("There is no go running to put back.")
        moments = self._moments
        if moments is None or not any(kept is moment for kept in moments.kept):
            raise ConfigurationError("That is not a saved moment of this go.")
        size = len(self._agent.space)
        moves = tuple(then)
        for move in moves:
            if isinstance(move, bool) or not isinstance(move, int) or not 0 <= move < size:
                raise ConfigurationError(f"The controls have no move {move!r}.")
        self._connection.call(lambda a: a.load_state(moment.slot))
        self._observation.expect_jump(back=True)  # nobody's event (T-1004)
        latest = self._observation.observe()
        snapshot = latest.snapshot if latest is not None else self._observation.latest
        # Whatever the jump looked like to the reward and goal services is dropped.
        self._signals.take()
        self._signals.take_setbacks()
        self._signals.take_scores()
        self._flush(self._state, done=True)
        if snapshot is not None:
            self._state = self._encoder.encode(snapshot)
        self._states.add(self._state.key)
        self._same_state = 0
        self._seen = None
        if self._watch is not None:
            if self._unseen_lost:
                self._watch.told(lost=self._unseen_lost)  # no look will come for it now
            if isinstance(moment.mark, Mark):
                self._watch.back_to(moment.mark)
        self._unseen_lost = self._unseen_scored = 0
        if self._route is not None:
            self._route.went_back(moments.steps - moment.step)
        moments.went_back(moment)
        self._stood_at = None  # wherever it stood, the game is somewhere else now (T-1018)
        self._stood_for = 0
        self._stood_spots.clear()
        self._forced = deque(moves)
        self._route_left = 0
        self._pressing = False
        self._times_back += 1
        return self._state

    def observe_step(self, buttons: Iterable[str], frames: int) -> bool:
        """Learn from a step someone else drove (the child, via the play clock).

        ``buttons`` are what was held and ``frames`` how many frames passed since the
        last step; the action learned is the one in the space that matches the buttons.
        Returns False once the episode is over.
        """
        if self._state is None:
            raise ConfigurationError("Call begin() before observe_step().")
        if self._ended_by is not None:
            return False
        if frames < 1:
            raise ConfigurationError("frames must be at least 1.")
        action = self._agent.space.matches(buttons).index
        self._last_buttons = self._agent.space.buttons_for(action)
        self._last_action = action
        self._forced.clear()  # somebody else is playing: no list of moves holds (T-1004)
        self._route_left = 0
        self._pressing = False
        if self._moments is not None:
            self._moments.forget()
        self._bus.publish(ButtonsHeld(self._observation.frame, self._last_buttons, "child"))
        return self._settle(self._state, action, frames, "child")

    def _settle(self, state: State, action: int | None, frames: int, source: str) -> bool:
        """After the frames passed: collect the reward, read the next state, learn."""
        self._frames += frames
        lost = self._signals.take_setbacks()
        scored = self._signals.take_scores()
        if lost:
            self._setbacks += lost
            if self._first_setback_step is None:
                self._first_setback_step = self._steps + 1  # this step, 1-based (T-934)
            if self._deaths is not None:  # the sight a lookahead before the loss (T-968)
                before = self._pending[0][0] if self._pending else state
                for _ in range(lost):
                    self._deaths.death(before)
        if self._deaths is not None:
            self._deaths.step(state)
        if self._watch is not None:
            # The watch looks every few steps but lives go on any step: what happened
            # since its last look is told with the next one (T-988; it used to be dropped).
            self._unseen_lost += lost
            self._unseen_scored += scored
            if self._seen is not None:
                # The memory read with this screen goes with it (T-1020): the watch
                # finds in it the game's own count of where its camera is.
                read = self._observation.latest
                with app_work.timed("step: look"):
                    self._watch.look(
                        self._seen,
                        lost=self._unseen_lost,
                        scored=self._unseen_scored,
                        memory=read.regions if read is not None else None,
                    )
                self._unseen_lost = self._unseen_scored = 0
                # New ground the look found is paid before this step's reward is
                # collected, so it lands in the step that got there (T-993).
                steps = self._watch.take_new_ground()
                if steps:
                    self._bus.publish(GroundCovered(self._observation.frame, steps))
                if self._stand_after and self._moments is not None and self._ways_per_go:
                    self._note_place(steps)
        self._stood_for += 1
        reward, resets = self._signals.take()
        if reward > 0:
            self._unpaid = 0
            self._press_next = 0
            self._pressing = False
            if action is not None:
                for name in self._agent.space.buttons_for(action):
                    self._paid[name] = self._paid.get(name, 0.0) + reward
        else:
            self._unpaid += 1
        self._seen = None
        self._last_state = state
        self._last_reward = reward
        # Putting the game back to the checkpoint looks like a restart to the engine, and
        # it may only notice a few frames later: a "reset" that soon after a restore is
        # the restore itself, not the game starting over.
        if self._restored and self._frames <= self._settle_frames:
            resets = 0
        snapshot = self._observation.latest
        with app_work.timed("step: encode"):
            next_state = self._encoder.encode(snapshot) if snapshot is not None else state
        done = resets > 0
        with app_work.timed("step: learn"):
            if action is None:
                self._flush(state, done=False)  # an unstick step is nobody's lesson (T-963)
            else:
                self._learn(state, action, reward, next_state, done)
                if self._on_learned is not None:
                    self._on_learned(Transition(state, action, reward, next_state, done, source))
        self._reward += reward
        self._steps += 1
        if self._route is not None:
            # What the story has been paid so far (T-1015): the person's rewards, so
            # the way remembered is the best go by what they pay for.
            self._rows_gone()
            self._route.did(
                action if source == "ai" else None,
                self._route.counted + reward,
                self._signals.parts,  # what paid this step (T-1027)
            )
        if action is not None:
            self._actions_seen.add(action)
        if len(self._fingerprint) < FINGERPRINT_STEPS:
            self._fingerprint.append(next_state.key)
        self._same_state = self._same_state + 1 if next_state == state else 0
        if not self._same_state:
            self._pressed_stuck = False
        self._state = next_state
        self._states.add(next_state.key)
        if done:
            # A life lost that also reads as the game starting over (it put the player
            # back at the start) ended the go before it could go back (T-1013). Where a
            # search can follow, the end waits for it: step() ends the go if none is found.
            if (
                lost
                and source == "ai"
                and self.own_steps < self._max_steps
                and self._way_back() is not None
            ):
                self._held_reset = True
            else:
                self._ended_by = "reset"
        elif self.own_steps >= self._max_steps and not (self._way_left and self._forced):
            # The steps of the way it remembers are not the go's own (T-1015): a long way
            # would leave a go no steps to get any further in. And a way a search has
            # just found is played to its end first (T-1018): a go that ran out part of
            # the way through it had gone back over ground it was paid for and ended
            # short of it again, so nothing of the way was remembered and the next go
            # did the same, search and all.
            self._ended_by = "max_steps"
        if self._ended_by is not None:
            self._signals.stop()
            return False
        return True

    def _watch_deaths(self, knowledge: Knowledge) -> None:
        """Count this go's deaths on the values the person penalises, else the AI's best
        life and health guesses (T-940, T-964); the log names them when they change."""
        asked = self._watched_setbacks() if self._watched_setbacks is not None else frozenset()
        yours = frozenset(i for i in asked if i in knowledge)
        if yours:
            ids = yours
        else:
            best = (knowledge.best(VariableKind.LIVES), knowledge.best(VariableKind.HEALTH))
            ids = frozenset(v.id for v in best if v is not None)
        self._signals.watch_setbacks(ids)
        if ids == self._watched_last:
            return
        self._watched_last = ids
        if not ids:
            log.info("Lives lost are not counted: the AI has no life count or health guess yet.")
            return
        names = ", ".join(
            v.display_name if not v.name else f"{v.name} at {v.region}[{v.offset}]"
            for v in sorted((knowledge.get(i) for i in ids), key=lambda v: (v.region, v.offset))
        )
        log.info(
            "Lives lost are counted on %s, %s.",
            names,
            "which your reward penalises"
            if yours
            else "the AI's best life count and health guesses",
        )

    # ------------------------------------------------------------ learning (T-963)

    def _learn(
        self, state: State, action: int, reward: float, next_state: State, done: bool
    ) -> None:
        """Queue the step; learn the oldest queued step once the lookahead is full.

        Each step is learned from the discounted sum of the rewards of the steps
        queued after it and the situation after them, so credit travels the whole
        lookahead in one pass. A reset settles every queued step as final.
        """
        self._pending.append((state, action, reward))
        if done:
            self._flush(next_state, done=True)
        elif len(self._pending) >= self._lookahead or self._gamma() is None:
            self._learn_oldest(next_state, done=False)

    def _flush(self, next_state: State, *, done: bool) -> None:
        while self._pending:
            self._learn_oldest(next_state, done=done)

    def _learn_oldest(self, next_state: State, *, done: bool) -> None:
        gamma = self._gamma()
        spanned = len(self._pending) if gamma is not None else 1
        total, weight = 0.0, 1.0
        for _, _, reward in list(self._pending)[:spanned]:
            total += weight * reward
            weight *= gamma if gamma is not None else 1.0
        state, action, _ = self._pending.popleft()
        self._agent.learn(state, action, total, next_state, done, steps=spanned)

    def _gamma(self) -> float | None:
        """The agent's discount, or None for one that learns no returns (a race's player)."""
        settings = getattr(self._agent, "settings", None)
        return settings.gamma if isinstance(settings, Settings) else None

    def finish(self, *, ended_by: str | None = None) -> EpisodeResult:
        """Release the buttons, count the episode, and report."""
        if self._state is None:
            raise ConfigurationError("No episode to finish.")
        if self._ended_by is None:
            self._ended_by = ended_by or "stopped"
        self._signals.stop()
        if self._connection.is_connected:
            try:
                self._connection.call(lambda a: a.set_input([]))
                self._observation.record_input([], "ai")
            except EmulatorError:
                pass  # the connection service reports link loss
        if self._watch is not None and self._unseen_lost:
            # A life that went on the go's last step, after the last look (T-988).
            self._watch.told(lost=self._unseen_lost)
        self._unseen_lost = self._unseen_scored = 0
        if self._pending:  # the go ended without a reset: the last steps bootstrap on (T-963)
            assert self._state is not None
            self._flush(self._state, done=False)
        self._agent.end_episode(self._reward)  # exploring follows results (T-932)
        self._last_buttons = frozenset()
        if self._route is not None:
            self._rows_gone()
            had, worth = len(self._route.moves), self._route.paid
            if self._route.end():
                log.info(
                    "The AI did better than the way it remembered: it now remembers %d steps "
                    "from the start, and its next go begins by playing them again.",
                    len(self._route.moves),
                )
            elif self._route.shortened and self._route.moves:
                log.info(
                    "Played again, the way the AI remembers was paid less than the %.1f "
                    "points it was remembered for: something that paid for it then pays no "
                    "longer. It now remembers %d step%s of it instead of %d, up to where it "
                    "was paid the most this time (%.1f points).",
                    worth,
                    len(self._route.moves),
                    "" if len(self._route.moves) == 1 else "s",
                    had,
                    self._route.paid,
                )
            elif self._route.shortened:
                log.info(
                    "Played again, the way the AI remembers was paid nothing (it was "
                    "remembered for %.1f points): what paid for it then pays no longer. "
                    "The AI forgets that way, and its next go begins without one.",
                    worth,
                )
        result = EpisodeResult(
            self._steps,
            self._frames,
            self._reward,
            self._ended_by,
            len(self._states),
            self._signals.completed,
            self._signals.earned,
            self._setbacks,
            self._first_setback_step,
            self._ways_sought,
            self._ways_found,
            self._pressed_on,
            self._remembered,
            self._ways_stood,
        )
        self._state = None
        return result

    def run_episode(self) -> EpisodeResult:
        self.begin()
        while self.step():
            pass
        return self.finish()


__all__ = [
    "DEFAULT_FRAMES_PER_STEP",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_SETTLE_FRAMES",
    "LOOKAHEAD_STEPS",
    "NOW_SLOT",
    "PRESS_FOR",
    "Earning",
    "EpisodeResult",
    "EpisodeRunner",
    "EpisodeSignals",
    "describe_earned",
]
