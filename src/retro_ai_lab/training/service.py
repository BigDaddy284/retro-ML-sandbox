"""Training service: modes, start / pause / stop, and the history of goes.

DESIGN.md §8 gives the child three modes. In *independent* and *guided* mode
the AI plays (guided only differs in what the child may touch meanwhile); in
*manual* mode the child plays and the AI learns from the demonstration. §9
asks for live and background training: the service is step-driven, so a
GUI timer can step it at game speed (LIVE) or as fast as it can (FAST), and a
script can call :meth:`TrainingService.train` with no GUI at all.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_ai_lab.core import ConfigurationError, EmulatorError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.core.worklog import app_work
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.demonstrations import CHILD, Demonstrations
from retro_ai_lab.learning.greedy import GreedyPlayer
from retro_ai_lab.learning.learner import (
    AUTO,
    Q_TABLE,
    Decision,
    Learner,
    algorithm_choices,
    describe_algorithm,
    make_learner,
)
from retro_ai_lab.learning.promotion import convert, promote
from retro_ai_lab.learning.qlearning import QLearningAgent, Settings
from retro_ai_lab.learning.replay import Transition
from retro_ai_lab.learning.skills import SkillsLearner
from retro_ai_lab.learning.state import State, StateEncoder
from retro_ai_lab.rewards.exploitation import ExploitWarning
from retro_ai_lab.rewards.goals import Goal
from retro_ai_lab.rewards.model import RewardProfile
from retro_ai_lab.services.connection import (
    ConnectionService,
    EmulatorConnected,
    EmulatorDisconnected,
)
from retro_ai_lab.services.discovery import DiscoveryService, game_id_for
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.services.goals import GoalService
from retro_ai_lab.services.observer import MemoryObserved, ObservationService
from retro_ai_lab.storage.brains import BrainStore
from retro_ai_lab.training.brain import Brain
from retro_ai_lab.training.curve import DEFAULT_WINDOW, LearningCurve
from retro_ai_lab.training.episode import (
    DEFAULT_FRAMES_PER_STEP,
    DEFAULT_MAX_STEPS,
    DEFAULT_UNSTICK_AFTER,
    EpisodeResult,
    EpisodeRunner,
    describe_earned,
)
from retro_ai_lab.training.insights import (
    BehaviourChange,
    Experiment,
    compare_policies,
    policy_of,
    suggest_experiments,
)
from retro_ai_lab.training.records import EpisodeRecord, Mode, Speed, Status
from retro_ai_lab.understanding.wakeup import find_wake_button, probe_inputs

CHECKPOINT_SLOT = "training"
# Two goes this long with the same states although the AI pressed different buttons
# mean the game is not listening (a title-screen demo); then it goes looking for the
# button that gets the game going (T-844).
IDENTICAL_GO_STEPS = 8
# Goes ended by a reset this early, this many times in a row, blaming the same values,
# mean those values cycle back to their start on their own (T-845; the window was 3
# steps until T-867, when two flags ended 88 goes of 7 to 80 steps in a row).
RESET_SUSPECT_STEPS = 120
RESET_SUSPECT_GOES = 3
log = logging.getLogger(__name__)

DEAF_TRIES = 3  # searches for a wake button before training stops (T-852)
DEAF_WAIT = 300  # frames the game runs between those searches (an intro may be playing)
NOT_LISTENING = (
    "The game is ignoring the AI and no button gets it going. Get the game going "
    "yourself, or pin a spot inside it, then press Start again."
)
FAST_SPEED = 64.0  # EmuHawk's client.speedmode tops out at 6400% (T-836)
PRACTISE_SLOT = "practise"  # the save-state slot of the spot the child pinned (T-508)
DEFAULT_HISTORY = 200
DEFAULT_PROMOTE_AT = 1000  # known situations before an "auto" brain moves to a network


@dataclass(frozen=True, slots=True)
class TrainingStateChanged:
    status: Status
    mode: Mode
    reason: str


@dataclass(frozen=True, slots=True)
class EpisodeFinished:
    record: EpisodeRecord


@dataclass(frozen=True, slots=True)
class TrainingStepped:
    """One step happened: what was pressed and, for the AI's own moves, why."""

    frame: int
    buttons: frozenset[str]
    decision: Decision | None  # None when the child played or Start was pressed to unstick
    step: int = 0  # the step's number within the go (1-based)
    reward: float = 0.0  # what the step collected
    situation: str = ""  # the state key the move was chosen in


class TrainingService:
    def __init__(
        self,
        connection: ConnectionService,
        observation: ObservationService,
        discovery: DiscoveryService,
        bus: EventBus,
        *,
        frames_per_step: int = DEFAULT_FRAMES_PER_STEP,
        max_steps: int = DEFAULT_MAX_STEPS,
        history: int = DEFAULT_HISTORY,
        checkpoint: bool = True,
        unstick_after: int | None = DEFAULT_UNSTICK_AFTER,
        settings: Settings | None = None,
        seed: int | None = None,
        store: BrainStore | None = None,
        algorithm: str = AUTO,
        promote_at: int = DEFAULT_PROMOTE_AT,
        goals: GoalService | None = None,
    ) -> None:
        _check_algorithm(algorithm)
        if promote_at < 1:
            raise ConfigurationError("promote_at must be at least 1.")
        if frames_per_step < 1:
            raise ConfigurationError("frames_per_step must be at least 1.")
        if max_steps < 1:
            raise ConfigurationError("max_steps must be at least 1.")
        if history < 1:
            raise ConfigurationError("history must keep at least one go.")
        self._connection = connection
        self._observation = observation
        self._discovery = discovery
        self._bus = bus
        self._frames_per_step = frames_per_step
        self._max_steps = max_steps
        self._default_max_steps = max_steps  # a game with no choice of its own uses these (T-873)
        self._default_unstick_after = unstick_after
        self._use_checkpoint = checkpoint
        self._unstick_after = unstick_after
        self._settings = settings or Settings()
        self._seed = seed
        self._algorithm = algorithm
        self._promote_at = promote_at
        self._last_change = ""
        self._demonstrations = Demonstrations(seed=seed)
        self._goals = goals
        self._store = store
        self._history: deque[EpisodeRecord] = deque(maxlen=history)
        self._policies: deque[tuple[int, dict[str, int]]] = deque(maxlen=DEFAULT_WINDOW + 1)
        self._game_id: str | None = None
        self._saves = 0
        self._fresh_start = ""
        self._agent: Learner | None = None
        self._encoder: StateEncoder | None = None
        self._harness: ExperimentHarness | None = None
        self._runner: EpisodeRunner | None = None
        self._status = Status.IDLE
        self._mode = Mode.GUIDED
        self._speed = Speed.LIVE
        self._fast_applied = False  # the emulator is running at FAST_SPEED for us (T-836)
        self._enabled = False
        self._episodes = 0
        self._total_steps = 0
        self._manual_frames = 0
        self._last_frame = 0
        self._busy = False
        self._spot_pinned = False
        self._wake_button: str | None = None
        self._needs_wake = False
        self._probe_listening = False  # ask the game directly at the next go start (T-848)
        self._last_go: tuple[tuple[str, ...], bool] | None = None
        self._short_resets: list[tuple[str, ...]] = []
        bus.subscribe(EmulatorConnected, self._on_connected)
        bus.subscribe(EmulatorDisconnected, self._on_disconnected)
        bus.subscribe(MemoryObserved, self._on_memory)

    # ------------------------------------------------------------ queries

    @property
    def algorithm(self) -> str:
        """How the AI learns: a registered algorithm, or ``auto`` (table first, then network)."""
        return self._algorithm

    @property
    def promote_at(self) -> int:
        """Known situations at which an ``auto`` brain moves from the table to a network."""
        return self._promote_at

    @property
    def demonstrations(self) -> Demonstrations:
        """The child's moves kept for this game (empty between games)."""
        return self._demonstrations

    @property
    def last_change(self) -> str:
        """The last promotion or conversion of the brain, in plain words ("" if none)."""
        return self._last_change

    @property
    def enabled(self) -> bool:
        """True while the connected emulator lets the AI see, press and step."""
        return self._enabled

    @property
    def status(self) -> Status:
        return self._status

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def speed(self) -> Speed:
        return self._speed

    @property
    def ai_in_control(self) -> bool:
        """True while a go is running or paused in a mode where the AI plays."""
        return self._status is not Status.IDLE and self._mode.ai_plays

    @property
    def agent(self) -> Learner | None:
        return self._agent

    @property
    def space(self) -> ActionSpace | None:
        return self._agent.space if self._agent is not None else None

    @property
    def encoder(self) -> StateEncoder | None:
        return self._encoder

    @property
    def episode(self) -> EpisodeRunner | None:
        """The go in progress, if one is."""
        return self._runner if self._runner is not None and self._runner.running else None

    @property
    def history(self) -> tuple[EpisodeRecord, ...]:
        return tuple(self._history)

    @property
    def episodes(self) -> int:
        """Goes finished since connect."""
        return self._episodes

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def last(self) -> EpisodeRecord | None:
        return self._history[-1] if self._history else None

    @property
    def frames_per_step(self) -> int:
        return self._frames_per_step

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def unstick_after(self) -> int | None:
        """Steps with nothing changing before a go gives up (None: never)."""
        return self._unstick_after

    def set_go_length(self, steps: int) -> None:
        """How many steps a go lasts at most, for this game from now on; idle only (T-873)."""
        if not isinstance(steps, int) or steps < 1:
            raise ConfigurationError("A go must last at least one step.")
        self._change_go_settings(max_steps=steps)

    def set_give_up_after(self, steps: int | None) -> None:
        """Steps with nothing changing before a go gives up (None: never); idle only (T-873)."""
        if steps is not None and (not isinstance(steps, int) or steps < 1):
            raise ConfigurationError("Giving up takes at least one step, or never.")
        self._change_go_settings(unstick_after=steps)

    def _change_go_settings(self, **changes: int | None) -> None:
        if self._status is not Status.IDLE:
            raise ConfigurationError("Stop training before changing how long a go lasts.")
        for name, value in changes.items():
            setattr(self, f"_{name}", value)
        self._runner = None  # the next go is built with the new settings
        if self.brain() is not None:
            self.save()
        self._announce("go settings")

    @property
    def harness(self) -> ExperimentHarness | None:
        """The checkpoint harness for the connected game (None without save states)."""
        return self._harness

    @property
    def store(self) -> BrainStore | None:
        return self._store

    @property
    def wake_button(self) -> str | None:
        """The button the AI found gets this game going (T-844), if it has looked."""
        return self._wake_button

    @property
    def spot_pinned(self) -> bool:
        """True while a pinned spot (DESIGN.md §9 checkpoint training) is in force."""
        return self._spot_pinned

    @property
    def game_id(self) -> str | None:
        return self._game_id

    @property
    def saves(self) -> int:
        """Times the brain was written to disk since connect."""
        return self._saves

    @property
    def fresh_start(self) -> str:
        """Why a saved brain wasn't used this time (empty when it was, or there was none)."""
        return self._fresh_start

    def brain(self) -> Brain | None:
        """What the AI knows about the connected game; None when disconnected."""
        if not self._enabled or self._agent is None or self._game_id is None:
            return None
        return Brain(
            self._game_id,
            self._agent,
            tuple(self._history),
            self._episodes,
            self._total_steps,
            self._demonstrations,
            self._max_steps if self._max_steps != self._default_max_steps else None,
            self._unstick_after if self._unstick_after != self._default_unstick_after else None,
        )

    def curve(self, window: int = DEFAULT_WINDOW) -> LearningCurve:
        """How the goes kept in the history have been going."""
        return LearningCurve(self._history, window)

    def behaviour_changes(
        self, window: int = DEFAULT_WINDOW
    ) -> tuple[tuple[BehaviourChange, ...], int, int]:
        """(changes, goes compared over, situations known now) against the policy
        ``window`` goes ago, or the oldest snapshot kept."""
        if self._agent is None or not self._policies:
            return (), 0, 0
        now = policy_of(self._agent)
        candidates = [p for p in self._policies if p[0] <= self._episodes - window]
        then_number, then = candidates[-1] if candidates else self._policies[0]
        goes_ago = self._episodes - then_number
        if goes_ago <= 0:
            return (), 0, len(now)
        return compare_policies(then, now), goes_ago, len(now)

    def experiments(
        self,
        warning: ExploitWarning | None = None,
        goal: Goal | None = None,
        profile: RewardProfile | None = None,
    ) -> tuple[Experiment, ...]:
        """Things worth trying next, from the curve, the agent and what the caller knows."""
        return suggest_experiments(self.curve(), self._agent, warning, goal, profile)

    def summary(self) -> str:
        if not self._enabled:
            return "Connect a game to start training."
        if self._status is Status.IDLE:
            if self._episodes:
                return f"Resting after {self._episodes} {_goes(self._episodes)}."
            return "Ready to train."
        doing = "watching you play" if self._mode is Mode.MANUAL else "playing"
        if self._status is Status.PAUSED:
            doing = "paused"
        go = self._episodes + 1
        runner = self._runner
        steps = runner.steps if runner is not None and runner.running else 0
        points = runner.reward if runner is not None and runner.running else 0.0
        text = f"Go {go}: {doing}, step {steps}, {points:+.0f} points so far."
        if self._mode is Mode.MANUAL and self._demonstrations.shown:
            n = self._demonstrations.shown
            text += f" You have shown the AI {n} move{'s' if n != 1 else ''}."
        return text

    # ------------------------------------------------------------ controls

    def pin_spot(self) -> None:
        """Keep the game's current position: every go and every Start restarts here.

        DESIGN.md §9 "Checkpoint Training": the child plays to the hard part once, pins
        it, and the AI practises that section until the spot is forgotten.
        """
        if not self._enabled:
            raise ConfigurationError("Connect a game before picking a spot to practise.")
        if self._status is not Status.IDLE:
            raise ConfigurationError("Stop training before picking a spot to practise.")
        harness = self._harness
        if harness is None or not harness.available():
            raise UnsupportedCapabilityError(
                "save_states", "This console can't keep a spot: it has no save states."
            )
        self._connection.call(lambda a: a.save_state(PRACTISE_SLOT))
        self._spot_pinned = True
        self._announce("spot pinned")

    def unpin_spot(self) -> None:
        """Forget the pinned spot; goes start wherever the game is again."""
        if not self._spot_pinned:
            return
        self._spot_pinned = False
        self._announce("spot forgotten")

    def _restore_spot(self) -> None:
        if self._spot_pinned and self._connection.is_connected:
            self._connection.call(lambda a: a.load_state(PRACTISE_SLOT))

    def set_mode(self, mode: Mode) -> None:
        if not isinstance(mode, Mode):
            raise ConfigurationError("mode must be a Mode.")
        if mode is self._mode:
            return
        was_running = self._status is not Status.IDLE
        if was_running:
            self._close_episode("mode change")
        self._mode = mode
        if was_running:
            self._prepare_checkpoint()
        self._apply_speed()
        self._announce("mode change")

    @property
    def settings(self) -> Settings:
        """The learning settings the brain in use learns with (new brains start from them)."""
        if self._agent is not None:
            found = getattr(self._agent, "settings", None)
            if isinstance(found, Settings):
                return found
        return self._settings

    def set_settings(self, settings: Settings) -> None:
        """Learn with these settings from now on; idle only. Applied to the brain in use."""
        if not isinstance(settings, Settings):
            raise ConfigurationError("Learning settings must be a Settings.")
        if self._status is not Status.IDLE:
            raise ConfigurationError("Stop training before changing the learning settings.")
        self._settings = settings
        if self._agent is not None:
            self._agent.configure(settings)
            self.save()
        self._announce("settings")

    def current_state(self) -> State | None:
        """The situation the game is in now, as the AI sees it (None without a game)."""
        if not self._enabled or self._encoder is None:
            return None
        snapshot = self._observation.latest
        if snapshot is None:
            return None
        self._encoder.refresh()  # discoveries may have changed since the last go
        return self._encoder.encode(snapshot)

    def set_algorithm(self, algorithm: str) -> str:
        """Choose how the AI learns; converts the current brain when the choice differs.

        Returns what happened to the brain in plain words ("" when nothing changed).
        """
        _check_algorithm(algorithm)
        if self._status is not Status.IDLE:
            raise ConfigurationError("Stop training before changing how the AI learns.")
        self._algorithm = algorithm
        if self._agent is None or algorithm == AUTO or self._agent.algorithm == algorithm:
            return ""
        was = describe_algorithm(self._agent.algorithm)[0].lower()
        self._agent = convert(self._agent, algorithm, seed=self._seed)
        self._runner = None
        now = describe_algorithm(algorithm)[0].lower()
        self._last_change = f"The AI moved from the {was} to the {now}, keeping what it learned."
        self.save()
        self._announce("algorithm change")
        return self._last_change

    def rehearse(self, passes: int = 1) -> int:
        """Let the AI learn from the child's moves again; returns the steps learned."""
        if self._status is not Status.IDLE:
            raise ConfigurationError("Stop training before the AI rehearses your moves.")
        if self._agent is None or not self._enabled:
            return 0
        steps = self._demonstrations.rehearse(self._agent, passes=passes)
        if steps:
            self.save()
        return steps

    def set_speed(self, speed: Speed) -> None:
        if not isinstance(speed, Speed):
            raise ConfigurationError("speed must be a Speed.")
        if speed is self._speed:
            return
        self._speed = speed
        self._apply_speed()
        self._announce("speed")

    def _apply_speed(self) -> None:
        """Run the emulator flat out while the AI practises fast, at 1x otherwise (T-836).

        Each training step is a round trip whose frames run at the emulator's own
        throttle, so "Practise fast" was no faster than watching. An emulator
        without speed control (mGBA) is left alone; a refused call is logged.
        """
        wanted = (
            self._status is Status.RUNNING and self._mode.ai_plays and self._speed is Speed.FAST
        )
        if wanted == self._fast_applied or not self._connection.is_connected:
            return
        try:
            if not self._connection.call(lambda a: a.capabilities().speed_control):
                return
            self._connection.call(lambda a: a.set_speed(FAST_SPEED if wanted else 1.0))
        except (EmulatorError, UnsupportedCapabilityError) as exc:
            log.warning("Could not change the emulator's speed: %s", exc)
            return
        self._fast_applied = wanted

    def start(self, mode: Mode | None = None) -> None:
        if not self._enabled:
            raise ConfigurationError(
                "Can't train yet: connect a game whose memory, buttons and frames the app can see."
            )
        if self._status is not Status.IDLE:
            raise ConfigurationError("Training is already going; pause or stop it first.")
        if mode is not None:
            if not isinstance(mode, Mode):
                raise ConfigurationError("mode must be a Mode.")
            self._mode = mode
        self._prepare_checkpoint()
        if not self._enabled:  # the link dropped while the checkpoint was being taken
            raise ConfigurationError("The game went away just as training was starting.")
        self._probe_listening = True  # the first go asks whether the game listens (T-848)
        self._status = Status.RUNNING
        self._discovery.set_experiments_allowed(False)  # not over the game's link (T-839)
        self._apply_speed()
        self._announce("start")

    def pause(self) -> None:
        if self._status is not Status.RUNNING:
            raise ConfigurationError("Nothing is running to pause.")
        self._status = Status.PAUSED
        self._apply_speed()
        self._announce("pause")

    def resume(self) -> None:
        if self._status is not Status.PAUSED:
            raise ConfigurationError("Training isn't paused.")
        self._status = Status.RUNNING
        self._apply_speed()
        self._announce("resume")

    def stop(self) -> None:
        if self._status is Status.IDLE:
            raise ConfigurationError("Nothing is running to stop.")
        self._close_episode("stopped")
        self._status = Status.IDLE
        self._discovery.set_experiments_allowed(True)
        self._apply_speed()
        self._announce("stop")

    def save(self) -> Path | None:
        """Write the brain to disk now. None without a store or a brain."""
        brain = self.brain()
        if self._store is None or brain is None:
            return None
        path = self._store.save(brain.to_dict())
        self._saves += 1
        return path

    def replace_brain(self, data: dict[str, Any]) -> Brain:
        """Swap in a brain for the connected game (loading a saved AI); idle only."""
        if self._status is not Status.IDLE:
            raise ConfigurationError("Stop training before loading another brain.")
        if self.brain() is None or self._agent is None or self._game_id is None:
            raise ConfigurationError("Connect to a game before loading a brain.")
        brain = Brain.from_dict(data, space=self._agent.space)
        if brain.game_id != self._game_id:
            raise ConfigurationError(
                f"That brain is for {brain.game_id!r}, not the connected game."
            )
        self._history.clear()
        self._policies.clear()
        self._runner = None
        self._fresh_start = ""
        self._adopt(brain)
        self.save()
        self._announce("loaded")
        return brain

    def _adopt(self, brain: Brain) -> None:
        self._agent = brain.agent
        self._demonstrations = (
            brain.demonstrations
            if brain.demonstrations is not None
            else Demonstrations(seed=self._seed)
        )
        self._history.extend(brain.goes)
        self._episodes = brain.episodes
        self._total_steps = brain.total_steps
        if brain.go_length is not None:
            self._max_steps = brain.go_length
        if brain.give_up_after is not None:
            self._unstick_after = brain.give_up_after

    def forget(self) -> None:
        """Start the AI from scratch for this game and delete what was saved."""
        if self._status is not Status.IDLE:
            raise ConfigurationError("Stop training before starting the AI from scratch.")
        if self.brain() is None or self._agent is None or self._game_id is None:
            raise ConfigurationError("There is no brain to forget.")
        self._agent = self._new_agent(self._agent.space)
        self._demonstrations.clear()
        self._history.clear()
        self._policies.clear()
        self._episodes = 0
        self._total_steps = 0
        self._runner = None
        self._fresh_start = ""
        if self._store is not None:
            self._store.delete(self._game_id)
        self._announce("forget")

    def step(self) -> bool:
        """One act-advance-learn step in an AI mode. Returns True if a step was taken.

        Begins a new go when none is running and closes one that has just ended.
        """
        if self._status is not Status.RUNNING or not self._mode.ai_plays:
            return False
        runner = self._ensure_runner()
        if runner is None:
            return False
        self._pick_skill()
        before = runner.steps
        with app_work.timed("ai step"):  # the emulator's share is in the link log (T-894)
            went_on = runner.step()
        if runner.steps > before:
            self._total_steps += 1
            decision = self._agent.last_decision if self._agent is not None else None
            if runner.last_action is None:
                decision = None
            self._bus.publish(_stepped(self._observation.frame, runner, decision))
        # Link loss closes the go from the disconnect handler, so check it's still open.
        if not went_on and runner.state is not None:
            self._record(runner.finish())
        return True

    def run_for(self, seconds: float, *, max_steps: int | None = None) -> int:
        """Take as many steps as fit in ``seconds`` (fast training). Returns the count."""
        if seconds < 0:
            raise ConfigurationError("seconds must be >= 0.")
        if max_steps is not None and max_steps < 1:
            raise ConfigurationError("max_steps must be at least 1 (or None).")
        deadline = time.monotonic() + seconds
        taken = 0
        while (max_steps is None or taken < max_steps) and self.step():
            taken += 1
            if time.monotonic() >= deadline:
                break
        return taken

    def compete(
        self,
        entrants: Sequence[tuple[str, Learner]],
        *,
        goes: int,
        max_steps: int | None = None,
        frames_per_step: int | None = None,
    ) -> dict[str, tuple[EpisodeResult, ...]]:
        """Race ``entrants`` from where the game is now, learning nothing.

        Every entrant plays its best known move for ``goes`` goes from the same
        save state; goal progress is reset before each go so each AI meets the
        same goal; the game is put back at the end. Nothing here touches the
        AI being trained or its brain.
        """
        if goes < 1:
            raise ConfigurationError("goes must be at least 1.")
        if self._status is not Status.IDLE:
            raise ConfigurationError("Stop training before a race.")
        if not self._enabled or self._encoder is None:
            raise ConfigurationError("Connect a game the AI can learn before a race.")
        harness = self._harness
        if harness is None or not harness.available():
            raise UnsupportedCapabilityError(
                "save_states", "This console can't race: it has no save states."
            )
        names = [name for name, _ in entrants]
        if len(set(names)) != len(names):
            raise ConfigurationError("Each entrant needs its own name.")
        harness.checkpoint()
        results: dict[str, tuple[EpisodeResult, ...]] = {}
        self._busy = True
        try:
            for name, learner in entrants:
                player = GreedyPlayer(learner)
                outcomes = []
                for _ in range(goes):
                    if self._goals is not None and self._goals.enabled:
                        self._goals.reset_progress()
                    runner = EpisodeRunner(
                        self._connection,
                        self._observation,
                        self._bus,
                        player,
                        self._encoder,
                        frames_per_step=frames_per_step or self._frames_per_step,
                        max_steps=max_steps or self._max_steps,
                        harness=harness,
                        unstick_after=self._unstick_after,
                    )
                    outcomes.append(runner.run_episode())
                results[name] = tuple(outcomes)
            harness.restore()
            self._restore_spot()
        finally:
            self._busy = False
            self._runner = None
        return results

    def train(self, episodes: int) -> tuple[EpisodeRecord, ...]:
        """Run whole goes with no GUI: start if needed, stop at the end."""
        if episodes < 1:
            raise ConfigurationError("episodes must be at least 1.")
        if not self._mode.ai_plays:
            raise ConfigurationError("The AI can't train by itself while you are the player.")
        if self._status is Status.IDLE:
            self.start()
        elif self._status is Status.PAUSED:
            self.resume()
        before = self._episodes
        while self._episodes < before + episodes and self.step():
            pass
        if self._status is not Status.IDLE:
            self.stop()
        return self.history[-(self._episodes - before) :] if self._episodes > before else ()

    # ------------------------------------------------------------- events

    def _on_connected(self, event: EmulatorConnected) -> None:
        caps = event.capabilities
        knowledge = self._discovery.knowledge
        self._history.clear()
        self._policies.clear()
        self._episodes = 0
        self._total_steps = 0
        self._runner = None
        self._status = Status.IDLE
        self._enabled = (
            caps.memory_read and caps.input and caps.frame_advance and knowledge is not None
        )
        if not self._enabled or knowledge is None:
            self._agent = None
            self._encoder = None
            self._harness = None
            self._game_id = None
            return
        info = event.game_info
        self._game_id = game_id_for(info.system, info.title, info.rom_hash)
        self._saves = 0
        space = ActionSpace.from_buttons(event.buttons)
        self._agent = self._new_agent(space)
        self._demonstrations = Demonstrations(seed=self._seed)
        self._max_steps = self._default_max_steps  # a saved brain may bring its own (T-873)
        self._unstick_after = self._default_unstick_after
        self._fresh_start = ""
        self._last_change = ""
        if self._store is not None and self._store.exists(self._game_id):
            try:
                brain = Brain.from_dict(self._store.load(self._game_id), space=space)
            except ConfigurationError as exc:
                self._fresh_start = f"Couldn't use the saved brain, so starting afresh: {exc}"
            else:
                self._adopt(brain)
        self._encoder = StateEncoder(knowledge)
        self._harness = (
            ExperimentHarness(self._connection, slot=CHECKPOINT_SLOT)
            if self._use_checkpoint and caps.save_states
            else None
        )
        self._wake_button = knowledge.wake_button
        self._needs_wake = False
        self._probe_listening = False
        self._last_go = None
        self._short_resets = []

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        was_running = self._status is not Status.IDLE
        if was_running:
            self._close_episode("disconnected")
        if self._enabled:
            self.save()
        self._enabled = False
        self._harness = None
        self._fast_applied = False  # the script puts the speed back itself
        self._spot_pinned = False  # save states don't outlive the link
        self._status = Status.IDLE
        if was_running:
            self._announce("disconnected")

    def _on_memory(self, event: MemoryObserved) -> None:
        """In manual mode the play clock drives the frames; learn every few of them."""
        if (
            self._busy
            or self._status is not Status.RUNNING
            or self._mode.ai_plays
            or not self._enabled
        ):
            self._last_frame = event.snapshot.frame
            return
        frame = event.snapshot.frame
        if self.episode is None:
            if self._ensure_runner() is None:
                return
            # The frame this event reports is the first one of the new go.
            self._manual_frames = 0
            self._last_frame = frame - 1
        runner = self._runner
        assert runner is not None
        self._manual_frames += max(0, frame - self._last_frame)
        self._last_frame = frame
        if self._manual_frames < self._frames_per_step:
            return
        held = self._observation.inputs.held_at(frame)
        frames, self._manual_frames = self._manual_frames, 0
        self._pick_skill()
        with app_work.timed("learn"):
            went_on = runner.observe_step(held, frames)
        self._total_steps += 1
        self._bus.publish(_stepped(frame, runner, None))
        if not went_on and runner.state is not None:
            self._record(runner.finish())

    # ------------------------------------------------------------ helpers

    def active_skill(self) -> str | None:
        """The goal step the AI is working on (its id), or None without one."""
        if self._goals is None or not self._goals.enabled:
            return None
        goal = self._goals.goal
        if goal is None or goal.is_complete:
            return None
        nxt = goal.next()
        return nxt.id if nxt is not None else None

    def _pick_skill(self) -> None:
        if isinstance(self._agent, SkillsLearner):
            self._agent.set_skill(self.active_skill())

    def _on_learned(self, transition: Transition) -> None:
        if transition.source == CHILD:
            self._demonstrations.add(transition)

    def _new_agent(self, space: ActionSpace) -> Learner:
        name = Q_TABLE if self._algorithm == AUTO else self._algorithm
        return make_learner(name, space, self._settings, seed=self._seed)

    def _maybe_promote(self) -> None:
        """Under ``auto``, move a table that has grown big to a network."""
        agent = self._agent
        if (
            self._algorithm != AUTO
            or not isinstance(agent, QLearningAgent)
            or agent.known < self._promote_at
        ):
            return
        self._agent, _ = promote(agent, seed=self._seed)
        self._runner = None
        self._last_change = (
            f"The game got big ({agent.known} situations), so the AI moved from the simple "
            "table to a neural network, keeping what it learned."
        )
        self._announce("promoted")

    def _ensure_runner(self) -> EpisodeRunner | None:
        if self._runner is not None and self._runner.running:
            return self._runner
        if self._agent is None or self._encoder is None or not self._connection.is_connected:
            return None
        if self._runner is None:
            self._runner = EpisodeRunner(
                self._connection,
                self._observation,
                self._bus,
                self._agent,
                self._encoder,
                frames_per_step=self._frames_per_step,
                max_steps=self._max_steps,
                harness=self._harness,
                unstick_after=self._unstick_after,
                unstick_buttons=(self._wake_button,) if self._wake_button else (),
                on_learned=self._on_learned,
            )
        self._busy = True
        try:
            # Between goes is the moment for a due button experiment (T-839): the
            # game is about to go back to the checkpoint anyway, and no go is mid-step.
            if self._mode.ai_plays and self._harness is not None:
                self._discovery.run_due_experiment()
                if self._probe_listening and not self._needs_wake:
                    with self._flat_out():
                        self._needs_wake = not self._listens()
                self._probe_listening = False
            if self._needs_wake and self._mode.ai_plays:
                with self._flat_out():
                    woken = self._wake_up_or_wait()
                if not woken:
                    return None  # training stopped: no go on a game that ignores the AI
            self._runner.begin(restore=self._mode.ai_plays)
            self._discovery.begin_go()  # what moves from here on was moved by play (T-870)
        except EmulatorError:
            return None
        finally:
            self._busy = False
        return self._runner

    @contextmanager
    def _flat_out(self) -> Iterator[None]:
        """Run the emulator at FAST_SPEED for a probe or search, then put it back (T-853).

        Nobody needs to watch the search at normal speed, and at normal speed its
        five-second steps are slow. Left alone when the emulator has no speed
        control, when it is already fast, or when the call is refused.
        """
        changed = False
        if not self._fast_applied and self._connection.is_connected:
            try:
                if self._connection.call(lambda a: a.capabilities().speed_control):
                    self._connection.call(lambda a: a.set_speed(FAST_SPEED))
                    changed = True
            except (EmulatorError, UnsupportedCapabilityError) as exc:
                log.warning("Could not speed the emulator up for the search: %s", exc)
        try:
            yield
        finally:
            if changed and self._connection.is_connected and not self._fast_applied:
                try:
                    self._connection.call(lambda a: a.set_speed(1.0))
                except (EmulatorError, UnsupportedCapabilityError) as exc:
                    log.warning("Could not put the emulator's speed back: %s", exc)

    def _listens(self) -> bool:
        """From the spot goes start from: do the AI's buttons change anything? (T-848)

        Asked at the first go after Start and after a go that ended stuck or by a
        reset, instead of waiting for goes to prove the game is ignoring the AI
        (a title demo keeps the screen moving for minutes). True when the probe
        can't be run, so nothing changes for an emulator without save states.
        """
        harness = self._harness
        if harness is None or not harness.has_checkpoint or not harness.available():
            return True
        harness.restore()
        try:
            probe = probe_inputs(self._connection, self._playing_buttons())
        except ConfigurationError as exc:
            log.info("Could not check whether the game listens: %s", exc)
            return True
        if not probe.matters:
            # A title demo moves everything by itself and ignores the pad, the real
            # score included: nothing the probe saw says anything about play (T-868).
            return False
        try:  # the same probe shows what moves by itself: doubt those guesses (T-855)
            self._discovery.doubt_self_moving(probe.self_moving, frame=self._observation.frame)
        except (ConfigurationError, UnsupportedCapabilityError) as exc:
            log.info("Could not note the values that move by themselves: %s", exc)
        return True

    def _playing_buttons(self) -> tuple[str, ...]:
        """The buttons the AI plays with (its action space), else all of them."""
        names = tuple(b.name for b in self._connection.call(lambda a: a.buttons()))
        space = self._agent.space if self._agent is not None else None
        if space is None:
            return names
        return tuple(dict.fromkeys(n for i in range(len(space)) for n in space.buttons_for(i)))

    def _wake_up_or_wait(self) -> bool:
        """Get a deaf game going, or stop training rather than play a demo (T-852).

        A failed search is followed by a few seconds of letting the game run (an
        intro may still be playing) and another look; after ``DEAF_TRIES`` searches
        training ends with ``NOT_LISTENING``. Returns False when it ended.
        """
        for attempt in range(1, DEAF_TRIES + 1):
            if self._wake_up() or self._harness is None:
                return True
            if attempt == DEAF_TRIES:
                break
            self._connection.call(lambda a: a.step(DEAF_WAIT))
            self._harness.checkpoint()
            if self._listens():
                return True
        log.warning(NOT_LISTENING)
        self._status = Status.IDLE
        self._discovery.set_experiments_allowed(True)
        self._apply_speed()
        self._announce("not listening")
        return False

    def _wake_up(self) -> bool:
        """The game stopped listening: find the button that gets it going (T-844).

        Runs from where the game is now (the screen that ignores the AI). When a
        button works the game is left running after it and that spot becomes the
        checkpoint, so the goes after this one start inside the game. Returns
        False only when the search ran and found nothing.
        """
        self._needs_wake = False
        self._last_go = None
        harness = self._harness
        if harness is None or not harness.available():
            log.info("The game is not listening to the AI, and this emulator can't try buttons.")
            return True
        names = tuple(b.name for b in self._connection.call(lambda a: a.buttons()))
        playing = self._playing_buttons()
        log.info("The game is not listening to the AI; trying each button to get it going.")
        found = find_wake_button(self._connection, names, playing, first=self._wake_button)
        if found is None:
            return False
        self._wake_button = found
        knowledge = self._discovery.knowledge
        if knowledge is not None and knowledge.wake_button != found:
            knowledge.wake_button = found
            self._discovery.save()
        if self._runner is not None:
            self._runner.set_unstick_buttons((found,))
        harness.checkpoint()
        return True

    def _prepare_checkpoint(self) -> None:
        """Every go in an AI mode restarts from where the child left the game."""
        if self._harness is None or not self._mode.ai_plays:
            return
        try:
            if self._harness.available():
                self._restore_spot()  # a pinned spot is where every go starts
                self._harness.checkpoint()
        except EmulatorError:
            pass  # link loss is announced by the connection service

    def _close_episode(self, ended_by: str) -> None:
        runner = self._runner
        if runner is not None and runner.running:
            self._record(runner.finish(ended_by=ended_by))

    def _record(self, result: EpisodeResult) -> None:
        self._episodes += 1
        record = EpisodeRecord(self._episodes, self._mode, result, self._observation.frame)
        why = self._runner.last_reset if self._runner is not None else ""
        earners = describe_earned(result.earned, most=None)  # the whole breakdown (T-877)
        log.info(
            "Go %d ended by %s after %d steps (%d frames), %+.1f points%s%s",
            self._episodes,
            result.ended_by,
            result.steps,
            result.frames,
            result.reward,
            f", from {earners}" if earners else "",
            f": {why}" if result.ended_by == "reset" and why else "",
        )
        self._history.append(record)
        self._note_listening(result)
        if not self._needs_wake:  # a go the game ignored says nothing about play (T-870)
            self._discovery.credit_played(frame=self._observation.frame)
        self._note_short_reset(result)
        if result.ended_by in ("stuck", "reset"):
            self._probe_listening = True  # ask the game directly next time (T-848)
        if self._agent is not None:
            self._policies.append((self._episodes, policy_of(self._agent)))
        self._bus.publish(EpisodeFinished(record))
        self._maybe_promote()
        if self._connection.is_connected:
            self.save()

    def _note_listening(self, result: EpisodeResult) -> None:
        """Did this go show the game ignoring the AI? Then the next go starts by waking it."""
        runner = self._runner
        if runner is None or not self._mode.ai_plays:
            return
        this = (runner.fingerprint, runner.actions_varied)
        # A go stuck part-way (the lives ran out) just needs the checkpoint back, which
        # the next go does anyway. One stuck from its first step, or two goes that ran
        # identically, mean the place the goes start from is not listening.
        if runner.stuck_from_start or (
            self._last_go is not None and goes_look_identical(self._last_go, this)
        ):
            self._needs_wake = True
        self._last_go = this

    def _note_short_reset(self, result: EpisodeResult) -> None:
        """Three goes in a row ended at once by the same values: they cycle (T-845)."""
        runner = self._runner
        ids = runner.last_reset_ids if runner is not None else ()
        if result.ended_by != "reset" or result.steps > RESET_SUSPECT_STEPS or not ids:
            self._short_resets = []
            return
        if self._short_resets and self._short_resets[-1] != ids:
            self._short_resets = []
        self._short_resets.append(ids)
        if len(self._short_resets) < RESET_SUSPECT_GOES:
            return
        self._short_resets = []
        try:
            self._discovery.doubt_reset(ids)
        except (ConfigurationError, UnsupportedCapabilityError) as exc:
            log.warning("Couldn't note the cycling values: %s", exc)

    def _announce(self, reason: str) -> None:
        self._bus.publish(TrainingStateChanged(self._status, self._mode, reason))


def goes_look_identical(
    earlier: tuple[tuple[str, ...], bool], later: tuple[tuple[str, ...], bool]
) -> bool:
    """Same states step for step, both long enough, both with the AI varying its buttons."""
    states_a, varied_a = earlier
    states_b, varied_b = later
    return varied_a and varied_b and len(states_a) >= IDENTICAL_GO_STEPS and states_a == states_b


def _stepped(frame: int, runner: EpisodeRunner, decision: Decision | None) -> TrainingStepped:
    state = runner.last_state
    return TrainingStepped(
        frame,
        runner.last_buttons,
        decision,
        step=runner.steps,
        reward=runner.last_reward,
        situation=state.key if state is not None else "",
    )


def _check_algorithm(name: str) -> None:
    if name not in algorithm_choices():
        raise ConfigurationError(
            f"There is no learning algorithm called {name!r} "
            f"(choices: {', '.join(algorithm_choices())})."
        )


def _goes(n: int) -> str:
    return "go" if n == 1 else "goes"


__all__ = [
    "CHECKPOINT_SLOT",
    "IDENTICAL_GO_STEPS",
    "RESET_SUSPECT_GOES",
    "RESET_SUSPECT_STEPS",
    "EpisodeFinished",
    "TrainingService",
    "TrainingStateChanged",
    "TrainingStepped",
    "goes_look_identical",
]
