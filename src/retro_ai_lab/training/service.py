"""Training service: modes, start / pause / stop, and the history of goes.

DESIGN.md §8 gives the child three modes. In *independent* and *guided* mode
the AI plays (guided only differs in what the child may touch meanwhile); in
*manual* mode the child plays and the AI learns from the demonstration. §9
asks for live and background training: the service is step-driven, so a
GUI timer can step it at game speed (LIVE) or as fast as it can (FAST), and a
script can call :meth:`TrainingService.train` with no GUI at all.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from retro_ai_lab.core import ConfigurationError, EmulatorError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.qlearning import QLearningAgent, Settings
from retro_ai_lab.learning.state import StateEncoder
from retro_ai_lab.services.connection import (
    ConnectionService,
    EmulatorConnected,
    EmulatorDisconnected,
)
from retro_ai_lab.services.discovery import DiscoveryService
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.services.observer import MemoryObserved, ObservationService
from retro_ai_lab.training.episode import (
    DEFAULT_FRAMES_PER_STEP,
    DEFAULT_MAX_STEPS,
    DEFAULT_UNSTICK_AFTER,
    EpisodeResult,
    EpisodeRunner,
)

CHECKPOINT_SLOT = "training"
DEFAULT_HISTORY = 200


class Mode(Enum):
    INDEPENDENT = "independent"
    GUIDED = "guided"
    MANUAL = "manual"

    @property
    def label(self) -> str:
        return {
            Mode.INDEPENDENT: "On its own",
            Mode.GUIDED: "With your help",
            Mode.MANUAL: "You play",
        }[self]

    @property
    def description(self) -> str:
        return {
            Mode.INDEPENDENT: "The AI plays and learns by itself.",
            Mode.GUIDED: "The AI plays while you change rewards and goals.",
            Mode.MANUAL: "You play; the AI watches and learns from you.",
        }[self]

    @property
    def ai_plays(self) -> bool:
        return self is not Mode.MANUAL

    @property
    def allows_guidance(self) -> bool:
        """May the child change rewards and goals while a go is running?"""
        return self is not Mode.INDEPENDENT


class Speed(Enum):
    LIVE = "live"  # one step per game tick: watch the AI play
    FAST = "fast"  # as many steps as fit: background practice

    @property
    def label(self) -> str:
        return "Watch it play" if self is Speed.LIVE else "Practise fast"


class Status(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    number: int  # 1-based, since connect
    mode: Mode
    result: EpisodeResult
    frame: int  # observation frame when the go ended


@dataclass(frozen=True, slots=True)
class TrainingStateChanged:
    status: Status
    mode: Mode
    reason: str


@dataclass(frozen=True, slots=True)
class EpisodeFinished:
    record: EpisodeRecord


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
    ) -> None:
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
        self._use_checkpoint = checkpoint
        self._unstick_after = unstick_after
        self._settings = settings or Settings()
        self._seed = seed
        self._history: deque[EpisodeRecord] = deque(maxlen=history)
        self._agent: QLearningAgent | None = None
        self._encoder: StateEncoder | None = None
        self._harness: ExperimentHarness | None = None
        self._runner: EpisodeRunner | None = None
        self._status = Status.IDLE
        self._mode = Mode.GUIDED
        self._speed = Speed.LIVE
        self._enabled = False
        self._episodes = 0
        self._total_steps = 0
        self._manual_frames = 0
        self._last_frame = 0
        self._busy = False
        bus.subscribe(EmulatorConnected, self._on_connected)
        bus.subscribe(EmulatorDisconnected, self._on_disconnected)
        bus.subscribe(MemoryObserved, self._on_memory)

    # ------------------------------------------------------------ queries

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
    def agent(self) -> QLearningAgent | None:
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
        return f"Go {go}: {doing}, step {steps}, {points:+.0f} points so far."

    # ------------------------------------------------------------ controls

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
        self._announce("mode change")

    def set_speed(self, speed: Speed) -> None:
        if not isinstance(speed, Speed):
            raise ConfigurationError("speed must be a Speed.")
        self._speed = speed

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
        self._status = Status.RUNNING
        self._announce("start")

    def pause(self) -> None:
        if self._status is not Status.RUNNING:
            raise ConfigurationError("Nothing is running to pause.")
        self._status = Status.PAUSED
        self._announce("pause")

    def resume(self) -> None:
        if self._status is not Status.PAUSED:
            raise ConfigurationError("Training isn't paused.")
        self._status = Status.RUNNING
        self._announce("resume")

    def stop(self) -> None:
        if self._status is Status.IDLE:
            raise ConfigurationError("Nothing is running to stop.")
        self._close_episode("stopped")
        self._status = Status.IDLE
        self._announce("stop")

    def step(self) -> bool:
        """One act-advance-learn step in an AI mode. Returns True if a step was taken.

        Begins a new go when none is running and closes one that has just ended.
        """
        if self._status is not Status.RUNNING or not self._mode.ai_plays:
            return False
        runner = self._ensure_runner()
        if runner is None:
            return False
        before = runner.steps
        went_on = runner.step()
        self._total_steps += runner.steps - before
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
            return
        space = ActionSpace.from_buttons(event.buttons)
        self._agent = QLearningAgent(space, self._settings, seed=self._seed)
        self._encoder = StateEncoder(knowledge)
        self._harness = (
            ExperimentHarness(self._connection, slot=CHECKPOINT_SLOT)
            if self._use_checkpoint and caps.save_states
            else None
        )

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        was_running = self._status is not Status.IDLE
        if was_running:
            self._close_episode("disconnected")
        self._enabled = False
        self._harness = None
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
        went_on = runner.observe_step(held, frames)
        self._total_steps += 1
        if not went_on and runner.state is not None:
            self._record(runner.finish())

    # ------------------------------------------------------------ helpers

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
            )
        self._busy = True
        try:
            self._runner.begin(restore=self._mode.ai_plays)
        except EmulatorError:
            return None
        finally:
            self._busy = False
        return self._runner

    def _prepare_checkpoint(self) -> None:
        """Every go in an AI mode restarts from where the child left the game."""
        if self._harness is None or not self._mode.ai_plays:
            return
        try:
            if self._harness.available():
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
        self._history.append(record)
        self._bus.publish(EpisodeFinished(record))

    def _announce(self, reason: str) -> None:
        self._bus.publish(TrainingStateChanged(self._status, self._mode, reason))


def _goes(n: int) -> str:
    return "go" if n == 1 else "goes"


__all__ = [
    "CHECKPOINT_SLOT",
    "EpisodeFinished",
    "EpisodeRecord",
    "Mode",
    "Speed",
    "Status",
    "TrainingService",
    "TrainingStateChanged",
]
