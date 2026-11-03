"""One episode of play-and-learn (DESIGN.md Phase 5).

Each step the agent looks at the game (a :class:`State` from the believed
variables), picks an action, presses it as the AI, lets a few frames pass,
and learns from the points the reward and goal services announced in the
meantime. An episode ends when the game starts over (a reset event), after
``max_steps``, or if the emulator link drops. With save states, every
episode can start from the same checkpoint (DESIGN.md §9).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from retro_ai_lab.core import Button, ConfigurationError, EmulatorError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.learning.learner import Learner
from retro_ai_lab.learning.replay import Transition
from retro_ai_lab.learning.state import State, StateEncoder
from retro_ai_lab.services.connection import ConnectionService
from retro_ai_lab.services.discovery import DiscoveryUpdated
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.services.goals import ObjectiveCompleted, ObjectiveSetback
from retro_ai_lab.services.observer import ObservationService
from retro_ai_lab.services.rewards import RewardEarned
from retro_ai_lab.understanding.events import EventKind

DEFAULT_FRAMES_PER_STEP = 4
DEFAULT_MAX_STEPS = 600
DEFAULT_UNSTICK_AFTER = 25  # steps with nothing changing before Start is pressed
DEFAULT_UNSTICK_BUTTONS: tuple[str, ...] = ("start",)
DEFAULT_SETTLE_FRAMES = 5  # the understanding engine's event window


class EpisodeSignals:
    """Adds up the points announced on the bus and notices game restarts."""

    def __init__(self, bus: EventBus) -> None:
        self._reward = 0.0
        self._resets = 0
        self._completed = 0
        self._listening = False
        bus.subscribe(RewardEarned, self._on_reward)
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

    def start(self) -> None:
        self._reward = 0.0
        self._resets = 0
        self._completed = 0
        self._listening = True

    def stop(self) -> None:
        self._listening = False

    def take(self) -> tuple[float, int]:
        """(reward, resets) since the last take; clears both."""
        out = (self._reward, self._resets)
        self._reward = 0.0
        self._resets = 0
        return out

    def _on_reward(self, event: RewardEarned) -> None:
        if self._listening:
            self._reward += event.amount

    def _on_completed(self, event: ObjectiveCompleted) -> None:
        if self._listening:
            self._reward += event.reward
            self._completed += 1

    def _on_setback(self, event: ObjectiveSetback) -> None:
        if self._listening:
            self._reward += event.penalty

    def _on_discovery(self, update: DiscoveryUpdated) -> None:
        if self._listening:
            self._resets += sum(1 for e in update.events if e.kind is EventKind.RESET)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    steps: int
    frames: int
    reward: float
    ended_by: str  # "reset", "max_steps", "disconnected", "stopped", "mode change"
    states_seen: int
    objectives: int = 0  # objectives completed during the go

    @property
    def succeeded(self) -> bool:
        return self.objectives > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "frames": self.frames,
            "reward": self.reward,
            "ended_by": self.ended_by,
            "states_seen": self.states_seen,
            "objectives": self.objectives,
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
    ) -> None:
        if frames_per_step < 1:
            raise ConfigurationError("frames_per_step must be at least 1.")
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
        self._restored = False
        self._unsticks = 0
        self._same_state = 0
        self._last_buttons: frozenset[str] = frozenset()
        self._last_action: int | None = None
        self._last_state: State | None = None
        self._last_reward = 0.0
        self._signals = EpisodeSignals(bus)
        self._buttons: dict[str, Button] = {}
        self._state: State | None = None
        self._steps = 0
        self._frames = 0
        self._reward = 0.0
        self._states: set[str] = set()
        self._ended_by: str | None = None

    # ------------------------------------------------------------ queries

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
        self._buttons = {b.name: b for b in self._connection.call(lambda a: a.buttons())}
        self._restored = restore and self._harness is not None and self._harness.has_checkpoint
        if self._restored:
            assert self._harness is not None
            self._harness.restore()
        self._encoder.refresh()
        # Observe before listening: putting the game back to the checkpoint looks like
        # a restart to the understanding engine, and that must not end the new episode.
        latest = self._observation.observe()
        snapshot = latest.snapshot if latest is not None else self._observation.latest
        if snapshot is None:
            raise ConfigurationError(
                "Can't see the game's memory, so there is nothing to learn from."
            )
        self._signals.start()
        self._state = self._encoder.encode(snapshot)
        self._states.add(self._state.key)
        return self._state

    def step(self) -> bool:
        """One act-advance-learn step. Returns False once the episode is over."""
        if self._state is None:
            raise ConfigurationError("Call begin() before step().")
        if self._ended_by is not None:
            return False
        state = self._state
        # On a screen where nothing changes (game over, a menu) the learner's menu has
        # no way out, so press the buttons it is normally kept away from, once.
        stuck = (
            self._unstick_after is not None
            and self._same_state >= self._unstick_after
            and any(n in self._buttons for n in self._unstick_buttons)
        )
        if stuck:
            action = None
            names = frozenset(n for n in self._unstick_buttons if n in self._buttons)
            self._unsticks += 1
            self._same_state = 0
        else:
            action = self._agent.choose(state)
            names = self._agent.space.buttons_for(action)
        held = [self._buttons[n] for n in names]
        self._last_buttons = frozenset(names)
        self._last_action = action
        try:
            self._connection.call(lambda a: a.set_input(held))
            self._observation.record_input(names, "ai")
            self._connection.call(lambda a: a.advance_frames(self._frames_per_step))
            self._observation.advance(self._frames_per_step)
        except EmulatorError:
            self._ended_by = "disconnected"
            self._signals.stop()
            return False
        return self._settle(state, action, self._frames_per_step, "ai")

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
        return self._settle(self._state, action, frames, "child")

    def _settle(self, state: State, action: int | None, frames: int, source: str) -> bool:
        """After the frames passed: collect the reward, read the next state, learn."""
        self._frames += frames
        reward, resets = self._signals.take()
        self._last_state = state
        self._last_reward = reward
        # Putting the game back to the checkpoint looks like a restart to the engine, and
        # it may only notice a few frames later: a "reset" that soon after a restore is
        # the restore itself, not the game starting over.
        if self._restored and self._frames <= self._settle_frames:
            resets = 0
        snapshot = self._observation.latest
        next_state = self._encoder.encode(snapshot) if snapshot is not None else state
        done = resets > 0
        if action is not None:
            self._agent.learn(state, action, reward, next_state, done)
            if self._on_learned is not None:
                self._on_learned(Transition(state, action, reward, next_state, done, source))
        self._reward += reward
        self._steps += 1
        self._same_state = self._same_state + 1 if next_state == state else 0
        self._state = next_state
        self._states.add(next_state.key)
        if done:
            self._ended_by = "reset"
        elif self._steps >= self._max_steps:
            self._ended_by = "max_steps"
        if self._ended_by is not None:
            self._signals.stop()
            return False
        return True

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
        self._agent.end_episode()
        self._last_buttons = frozenset()
        result = EpisodeResult(
            self._steps,
            self._frames,
            self._reward,
            self._ended_by,
            len(self._states),
            self._signals.completed,
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
    "EpisodeResult",
    "EpisodeRunner",
    "EpisodeSignals",
]
