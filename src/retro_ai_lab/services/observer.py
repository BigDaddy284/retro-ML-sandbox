"""Observation service: snapshots memory every frame while connected.

Driven by the play clock (or any caller that advances frames), it feeds a
:class:`SnapshotRing` and a :class:`ChangeTracker`, and publishes
:class:`MemoryObserved` so the UI and the understanding engine can react
without touching the emulator themselves.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError, EmulatorDisconnectedError, EmulatorError, Frame
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.observation.diff import ChangeTracker, SnapshotDiff
from retro_ai_lab.observation.inputs import InputEvent, InputLog
from retro_ai_lab.observation.memory import MemoryCapture, MemorySnapshot, SnapshotRing
from retro_ai_lab.observation.screen import FrameDiff, ScreenTracker
from retro_ai_lab.services.connection import (
    ConnectionService,
    EmulatorConnected,
    EmulatorDisconnected,
)


@dataclass(frozen=True, slots=True)
class MemoryObserved:
    snapshot: MemorySnapshot
    diff: SnapshotDiff | None  # None for the first snapshot after connect


@dataclass(frozen=True, slots=True)
class ScreenObserved:
    frame: Frame
    diff: FrameDiff | None  # None for the first frame after connect


@dataclass(frozen=True, slots=True)
class InputRecorded:
    event: InputEvent


@dataclass(frozen=True, slots=True)
class ButtonsHeld:
    """What a go's step held, every step (T-874), unlike ``InputRecorded`` (changes only).

    Published by the episode runner before the step's reward is collected, so a
    reward on a button lands in the step that pressed it.
    """

    frame: int
    buttons: frozenset[str]
    source: str  # "ai" or "child"


class ObservationService:
    def __init__(
        self,
        connection: ConnectionService,
        bus: EventBus,
        *,
        ring_capacity: int = 600,
    ) -> None:
        self._connection = connection
        self._bus = bus
        self._ring = SnapshotRing(ring_capacity)
        self._tracker = ChangeTracker()
        self._screen = ScreenTracker()
        self._inputs = InputLog()
        self._frames_seen = 0
        self._capture: MemoryCapture | None = None
        self._enabled = False
        self._frame = 0
        self._errors = 0
        bus.subscribe(EmulatorConnected, self._on_connected)
        bus.subscribe(EmulatorDisconnected, self._on_disconnected)

    # ------------------------------------------------------------ queries

    @property
    def enabled(self) -> bool:
        """True while a connected emulator lets us read memory."""
        return self._enabled

    @property
    def ring(self) -> SnapshotRing:
        return self._ring

    @property
    def tracker(self) -> ChangeTracker:
        return self._tracker

    @property
    def screen(self) -> ScreenTracker:
        return self._screen

    @property
    def inputs(self) -> InputLog:
        return self._inputs

    @property
    def frames_seen(self) -> int:
        """Screen frames fed through :meth:`observe_frame` since connect."""
        return self._frames_seen

    @property
    def frame(self) -> int:
        """Frames counted since connect (what snapshots are tagged with)."""
        return self._frame

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def latest(self) -> MemorySnapshot | None:
        return self._ring.latest

    # ------------------------------------------------------------ driving

    def snapshot_reads(self) -> tuple[tuple[str, int, int], ...]:
        """The memory reads one snapshot needs, to bundle into a ``step`` (T-823).

        Empty when observation is off or the plan cannot be made, so the
        caller sends a plain step and :meth:`advance` does the usual read.
        """
        if not self._enabled or self._capture is None or not self._connection.is_connected:
            return ()
        try:
            return self._capture.reads()
        except (EmulatorError, ConfigurationError):
            return ()

    def advance(
        self, frames: int = 1, *, snapshot: Sequence[bytes] | None = None
    ) -> MemoryObserved | None:
        """Record that ``frames`` frames passed, then snapshot if enabled.

        ``snapshot`` holds the bytes a bundled ``step`` already fetched, in
        :meth:`snapshot_reads` order; without it the service reads them now.
        """
        if frames < 0:
            raise ValueError("frames must be >= 0")
        self._frame += frames
        return self.observe(snapshot=snapshot)

    def observe(self, *, snapshot: Sequence[bytes] | None = None) -> MemoryObserved | None:
        """Take a snapshot now. Returns None if observation is off or failed."""
        if not self._enabled or self._capture is None or not self._connection.is_connected:
            return None
        try:
            if snapshot is not None:
                captured = self._capture.from_reads(self._frame, snapshot)
            else:
                captured = self._capture.capture(self._frame)
        except EmulatorDisconnectedError:
            # The connection service has already announced the loss.
            return None
        except EmulatorError:
            # A glitchy read is counted but must not stop the game.
            self._errors += 1
            return None
        self._ring.append(captured)
        diff = self._tracker.observe(captured)
        event = MemoryObserved(captured, diff)
        self._bus.publish(event)
        return event

    def observe_frame(self, frame: Frame) -> ScreenObserved | None:
        """Feed a captured screen frame (the game panel's pump calls this)."""
        if not self._connection.is_connected:
            return None
        self._frames_seen += 1
        diff = self._screen.observe(frame)
        event = ScreenObserved(frame, diff)
        self._bus.publish(event)
        return event

    def record_input(self, buttons: Iterable[str], source: str = "human") -> InputRecorded | None:
        """Record the held button set as of the current frame."""
        if not self._connection.is_connected:
            return None
        event = self._inputs.record(self._frame, buttons, source)
        if event is None:
            return None
        recorded = InputRecorded(event)
        self._bus.publish(recorded)
        return recorded

    # ------------------------------------------------------------- events

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._ring.clear()
        self._tracker.reset()
        self._screen.reset()
        self._inputs.clear()
        self._frames_seen = 0
        self._frame = 0
        self._errors = 0
        self._enabled = event.capabilities.memory_read
        self._capture = MemoryCapture(self._connection) if self._enabled else None

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._enabled = False
        self._capture = None


__all__ = ["InputRecorded", "MemoryObserved", "ObservationService", "ScreenObserved"]
