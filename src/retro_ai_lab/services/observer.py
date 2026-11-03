"""Observation service: snapshots memory every frame while connected.

Driven by the play clock (or any caller that advances frames), it feeds a
:class:`SnapshotRing` and a :class:`ChangeTracker`, and publishes
:class:`MemoryObserved` so the UI and the understanding engine can react
without touching the emulator themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_ai_lab.core import EmulatorDisconnectedError, EmulatorError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.observation.diff import ChangeTracker, SnapshotDiff
from retro_ai_lab.observation.memory import MemoryCapture, MemorySnapshot, SnapshotRing
from retro_ai_lab.services.connection import (
    ConnectionService,
    EmulatorConnected,
    EmulatorDisconnected,
)


@dataclass(frozen=True, slots=True)
class MemoryObserved:
    snapshot: MemorySnapshot
    diff: SnapshotDiff | None  # None for the first snapshot after connect


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

    def advance(self, frames: int = 1) -> MemoryObserved | None:
        """Record that ``frames`` frames passed, then snapshot if enabled."""
        if frames < 0:
            raise ValueError("frames must be >= 0")
        self._frame += frames
        return self.observe()

    def observe(self) -> MemoryObserved | None:
        """Take a snapshot now. Returns None if observation is off or failed."""
        if not self._enabled or self._capture is None or not self._connection.is_connected:
            return None
        try:
            snapshot = self._capture.capture(self._frame)
        except EmulatorDisconnectedError:
            # The connection service has already announced the loss.
            return None
        except EmulatorError:
            # A glitchy read is counted but must not stop the game.
            self._errors += 1
            return None
        self._ring.append(snapshot)
        diff = self._tracker.observe(snapshot)
        event = MemoryObserved(snapshot, diff)
        self._bus.publish(event)
        return event

    # ------------------------------------------------------------- events

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._ring.clear()
        self._tracker.reset()
        self._frame = 0
        self._errors = 0
        self._enabled = event.capabilities.memory_read
        self._capture = MemoryCapture(self._connection) if self._enabled else None

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._enabled = False
        self._capture = None


__all__ = ["MemoryObserved", "ObservationService"]
