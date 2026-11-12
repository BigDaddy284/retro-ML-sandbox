"""Save-state experiment harness: try things, then put the game back.

The understanding engine (Phase 3) learns what inputs do by running short,
repeatable *trials*: save the game, hold some buttons for a few frames,
look at what changed in memory and on screen, then restore the save so
the next trial starts from exactly the same place. Comparing trials that
differ only in their inputs isolates the bytes those inputs influence.

Callers that drive the emulator on a timer (the play clock) must pause it
around a trial; the harness is synchronous and never touches the UI.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from retro_ai_lab.core import (
    Button,
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
    Frame,
    UnsupportedCapabilityError,
)
from retro_ai_lab.emulator.base import EmulatorAdapter
from retro_ai_lab.observation.diff import SnapshotDiff, diff_snapshots
from retro_ai_lab.observation.memory import MemoryCapture, MemorySnapshot
from retro_ai_lab.observation.screen import FrameDiff, diff_frames
from retro_ai_lab.services.connection import ConnectionService

DEFAULT_SLOT = "experiment"
REQUIRED = ("save_states", "memory_read", "input", "frame_advance")

InputPlan = Sequence[Iterable[str]]
"""Button names to hold on each frame of a trial, one entry per frame."""


@dataclass(frozen=True, slots=True)
class Trial:
    label: str
    plan: tuple[frozenset[str], ...]
    before: MemorySnapshot
    after: MemorySnapshot
    diff: SnapshotDiff
    screen_before: Frame | None
    screen_after: Frame | None
    screen_diff: FrameDiff | None

    @property
    def frames(self) -> int:
        return len(self.plan)

    @property
    def changed(self) -> frozenset[tuple[str, int]]:
        return frozenset((c.region, c.offset) for c in self.diff.changes)


@dataclass(frozen=True, slots=True)
class Comparison:
    """Which bytes ended up different between trials that started alike."""

    labels: tuple[str, ...]
    discriminating: Mapping[
        tuple[str, int], Mapping[str, int]
    ]  # (region, offset) -> label -> value
    common: frozenset[tuple[str, int]]  # changed in every trial

    def values(self, region: str, offset: int) -> Mapping[str, int]:
        try:
            return self.discriminating[(region, offset)]
        except KeyError:
            raise ConfigurationError(f"{region}[{offset}] did not discriminate.") from None


def constant_plan(buttons: Iterable[str], frames: int) -> tuple[frozenset[str], ...]:
    """Hold the same buttons for ``frames`` frames."""
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ConfigurationError("frames must be a positive integer.")
    held = frozenset(buttons)
    return tuple(held for _ in range(frames))


class ExperimentHarness:
    def __init__(
        self,
        connection: ConnectionService,
        *,
        slot: str = DEFAULT_SLOT,
        on_restore: Callable[[], None] | None = None,
    ) -> None:
        if not slot or not isinstance(slot, str):
            raise ConfigurationError("slot must be a non-empty name.")
        if on_restore is not None and not callable(on_restore):
            raise ConfigurationError("on_restore must be callable.")
        self._connection = connection
        self._slot = slot
        self._on_restore = on_restore  # told after every load_state (T-935)
        self._capture = MemoryCapture(connection)
        self._has_checkpoint = False
        self._trials = 0

    # ------------------------------------------------------------ queries

    @property
    def slot(self) -> str:
        return self._slot

    @property
    def has_checkpoint(self) -> bool:
        return self._has_checkpoint

    @property
    def trials(self) -> int:
        """Trials run since the harness was created."""
        return self._trials

    def available(self) -> bool:
        """True if the connected emulator supports everything a trial needs.

        A link that drops between the check and the call makes it False (T-980):
        the user's log ended with that error from a memory handler that ran as the
        app shut down. The connection service records the disconnect itself.
        """
        if not self._connection.is_connected:
            return False
        try:
            caps = self._connection.call(lambda a: a.capabilities())
        except EmulatorError:
            return False
        return all(caps.supports(name) for name in REQUIRED)

    def _require(self) -> None:
        if not self._connection.is_connected:
            raise EmulatorDisconnectedError()
        caps = self._connection.call(lambda a: a.capabilities())
        for name in REQUIRED:
            if not caps.supports(name):
                raise UnsupportedCapabilityError(
                    name,
                    "This emulator can't run experiments: it has no "
                    + name.replace("_", " ")
                    + ".",
                )

    # ------------------------------------------------------------ driving

    def checkpoint(self) -> None:
        """Save the current game state as the starting point for trials."""
        self._require()
        self._connection.call(lambda a: a.save_state(self._slot))
        self._has_checkpoint = True

    def restore(self) -> None:
        """Put the game back to the checkpoint."""
        self._require()
        if not self._has_checkpoint:
            raise ConfigurationError("No checkpoint yet; call checkpoint() first.")
        self._load()

    def _load(self) -> None:
        self._connection.call(lambda a: a.load_state(self._slot))
        if self._on_restore is not None:
            self._on_restore()

    def run(self, plan: InputPlan, *, label: str = "", restore: bool = True) -> Trial:
        """Run one trial from the checkpoint and (by default) restore afterwards.

        ``plan`` gives the button names to hold on each frame. On link loss
        the error propagates after the harness forgets its checkpoint, since
        a fresh connection starts from a different place.
        """
        self._require()
        if not self._has_checkpoint:
            raise ConfigurationError("No checkpoint yet; call checkpoint() first.")
        steps = tuple(frozenset(step) for step in plan)
        if not steps:
            raise ConfigurationError("A trial needs at least one frame.")
        name = label or f"trial-{self._trials + 1}"
        try:
            return self._run(steps, name, restore)
        except EmulatorDisconnectedError:
            self._has_checkpoint = False
            raise

    def _run(self, steps: tuple[frozenset[str], ...], label: str, restore: bool) -> Trial:
        conn = self._connection
        self._load()
        can_see = conn.call(lambda a: a.capabilities().screen_capture)
        buttons = {b.name: b for b in conn.call(lambda a: a.buttons())}
        unknown = {n for step in steps for n in step} - buttons.keys()
        if unknown:
            raise ConfigurationError(f"Unknown buttons for this game: {sorted(unknown)}.")
        before = self._capture.capture(0)
        screen_before = conn.call(lambda a: a.capture_frame()) if can_see else None
        for step in steps:
            held = [buttons[n] for n in step]
            conn.call(_stepper(held))
        after = self._capture.capture(len(steps))
        screen_after = conn.call(lambda a: a.capture_frame()) if can_see else None
        conn.call(lambda a: a.set_input(()))
        if restore:
            self._load()
        self._trials += 1
        screen_diff = (
            diff_frames(screen_before, screen_after)
            if screen_before is not None and screen_after is not None
            else None
        )
        return Trial(
            label,
            steps,
            before,
            after,
            diff_snapshots(before, after),
            screen_before,
            screen_after,
            screen_diff,
        )

    def run_many(
        self, plans: Mapping[str, InputPlan], *, restore: bool = True
    ) -> tuple[Trial, ...]:
        """Run several labelled trials from the same checkpoint."""
        if not plans:
            raise ConfigurationError("run_many needs at least one plan.")
        return tuple(self.run(plan, label=label, restore=restore) for label, plan in plans.items())

    # ------------------------------------------------------------ analysis

    @staticmethod
    def compare(trials: Sequence[Trial]) -> Comparison:
        """Find bytes whose final value depends on which trial was run.

        All trials must start from the same ``before`` snapshot.
        """
        if len(trials) < 2:
            raise ConfigurationError("compare needs at least two trials.")
        first = trials[0]
        labels = tuple(t.label for t in trials)
        if len(set(labels)) != len(labels):
            raise ConfigurationError("Trial labels must be unique.")
        for t in trials[1:]:
            if t.before != first.before:
                raise ConfigurationError(
                    f"Trial {t.label!r} did not start from the same state as {first.label!r}."
                )
        discriminating: dict[tuple[str, int], dict[str, int]] = {}
        for name in first.after.region_names:
            columns = [t.after.regions[name] for t in trials]
            for offset in range(len(columns[0])):
                values = [c[offset] for c in columns]
                if len(set(values)) > 1:
                    discriminating[(name, offset)] = dict(zip(labels, values, strict=True))
        common = frozenset.intersection(*(t.changed for t in trials))
        return Comparison(labels, discriminating, common)


def _stepper(held: list[Button]) -> Callable[[EmulatorAdapter], None]:
    def step(adapter: EmulatorAdapter) -> None:
        adapter.set_input(held)
        adapter.advance_frames(1)

    return step


__all__ = [
    "DEFAULT_SLOT",
    "REQUIRED",
    "Comparison",
    "ExperimentHarness",
    "InputPlan",
    "Trial",
    "constant_plan",
]
