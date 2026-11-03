"""Research mode (DESIGN.md §16 Phase 8): the AI's training laid bare.

Three things an older child, a teacher or a researcher wants that the panels keep
out of sight: every step the AI took (as a log they can take away as CSV), what the
AI thinks each move is worth in the situation in front of it, and the learning
settings themselves. The dialog that shows them is built on this module.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.learning.learner import Learner
from retro_ai_lab.learning.state import State
from retro_ai_lab.training.service import TrainingService, TrainingStateChanged, TrainingStepped
from retro_ai_lab.understanding import Knowledge

DEFAULT_LIMIT = 10_000
COLUMNS = ("go", "step", "frame", "situation", "buttons", "explored", "value", "reward")
CLEARING_REASONS = frozenset({"forget", "loaded"})


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One step of training as the log keeps it."""

    go: int  # the go's number over the life of the brain
    step: int  # 1-based within the go
    frame: int
    situation: str  # the state key the move was chosen in
    buttons: str  # what was held, "+"-joined and sorted ("" for nothing)
    explored: bool | None  # None when the AI didn't choose (the child played, or an unstick)
    value: float | None  # what the AI thought the move was worth (None as above)
    reward: float

    @classmethod
    def from_event(cls, go: int, event: TrainingStepped) -> StepRecord:
        decision = event.decision
        return cls(
            go,
            event.step,
            event.frame,
            event.situation,
            "+".join(sorted(event.buttons)),
            None if decision is None else decision.explored,
            None if decision is None else decision.value,
            event.reward,
        )

    def row(self) -> tuple[str, ...]:
        """The record as CSV cells, in :data:`COLUMNS` order."""
        return (
            str(self.go),
            str(self.step),
            str(self.frame),
            self.situation,
            self.buttons,
            "" if self.explored is None else ("yes" if self.explored else "no"),
            "" if self.value is None else repr(self.value),
            repr(self.reward),
        )

    @classmethod
    def from_row(cls, cells: tuple[str, ...] | list[str]) -> StepRecord:
        if len(cells) != len(COLUMNS):
            raise ConfigurationError(f"A step row needs {len(COLUMNS)} cells.")
        go, step, frame, situation, buttons, explored, value, reward = cells
        try:
            return cls(
                int(go),
                int(step),
                int(frame),
                situation,
                buttons,
                None if explored == "" else explored == "yes",
                None if value == "" else float(value),
                float(reward),
            )
        except ValueError as exc:
            raise ConfigurationError(f"Bad step row: {exc}") from exc


class StepLog:
    """The newest ``limit`` training steps, filled from the training events.

    The go number is the one in progress when the step happened. A brain started
    from scratch or a loaded one restarts the go count, so the log empties then.
    """

    def __init__(self, training: TrainingService, bus: EventBus, *, limit: int = DEFAULT_LIMIT):
        if limit < 1:
            raise ConfigurationError("The step log must keep at least one step.")
        self._training = training
        self._limit = limit
        self._records: list[StepRecord] = []
        self._dropped = 0
        bus.subscribe(TrainingStepped, self._on_stepped)
        bus.subscribe(TrainingStateChanged, self._on_state)

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def dropped(self) -> int:
        """Steps that fell off the front because the log was full."""
        return self._dropped

    @property
    def records(self) -> tuple[StepRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[StepRecord]:
        return iter(tuple(self._records))

    def clear(self) -> None:
        self._records.clear()
        self._dropped = 0

    def add(self, record: StepRecord) -> None:
        self._records.append(record)
        if len(self._records) > self._limit:
            del self._records[0]
            self._dropped += 1

    def write_csv(self, path: Path) -> int:
        """Write the log to ``path`` (header first). Returns the number of rows written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for record in self._records:
                writer.writerow(record.row())
        return len(self._records)

    @staticmethod
    def read_csv(path: Path) -> tuple[StepRecord, ...]:
        """Read a log written by :meth:`write_csv` back (for tests and tools)."""
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None or tuple(header) != COLUMNS:
                raise ConfigurationError("That file is not a step log.")
            return tuple(StepRecord.from_row(cells) for cells in reader if cells)

    # ------------------------------------------------------------ events

    def _on_stepped(self, event: TrainingStepped) -> None:
        self.add(StepRecord.from_event(self._training.episodes + 1, event))

    def _on_state(self, event: TrainingStateChanged) -> None:
        if event.reason in CLEARING_REASONS:
            self.clear()


@dataclass(frozen=True, slots=True)
class ActionValue:
    label: str  # the move, in the action space's words
    value: float
    best: bool  # True for the move(s) the AI would play


def situation_values(learner: Learner, state: State) -> tuple[ActionValue, ...]:
    """What the learner thinks every move is worth in ``state``, best marked."""
    values = tuple(float(v) for v in learner.values(state))
    labels = learner.space.labels
    if len(values) != len(labels):
        raise ConfigurationError("The learner's values don't match its controls.")
    top = max(values) if values else 0.0
    return tuple(
        ActionValue(label, value, value == top) for label, value in zip(labels, values, strict=True)
    )


@dataclass(frozen=True, slots=True)
class SituationReport:
    """The situation in front of the AI and its opinion of every move."""

    state: State
    values: tuple[ActionValue, ...]

    @classmethod
    def of(cls, learner: Learner, state: State) -> SituationReport:
        return cls(state, situation_values(learner, state))

    def best(self) -> tuple[str, ...]:
        return tuple(v.label for v in self.values if v.best)

    def describe(self, knowledge: Knowledge | None = None) -> str:
        where = self.state.describe(knowledge)
        if not self.values:
            return f"In the situation {where} the AI has no moves."
        parts = ", ".join(
            f"{v.label} {v.value:+.2f}{' (best)' if v.best else ''}" for v in self.values
        )
        return f"In the situation {where} the AI rates its moves: {parts}."
