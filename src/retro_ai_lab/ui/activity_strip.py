"""Three lines on what the AI is doing right now (DESIGN.md §10 "AI Activity").

* **Now**: the buttons being pressed and whether that was a random try or
  the best move it knows (and what it thought that move was worth).
* **Before**: the last few moves, newest first.
* **Working on**: the goal step the AI is going for.

Steps arrive as :class:`TrainingStepped` events; in fast training there can
be hundreds a second, so the labels are redrawn once per event-loop turn.
"""

from __future__ import annotations

from collections import deque

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from retro_ai_lab.learning.actions import NOTHING, Action
from retro_ai_lab.learning.qlearning import Decision
from retro_ai_lab.services import (
    EmulatorConnected,
    EmulatorDisconnected,
    GoalChanged,
    ObjectiveCompleted,
)
from retro_ai_lab.training import (
    EpisodeFinished,
    Status,
    TrainingStateChanged,
    TrainingStepped,
)
from retro_ai_lab.ui.context import AppContext

RESTING = "Resting."
PAUSED = "Paused."
LOOKING = "Now: looking at the game."
NO_MOVES = "Before: nothing yet."
NO_GOAL = "No goal step to work on."
GOAL_DONE = "Working on: nothing left, the goal is complete!"
REMEMBER = 8


def buttons_text(buttons: frozenset[str]) -> str:
    return Action(0, buttons).label if buttons else NOTHING


def now_line(
    status: Status, ai_plays: bool, buttons: frozenset[str] | None, decision: Decision | None
) -> str:
    """The "Now" line for a state of play; ``buttons`` is None before any move."""
    if status is Status.IDLE:
        return RESTING
    if status is Status.PAUSED:
        return PAUSED
    if buttons is None:
        return LOOKING
    pressing = buttons_text(buttons)
    if not ai_plays:
        return f"Watching you press {pressing}."
    if decision is None:
        return f"Now: pressing {pressing} to get the game going again."
    if decision.explored:
        return f"Now: pressing {pressing}, trying something new."
    return f"Now: pressing {pressing}, its best known move (worth {decision.value:+.1f})."


class ActivityStrip(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("activity_strip")
        self._context = context
        self._moves: deque[frozenset[str]] = deque(maxlen=REMEMBER)
        self._decision: Decision | None = None
        self._pending = False
        self._now = QLabel(RESTING, self)
        self._now.setObjectName("activity_now")
        self._now.setWordWrap(True)
        self._before = QLabel(NO_MOVES, self)
        self._before.setObjectName("activity_before")
        self._before.setWordWrap(True)
        self._goal = QLabel(NO_GOAL, self)
        self._goal.setObjectName("activity_goal")
        self._goal.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._now)
        layout.addWidget(self._before)
        layout.addWidget(self._goal)
        bridge = context.bridge
        unsubscribe = (
            bridge.subscribe(TrainingStepped, self._on_step),
            bridge.subscribe(TrainingStateChanged, self._on_change),
            bridge.subscribe(EpisodeFinished, self._on_change),
            bridge.subscribe(GoalChanged, self._on_change),
            bridge.subscribe(ObjectiveCompleted, self._on_change),
            bridge.subscribe(EmulatorConnected, self._on_connection),
            bridge.subscribe(EmulatorDisconnected, self._on_connection),
        )
        self.destroyed.connect(lambda *_: [off() for off in unsubscribe])
        self.refresh()

    # --------------------------------------------------------------- API

    @property
    def now_text(self) -> str:
        return self._now.text()

    @property
    def before_text(self) -> str:
        return self._before.text()

    @property
    def goal_text(self) -> str:
        return self._goal.text()

    @property
    def moves(self) -> tuple[frozenset[str], ...]:
        """The remembered moves, newest first."""
        return tuple(reversed(self._moves))

    def refresh(self) -> None:
        self._pending = False
        training = self._context.training
        status = training.status if training.enabled else Status.IDLE
        latest = self._moves[-1] if self._moves else None
        self._now.setText(now_line(status, training.mode.ai_plays, latest, self._decision))
        self._before.setText(self._before_line())
        self._goal.setText(self._goal_line())

    # ----------------------------------------------------------- internals

    def _before_line(self) -> str:
        if len(self._moves) < 2:
            return NO_MOVES
        earlier = list(self._moves)[:-1]
        return "Before: " + ", ".join(buttons_text(m) for m in reversed(earlier)) + "."

    def _goal_line(self) -> str:
        goals = self._context.goals
        goal = goals.goal
        if not goals.enabled or goal is None or not len(goal):
            return NO_GOAL
        if goal.is_complete:
            return GOAL_DONE
        nxt = goal.next()
        return f"Working on: {nxt.description}." if nxt is not None else NO_GOAL

    def _schedule(self) -> None:
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self.refresh)

    def _on_step(self, event: TrainingStepped) -> None:
        self._moves.append(event.buttons)
        self._decision = event.decision
        self._schedule()

    def _on_change(self, event: object) -> None:
        self.refresh()

    def _on_connection(self, event: object) -> None:
        self._moves.clear()
        self._decision = None
        self.refresh()


__all__ = [
    "GOAL_DONE",
    "LOOKING",
    "NO_GOAL",
    "NO_MOVES",
    "PAUSED",
    "REMEMBER",
    "RESTING",
    "ActivityStrip",
    "buttons_text",
    "now_line",
]
