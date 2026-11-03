"""The Goals panel (in the Progress slot): the big goal and its steps.

A child names the goal, sees each step with how the AI will know it is
done, how sure the AI is about that, and whether the step is done, ready,
waiting for another step, or switched off. They can add steps (detected by
something the AI already knows, or ticked by hand), remove and reorder
them, say what a step must come after, tick steps off and start again
(DESIGN.md §4, §10). Every change goes through :class:`GoalService`, so it
is saved and announced.
"""

from __future__ import annotations

from PySide6.QtCore import QModelIndex, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from retro_ai_lab.core import ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.rewards import Goal, Objective, Trigger
from retro_ai_lab.services import (
    EmulatorConnected,
    EmulatorDisconnected,
    GoalChanged,
    ObjectiveCompleted,
    ObjectiveSetback,
)
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.discovery_panel import BADGES

IDLE_TEXT = "Connect to a game to set a goal for the AI."
BY_HAND = "I'll tick it myself"
NOTHING = "nothing"
COLUMNS = ("Step", "Reward", "Detected by", "Status", "On")


def status_text(goal: Goal, objective: Objective) -> str:
    if not objective.enabled:
        return "Off"
    if objective.completed:
        return "Done"
    if goal.is_ready(objective.id):
        return "Ready"
    waiting = [
        goal.get(d).description
        for d in objective.depends_on
        if d in goal and not goal.get(d).completed
    ]
    return "Waiting for: " + ", ".join(waiting) if waiting else "Ready"


def detection_text(objective: Objective, options: dict[str, str]) -> str:
    if objective.trigger is None:
        return BY_HAND
    label = options.get(objective.trigger.key, objective.trigger.describe())
    return f"{label} ({BADGES[objective.confidence]})"


class GoalsPanel(QWidget):
    selection_changed = Signal(object)  # objective id or None

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("goals_panel")
        self._context = context
        self._selected: str | None = None
        self._items: dict[str, QTreeWidgetItem] = {}
        self._options: dict[str, str] = {}
        self._in_item_signal = False
        self._refresh_pending = False

        self._title = QLineEdit(self)
        self._title.setObjectName("goal_title")
        self._title.setPlaceholderText("What should the AI try to do?")
        self._title.returnPressed.connect(self._on_title_entered)
        self._summary = QLabel(IDLE_TEXT, self)
        self._summary.setObjectName("goals_summary")
        self._summary.setWordWrap(True)

        self._tree = QTreeWidget(self)
        self._tree.setObjectName("goals_tree")
        self._tree.setColumnCount(len(COLUMNS))
        self._tree.setHeaderLabels(list(COLUMNS))
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.setRootIsDecorated(False)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        self._tree.itemChanged.connect(self._on_item_changed)

        self._new_step = QLineEdit(self)
        self._new_step.setObjectName("new_step_edit")
        self._new_step.setPlaceholderText("A new step, in your words")
        self._trigger = QComboBox(self)
        self._trigger.setObjectName("trigger_combo")
        self._add = QPushButton("Add a step", self)
        self._add.setObjectName("add_button")
        self._add.clicked.connect(self._on_add_clicked)
        self._new_step.returnPressed.connect(self._on_add_clicked)

        self._remove = QPushButton("Remove", self)
        self._remove.setObjectName("remove_button")
        self._up = QPushButton("Move up", self)
        self._up.setObjectName("up_button")
        self._down = QPushButton("Move down", self)
        self._down.setObjectName("down_button")
        self._done = QPushButton("Done it!", self)
        self._done.setObjectName("done_button")
        self._remove.clicked.connect(self.remove_selected)
        self._up.clicked.connect(lambda: self.move_selected(-1))
        self._down.clicked.connect(lambda: self.move_selected(1))
        self._done.clicked.connect(self.toggle_done_selected)
        self._after = QComboBox(self)
        self._after.setObjectName("after_combo")
        self._after.activated.connect(self._on_after_activated)
        self._reset = QPushButton("Start again", self)
        self._reset.setObjectName("reset_button")
        self._reset.clicked.connect(self.reset_progress)
        self._note = QLabel("", self)
        self._note.setObjectName("goals_note")
        self._note.setWordWrap(True)

        add_row = QHBoxLayout()
        add_row.addWidget(self._new_step, 2)
        add_row.addWidget(self._trigger, 1)
        add_row.addWidget(self._add)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self._done)
        edit_row.addWidget(self._up)
        edit_row.addWidget(self._down)
        edit_row.addWidget(self._remove)
        after_row = QHBoxLayout()
        after_row.addWidget(QLabel("Must come after", self))
        after_row.addWidget(self._after, 1)
        after_row.addWidget(self._reset)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._summary)
        layout.addWidget(self._tree, 1)
        layout.addLayout(edit_row)
        layout.addLayout(after_row)
        layout.addLayout(add_row)
        layout.addWidget(self._note)

        bridge = context.bridge
        self._unsubscribe = (
            bridge.subscribe(GoalChanged, self._on_goal_changed),
            bridge.subscribe(ObjectiveCompleted, self._on_completed),
            bridge.subscribe(ObjectiveSetback, self._on_setback),
            bridge.subscribe(EmulatorConnected, self._on_connected),
            bridge.subscribe(EmulatorDisconnected, self._on_disconnected),
        )
        unsubscribe = self._unsubscribe
        self.destroyed.connect(lambda *_: [u() for u in unsubscribe])
        self.refresh()

    # --------------------------------------------------------------- API

    @property
    def goal(self) -> Goal | None:
        return self._context.goals.goal

    @property
    def selected_id(self) -> str | None:
        return self._selected

    @property
    def title_text(self) -> str:
        return self._title.text()

    @property
    def summary_text(self) -> str:
        return self._summary.text()

    @property
    def note_text(self) -> str:
        return self._note.text()

    def objective_ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def row_text(self, objective_id: str) -> tuple[str, ...]:
        item = self._items[objective_id]
        on = "on" if item.checkState(4) == Qt.CheckState.Checked else "off"
        return (item.text(0), item.text(1), item.text(2), item.text(3), on)

    def trigger_keys(self) -> tuple[str | None, ...]:
        return tuple(self._trigger.itemData(i) for i in range(self._trigger.count()))

    def after_choices(self) -> tuple[str | None, ...]:
        return tuple(self._after.itemData(i) for i in range(self._after.count()))

    def select(self, objective_id: str | None) -> None:
        if objective_id is None:
            self._tree.clearSelection()
            self._tree.setCurrentIndex(QModelIndex())
            return
        self._tree.setCurrentItem(self._items[objective_id])

    def set_title(self, title: str) -> None:
        self._try(lambda: self._context.goals.set_title(title), "Goal renamed.")

    def add_step(self, description: str, trigger_key: str | None) -> None:
        trigger = self._trigger_for(trigger_key)
        if trigger_key is not None and trigger is None:
            self._note.setText("The AI doesn't know how to spot that yet.")
            return
        if self._try(
            lambda: self._context.goals.add_objective(description, trigger), "Step added."
        ):
            self._new_step.clear()

    def remove_selected(self) -> None:
        self._with_selected(lambda oid: self._context.goals.remove_objective(oid), "Step removed.")

    def move_selected(self, delta: int) -> None:
        goal = self.goal
        oid = self._selected
        if goal is None or oid is None or oid not in goal:
            return
        position = goal.ids().index(oid) + delta
        if not 0 <= position < len(goal):
            return
        self._with_selected(
            lambda o: self._context.goals.move_objective(o, position), "Step moved."
        )

    def toggle_done_selected(self) -> None:
        goal = self.goal
        oid = self._selected
        if goal is None or oid is None or oid not in goal:
            return
        if goal.get(oid).completed:
            self._with_selected(lambda o: self._context.goals.mark_not_complete(o), "Not done.")
        else:
            self._with_selected(lambda o: self._context.goals.mark_complete(o), "Done!")

    def set_selected_after(self, after_id: str | None) -> None:
        deps = (after_id,) if after_id else ()
        self._with_selected(
            lambda o: self._context.goals.set_dependencies(o, deps), "Order remembered."
        )

    def set_selected_enabled(self, enabled: bool) -> None:
        self._with_selected(
            lambda o: self._context.goals.update_objective(o, enabled=enabled),
            "Switched on." if enabled else "Switched off.",
        )

    def reset_progress(self) -> None:
        self._try(lambda: self._context.goals.reset_progress(), "Starting again from the top.")

    def refresh(self, *, keep: str | None = None) -> None:
        goal = self.goal
        enabled = self._context.goals.enabled
        self._fill_options()
        self._tree.blockSignals(True)
        self._tree.clear()
        self._items.clear()
        if goal is not None:
            for o in goal:
                self._items[o.id] = self._add_row(goal, o, enabled)
        self._tree.blockSignals(False)
        self._title.blockSignals(True)
        self._title.setText(goal.title if goal is not None else "")
        self._title.blockSignals(False)
        self._summary.setText(
            self._context.goals.summary() if enabled and goal is not None else IDLE_TEXT
        )
        for w in (self._tree, self._title, self._new_step, self._trigger, self._add, self._reset):
            w.setEnabled(enabled and goal is not None)
        keep = keep if keep is not None else self._selected
        if keep is not None and keep in self._items:
            self.select(keep)
            self._selected = keep
        else:
            self._selected = None
        self._show_selection()

    # ----------------------------------------------------------- internals

    def _fill_options(self) -> None:
        self._options = {o.key: o.label for o in self._context.goals.trigger_options()}
        current = self._trigger.currentData()
        self._trigger.blockSignals(True)
        self._trigger.clear()
        self._trigger.addItem(BY_HAND, None)
        for key, label in self._options.items():
            self._trigger.addItem(label, key)
        index = self._trigger.findData(current) if current else 0
        self._trigger.setCurrentIndex(max(index, 0))
        self._trigger.blockSignals(False)

    def _trigger_for(self, key: str | None) -> Trigger | None:
        if key is None:
            return None
        for option in self._context.goals.trigger_options():
            if option.key == key:
                return option.trigger
        return None

    def _add_row(self, goal: Goal, o: Objective, enabled: bool) -> QTreeWidgetItem:
        row = QTreeWidgetItem(
            [
                o.description,
                f"{o.reward:+g}" + (f" / {o.penalty:+g}" if o.penalty else ""),
                detection_text(o, self._options),
                status_text(goal, o),
                "",
            ]
        )
        row.setData(0, Qt.ItemDataRole.UserRole, o.id)
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
        if enabled:
            flags |= Qt.ItemFlag.ItemIsEnabled
        row.setFlags(flags)
        row.setCheckState(4, Qt.CheckState.Checked if o.enabled else Qt.CheckState.Unchecked)
        row.setToolTip(0, o.notes)
        if o.completed:
            row.setForeground(0, Qt.GlobalColor.darkGreen)
        elif not o.enabled:
            row.setForeground(0, Qt.GlobalColor.gray)
        self._tree.addTopLevelItem(row)
        return row

    def _show_selection(self) -> None:
        goal = self.goal
        oid = self._selected
        have = goal is not None and oid is not None and oid in goal and self._context.goals.enabled
        for w in (self._remove, self._up, self._down, self._done, self._after):
            w.setEnabled(have)
        self._after.blockSignals(True)
        self._after.clear()
        self._after.addItem(NOTHING, None)
        if have:
            assert goal is not None and oid is not None
            current = goal.get(oid)
            self._done.setText("Not done" if current.completed else "Done it!")
            for other in goal:
                if other.id != oid:
                    self._after.addItem(other.description, other.id)
            if current.depends_on:
                index = self._after.findData(current.depends_on[0])
                self._after.setCurrentIndex(max(index, 0))
        else:
            self._done.setText("Done it!")
        self._after.blockSignals(False)

    def _try(self, action: object, success: str) -> bool:
        try:
            action()  # type: ignore[operator]
        except (UnsupportedCapabilityError, ConfigurationError) as exc:
            self._note.setText(str(exc))
            return False
        self._note.setText(success)
        return True

    def _with_selected(self, action: object, success: str) -> None:
        oid = self._selected
        if oid is None:
            return
        self._try(lambda: action(oid), success)  # type: ignore[operator]

    def _on_title_entered(self) -> None:
        self.set_title(self._title.text())

    def _on_add_clicked(self) -> None:
        self.add_step(self._new_step.text(), self._trigger.currentData())

    def _on_after_activated(self, index: int) -> None:
        self.set_selected_after(self._after.itemData(index))

    def _on_current_changed(self, current: QTreeWidgetItem | None, _previous: object) -> None:
        oid = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        self._selected = str(oid) if oid is not None else None
        self._show_selection()
        self.selection_changed.emit(self._selected)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 4:
            return
        oid = item.data(0, Qt.ItemDataRole.UserRole)
        if oid is None:
            return
        on = item.checkState(4) == Qt.CheckState.Checked
        # Qt is still inside this item's signal: rebuilding the tree now would delete
        # the item under its feet (a crash on Windows), so any refresh waits a tick.
        self._in_item_signal = True
        try:
            self._try(
                lambda: self._context.goals.update_objective(str(oid), enabled=on),
                "Switched on." if on else "Switched off.",
            )
        finally:
            self._in_item_signal = False
        if self._refresh_pending:
            self._refresh_pending = False
            QTimer.singleShot(0, self.refresh)

    def _on_goal_changed(self, event: GoalChanged) -> None:
        if self._in_item_signal:
            self._refresh_pending = True
            return
        self.refresh()

    def _on_completed(self, event: ObjectiveCompleted) -> None:
        self._note.setText(f"Done: {event.objective.description}!")

    def _on_setback(self, event: ObjectiveSetback) -> None:
        self._note.setText(f"Setback on: {event.objective.description}.")

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._note.setText("")
        self.refresh()

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self.refresh()


__all__ = ["BY_HAND", "IDLE_TEXT", "GoalsPanel", "detection_text", "status_text"]
