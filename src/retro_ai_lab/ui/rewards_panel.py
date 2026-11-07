"""The Rewards panel: how a child tells the AI what matters (DESIGN.md §5).

Behaviours sit under categories that expand. Each row shows the current
reward, the suggested reward and a tick to switch it on or off. The + and -
buttons move the selected reward one step; "Use suggested" puts it back.
A personality picker applies one of the starting configurations, and a
warning line explains, in plain words, when the AI looks like it is
farming points instead of getting anywhere.

Every change goes through :class:`RewardService`, so it is saved and
announced; the panel only draws what the service holds.
"""

from __future__ import annotations

from PySide6.QtCore import QModelIndex, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from retro_ai_lab.core import ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.rewards import (
    MAX_VALUE,
    PERSONALITIES,
    PICK_ONE,
    Behaviour,
    Category,
    RewardProfile,
    describe_category,
    explain_behaviour,
    is_own,
)
from retro_ai_lab.services import (
    EmulatorConnected,
    EmulatorDisconnected,
    ExploitSuspected,
    RewardEarned,
    RewardProfileChanged,
)
from retro_ai_lab.ui.coalesce import Coalescer
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.panels import name_column
from retro_ai_lab.understanding import Knowledge, plain_names
from retro_ai_lab.understanding.knowledge import address_text

IDLE_TEXT = "Connect to a game to choose what the AI gets points for."
CUSTOM_LABEL = "Custom"
YOU_SET_THIS = "you set this"
COLUMNS = ("Behaviour", "Reward", "Suggested", "On", "Address")
ADDRESS_COLUMN = 4
OWN_POINTS_DEFAULT = 2.0
OWN_POINTS_LIMIT = 50.0


def signed(value: float) -> str:
    return f"{value:+g}"


def behaviour_cells(behaviour: Behaviour) -> tuple[str, str, str]:
    reward = signed(behaviour.value)
    if behaviour.custom:
        reward += f"  ({YOU_SET_THIS})"
    return behaviour.name, reward, signed(behaviour.suggested)


def address_cell(behaviour: Behaviour, knowledge: Knowledge | None) -> str:
    """The memory address behind a row, "ram[777] (0x0309)", or "" when there is none (T-879)."""
    var_id = behaviour.trigger.variable_id
    if knowledge is None or var_id is None:
        return ""
    try:
        v = knowledge.get(var_id)
    except ConfigurationError:
        return ""  # a button name, or a value the knowledge no longer has
    return f"{v.address} ({address_text(v.region, v.offset, v.width, hex_form=True)})"


def category_text(category: Category, count: int) -> str:
    return f"{category.label} ({count})"


class RewardsPanel(QWidget):
    selection_changed = Signal(object)  # behaviour id or None

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("rewards_panel")
        self._context = context
        self._selected: str | None = None
        self._items: dict[str, QTreeWidgetItem] = {}
        self._in_item_signal = False
        self._refresh_pending = False
        # Suggestions change on many training steps; the tree is rebuilt a few
        # times a second at most (T-843).
        self._refresher = Coalescer(self._refresh_from_change, name="rewards refresh", parent=self)

        self._personality = QComboBox(self)
        self._personality.setObjectName("personality_combo")
        for p in PERSONALITIES:
            self._personality.addItem(p.name, p.id)
        self._personality.addItem(CUSTOM_LABEL, "custom")
        self._personality.activated.connect(self._on_personality_activated)

        self._summary = QLabel(IDLE_TEXT, self)
        self._summary.setObjectName("rewards_summary")
        self._summary.setWordWrap(True)

        self._tree = QTreeWidget(self)
        self._tree.setObjectName("rewards_tree")
        self._tree.setColumnCount(len(COLUMNS))
        self._tree.setHeaderLabels(list(COLUMNS))
        name_column(self._tree)
        self._tree.setRootIsDecorated(True)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        self._tree.itemChanged.connect(self._on_item_changed)

        # Type the points straight in (T-872): the suggestion may be 0.65 and the
        # +/- buttons move by whole steps, so a kid could never get rid of a decimal.
        self._points = QDoubleSpinBox(self)
        self._points.setObjectName("points_spin")
        self._points.setRange(-MAX_VALUE, MAX_VALUE)
        self._points.setDecimals(2)
        self._points.setSingleStep(1.0)
        self._points.setKeyboardTracking(False)  # applies on Enter or leaving the box
        self._points.setToolTip(
            "Type the points for the picked reward; below zero takes points away"
        )
        self._points.valueChanged.connect(self._on_points_edited)
        self._plus = QPushButton("+", self)
        self._plus.setObjectName("plus_button")
        self._plus.setToolTip("Worth a bit more")
        self._minus = QPushButton("-", self)
        self._minus.setObjectName("minus_button")
        self._minus.setToolTip("Worth a bit less")
        self._suggested = QPushButton("Use suggested", self)
        self._suggested.setObjectName("suggested_button")
        self._plus.clicked.connect(self.increase_selected)
        self._minus.clicked.connect(self.decrease_selected)
        self._suggested.clicked.connect(self.use_suggested_selected)

        self._explain = QLabel(PICK_ONE, self)
        self._explain.setObjectName("rewards_explain")
        self._explain.setWordWrap(True)

        # A reward of the child's own: a value, a direction and the points (T-861).
        self._own_value = QComboBox(self)
        self._own_value.setObjectName("own_value_combo")
        self._own_direction = QComboBox(self)
        self._own_direction.setObjectName("own_direction_combo")
        self._own_direction.addItem("goes up", True)
        self._own_direction.addItem("goes down", False)
        self._own_points = QDoubleSpinBox(self)
        self._own_points.setObjectName("own_points_spin")
        self._own_points.setRange(-OWN_POINTS_LIMIT, OWN_POINTS_LIMIT)
        self._own_points.setDecimals(1)
        self._own_points.setSingleStep(1.0)
        self._own_points.setValue(OWN_POINTS_DEFAULT)
        self._own_points.setToolTip("Points each time it happens; below zero takes points away")
        self._add_own = QPushButton("Add a reward", self)
        self._add_own.setObjectName("add_own_button")
        self._add_own.clicked.connect(self._on_add_own_clicked)
        self._remove_own = QPushButton("Remove", self)
        self._remove_own.setObjectName("remove_own_button")
        self._remove_own.setToolTip(
            "Take out the picked reward. A suggested row stays out for this game, whichever "
            'value the AI finds next; "Bring back suggestions" undoes that.'
        )
        self._remove_own.clicked.connect(self.remove_selected)
        self._suggest_again = QPushButton("Bring back suggestions", self)
        self._suggest_again.setObjectName("suggest_again_button")
        self._suggest_again.setToolTip("Put back every suggested row you removed.")
        self._suggest_again.clicked.connect(self.suggest_again)

        self._warning = QLabel("", self)
        self._warning.setObjectName("rewards_warning")
        self._warning.setWordWrap(True)
        self._warning.setProperty("warning", True)  # the theme picks the colour (T-876)
        self._note = QLabel("", self)
        self._note.setObjectName("rewards_note")
        self._note.setWordWrap(True)

        picker = QHBoxLayout()
        picker.addWidget(QLabel("Personality", self))
        picker.addWidget(self._personality, 1)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Points", self))
        controls.addWidget(self._points)
        controls.addWidget(self._minus)
        controls.addWidget(self._plus)
        controls.addWidget(self._suggested)
        controls.addWidget(self._remove_own)
        controls.addWidget(self._suggest_again)
        own = QHBoxLayout()
        own.addWidget(QLabel("Add a reward:", self))
        own.addWidget(self._own_value, 1)
        own.addWidget(self._own_direction)
        own.addWidget(self._own_points)
        own.addWidget(self._add_own)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(picker)
        layout.addWidget(self._summary)
        layout.addWidget(self._tree, 1)
        layout.addWidget(self._explain)
        layout.addLayout(controls)
        layout.addLayout(own)
        layout.addWidget(self._warning)
        layout.addWidget(self._note)

        bridge = context.bridge
        self._unsubscribe = (
            bridge.subscribe(RewardProfileChanged, self._on_profile_changed),
            bridge.subscribe(RewardEarned, self._on_earned),
            bridge.subscribe(ExploitSuspected, self._on_exploit),
            bridge.subscribe(EmulatorConnected, self._on_connected),
            bridge.subscribe(EmulatorDisconnected, self._on_disconnected),
        )
        unsubscribe = self._unsubscribe
        self.destroyed.connect(lambda *_: [u() for u in unsubscribe])
        self.refresh()

    # --------------------------------------------------------------- API

    @property
    def profile(self) -> RewardProfile | None:
        return self._context.rewards.profile

    @property
    def selected_id(self) -> str | None:
        return self._selected

    @property
    def summary_text(self) -> str:
        return self._summary.text()

    @property
    def warning_text(self) -> str:
        return self._warning.text()

    @property
    def note_text(self) -> str:
        return self._note.text()

    @property
    def explain_text(self) -> str:
        """What the picked reward means, in a sentence (T-859)."""
        return self._explain.text()

    @property
    def personality_id(self) -> str:
        return str(self._personality.currentData())

    def behaviour_ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def row_text(self, behaviour_id: str) -> tuple[str, ...]:
        item = self._items[behaviour_id]
        on = "on" if item.checkState(3) == Qt.CheckState.Checked else "off"
        return (item.text(0), item.text(1), item.text(2), on, item.text(ADDRESS_COLUMN))

    def category_texts(self) -> tuple[str, ...]:
        out = []
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item is not None:
                out.append(item.text(0))
        return tuple(out)

    def select(self, behaviour_id: str | None) -> None:
        if behaviour_id is None:
            self._tree.clearSelection()
            self._tree.setCurrentIndex(QModelIndex())
            return
        self._tree.setCurrentItem(self._items[behaviour_id])

    def increase_selected(self) -> None:
        self._control(lambda bid: self._context.rewards.increase(bid))

    def decrease_selected(self) -> None:
        self._control(lambda bid: self._context.rewards.decrease(bid))

    def set_selected_value(self, value: float) -> None:
        """Give the picked reward exactly these points (the Points box, T-872)."""
        self._control(lambda bid: self._context.rewards.set_value(bid, value))

    @property
    def points_value(self) -> float:
        """What the Points box shows."""
        return float(self._points.value())

    def use_suggested_selected(self) -> None:
        if self._selected is not None:
            self._control(lambda bid: self._context.rewards.reset_to_suggested(bid))

    def own_choices(self) -> tuple[tuple[str, str], ...]:
        """(variable id, name) pairs a reward of one's own can be put on (T-861)."""
        return tuple(
            (str(self._own_value.itemData(i)), self._own_value.itemText(i))
            for i in range(self._own_value.count())
        )

    def add_own(self, variable_id: str, goes_up: bool, value: float) -> None:
        """Add a reward of the child's own; the note says how it went."""
        name = next((n for vid, n in self.own_choices() if vid == variable_id), None)
        try:
            b = self._context.rewards.add_own(variable_id, goes_up=goes_up, value=value, name=name)
        except (UnsupportedCapabilityError, ConfigurationError) as exc:
            self._note.setText(str(exc))
            return
        self._note.setText("Reward added.")
        self.select(b.id)

    def remove_selected(self) -> None:
        """Take out the picked row: one of my own, or the AI's guess behind it (T-937)."""
        bid = self._selected
        profile = self.profile
        if bid is None or profile is None:
            return
        try:
            if is_own(bid):
                self._context.rewards.remove_own(bid)
                self._note.setText("Reward removed.")
                return
            behaviour = profile.get(bid)
            self._context.rewards.dismiss(bid)
        except (UnsupportedCapabilityError, ConfigurationError) as exc:
            self._note.setText(str(exc))
            return
        self._note.setText(
            f'Removed "{behaviour.name}". It stays out for this game, whichever value the AI '
            'finds next; "Bring back suggestions" undoes that.'
        )

    def suggest_again(self) -> None:
        """Forget every removal of a suggested row and bring them back (T-943)."""
        try:
            report = self._context.rewards.suggest_again()
        except (UnsupportedCapabilityError, ConfigurationError) as exc:
            self._note.setText(str(exc))
            return
        count = len(report.added)
        self._note.setText(
            f"Brought back {count} suggested reward{'s' if count != 1 else ''}."
            if count
            else "Nothing was removed; the suggestions are all here."
        )

    def set_selected_enabled(self, enabled: bool) -> None:
        self._control(lambda bid: self._context.rewards.set_enabled(bid, enabled))

    def choose_personality(self, personality_id: str) -> None:
        try:
            chosen = self._context.rewards.choose_personality(personality_id)
        except (UnsupportedCapabilityError, ConfigurationError) as exc:
            self._note.setText(str(exc))
            return
        self._note.setText(f"{chosen.name}: {chosen.description}")

    def refresh(self, *, keep: str | None = None) -> None:
        """Rebuild the tree from the profile."""
        profile = self.profile
        enabled = self._context.rewards.enabled
        self._tree.blockSignals(True)
        self._tree.clear()
        self._items.clear()
        if profile is not None:
            knowledge = self._context.discovery.knowledge
            for category, behaviours in profile.by_category().items():
                if not behaviours:
                    continue
                top = QTreeWidgetItem([category_text(category, len(behaviours))])
                top.setFlags(Qt.ItemFlag.ItemIsEnabled)
                top.setToolTip(0, describe_category(category))
                self._tree.addTopLevelItem(top)
                for b in behaviours:
                    self._items[b.id] = self._add_row(top, b, enabled, knowledge)
                top.setExpanded(True)
        self._tree.blockSignals(False)
        self._summary.setText(
            self._context.rewards.summary() if profile is not None and enabled else IDLE_TEXT
        )
        self._show_personality(profile)
        self._show_warning()
        self._fill_own_choices()
        self._tree.setEnabled(enabled)
        self._personality.setEnabled(enabled and profile is not None)
        can_add = enabled and profile is not None and self._own_value.count() > 0
        for w in (self._own_value, self._own_direction, self._own_points, self._add_own):
            w.setEnabled(can_add)
        keep = keep if keep is not None else self._selected
        if keep is not None and keep in self._items:
            self.select(keep)
            self._selected = keep
        else:
            self._selected = None
        self._set_controls_enabled(self._selected is not None and enabled)
        self._show_explanation()

    # ----------------------------------------------------------- internals

    def _add_row(
        self, top: QTreeWidgetItem, b: Behaviour, enabled: bool, knowledge: Knowledge | None
    ) -> QTreeWidgetItem:
        name, reward, suggested = behaviour_cells(b)
        row = QTreeWidgetItem(top, [name, reward, suggested, "", address_cell(b, knowledge)])
        row.setData(0, Qt.ItemDataRole.UserRole, b.id)
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
        if enabled:
            flags |= Qt.ItemFlag.ItemIsEnabled
        row.setFlags(flags)
        row.setCheckState(3, Qt.CheckState.Checked if b.enabled else Qt.CheckState.Unchecked)
        row.setToolTip(0, explain_behaviour(b))
        if not b.enabled:
            row.setForeground(0, Qt.GlobalColor.gray)
        return row

    def _control(self, action: object) -> None:
        bid = self._selected
        if bid is None:
            return
        try:
            action(bid)  # type: ignore[operator]
        except (UnsupportedCapabilityError, ConfigurationError) as exc:
            self._note.setText(str(exc))
            return
        self._note.setText("Remembered for next time.")

    def _set_controls_enabled(self, on: bool) -> None:
        for w in (self._plus, self._minus, self._suggested, self._points):
            w.setEnabled(on)
        self._show_points()
        self._remove_own.setEnabled(on and self._selected is not None)
        # Bringing back what was removed needs no row picked (T-943).
        profile = self.profile
        enabled = self._tree.isEnabled() and profile is not None and bool(profile.dismissed)
        self._suggest_again.setEnabled(enabled)

    def _fill_own_choices(self) -> None:
        """The AI's offered guesses by their plain names, then any other named value."""
        knowledge = self._context.discovery.knowledge
        current = self._own_value.currentData()
        self._own_value.blockSignals(True)
        self._own_value.clear()
        if knowledge is not None and self._context.rewards.enabled:
            offered = knowledge.offered()
            names = plain_names(offered)
            seen: set[str] = set()
            for v in offered:
                self._own_value.addItem(names[v.id], v.id)
                seen.add(v.id)
            for v in knowledge:
                if v.name and not v.rejected and v.id not in seen:
                    self._own_value.addItem(v.name, v.id)
        index = self._own_value.findData(current) if current else 0
        self._own_value.setCurrentIndex(max(index, 0))
        self._own_value.blockSignals(False)

    def _on_add_own_clicked(self) -> None:
        vid = self._own_value.currentData()
        if vid is None:
            return
        self.add_own(str(vid), bool(self._own_direction.currentData()), self._own_points.value())

    def _show_personality(self, profile: RewardProfile | None) -> None:
        wanted = profile.personality if profile is not None else "custom"
        index = self._personality.findData(wanted)
        if index < 0:
            index = self._personality.findData("custom")
        self._personality.blockSignals(True)
        self._personality.setCurrentIndex(index)
        self._personality.blockSignals(False)

    def _show_warning(self) -> None:
        current = self._context.rewards.current_warning
        if current is None or not self._context.rewards.enabled:
            self._warning.setText("")
            return
        w = current.warning
        self._warning.setText(f"Hmm. {w.explanation}\n{w.suggestion}")

    def _on_current_changed(self, current: QTreeWidgetItem | None, _previous: object) -> None:
        bid = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        self._selected = str(bid) if bid is not None else None
        self._set_controls_enabled(self._selected is not None and self._context.rewards.enabled)
        self._show_explanation()
        self.selection_changed.emit(self._selected)

    def _show_points(self) -> None:
        profile = self.profile
        b = profile.find(self._selected) if profile is not None and self._selected else None
        self._points.blockSignals(True)
        self._points.setValue(float(b.value) if b is not None else 0.0)
        self._points.blockSignals(False)

    def _on_points_edited(self, value: float) -> None:
        profile = self.profile
        b = profile.find(self._selected) if profile is not None and self._selected else None
        if b is None or float(b.value) == float(value):
            return
        self.set_selected_value(float(value))

    def _show_explanation(self) -> None:
        profile = self.profile
        b = profile.find(self._selected) if profile is not None and self._selected else None
        self._explain.setText(explain_behaviour(b) if b is not None else PICK_ONE)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 3:
            return
        bid = item.data(0, Qt.ItemDataRole.UserRole)
        if bid is None:
            return
        on = item.checkState(3) == Qt.CheckState.Checked
        # Qt is still inside this item's signal: rebuilding the tree now would delete
        # the item under its feet (a crash on Windows), so any refresh waits a tick.
        self._in_item_signal = True
        try:
            self._context.rewards.set_enabled(str(bid), on)
        except (UnsupportedCapabilityError, ConfigurationError) as exc:
            self._note.setText(str(exc))
        finally:
            self._in_item_signal = False
        if self._refresh_pending:
            self._refresh_pending = False
            QTimer.singleShot(0, self.refresh)

    def _on_personality_activated(self, index: int) -> None:
        pid = str(self._personality.itemData(index))
        if pid == "custom":
            self._show_personality(self.profile)
            return
        self.choose_personality(pid)

    def _on_profile_changed(self, event: RewardProfileChanged) -> None:
        if event.reason == "suggestions":
            self._refresher.request()  # the AI's churn: a few times a second is plenty
        else:
            self._refresher.cancel()
            self._refresh_from_change()  # a person's change shows at once

    def _refresh_from_change(self) -> None:
        if self._in_item_signal:
            self._refresh_pending = True
            return
        self.refresh()

    @property
    def refresher(self) -> Coalescer:
        return self._refresher

    def _on_earned(self, event: RewardEarned) -> None:
        self._summary.setText(self._context.rewards.summary())
        self._show_warning()

    def _on_exploit(self, event: ExploitSuspected) -> None:
        self._show_warning()

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._note.setText("")
        self.refresh()

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self.refresh()


__all__ = ["CUSTOM_LABEL", "IDLE_TEXT", "YOU_SET_THIS", "RewardsPanel", "behaviour_cells"]
