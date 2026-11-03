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

from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from retro_ai_lab.core import ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.rewards import PERSONALITIES, Behaviour, Category, RewardProfile
from retro_ai_lab.services import (
    EmulatorConnected,
    EmulatorDisconnected,
    ExploitSuspected,
    RewardEarned,
    RewardProfileChanged,
)
from retro_ai_lab.ui.context import AppContext

IDLE_TEXT = "Connect to a game to choose what the AI gets points for."
CUSTOM_LABEL = "Custom"
YOU_SET_THIS = "you set this"
COLUMNS = ("Behaviour", "Reward", "Suggested", "On")


def signed(value: float) -> str:
    return f"{value:+g}"


def behaviour_cells(behaviour: Behaviour) -> tuple[str, str, str]:
    reward = signed(behaviour.value)
    if behaviour.custom:
        reward += f"  ({YOU_SET_THIS})"
    return behaviour.name, reward, signed(behaviour.suggested)


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
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.setRootIsDecorated(True)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        self._tree.itemChanged.connect(self._on_item_changed)

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

        self._warning = QLabel("", self)
        self._warning.setObjectName("rewards_warning")
        self._warning.setWordWrap(True)
        self._warning.setStyleSheet("color: #a04000;")
        self._note = QLabel("", self)
        self._note.setObjectName("rewards_note")
        self._note.setWordWrap(True)

        picker = QHBoxLayout()
        picker.addWidget(QLabel("Personality", self))
        picker.addWidget(self._personality, 1)
        controls = QHBoxLayout()
        controls.addWidget(self._minus)
        controls.addWidget(self._plus)
        controls.addWidget(self._suggested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(picker)
        layout.addWidget(self._summary)
        layout.addWidget(self._tree, 1)
        layout.addLayout(controls)
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
    def personality_id(self) -> str:
        return str(self._personality.currentData())

    def behaviour_ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def row_text(self, behaviour_id: str) -> tuple[str, ...]:
        item = self._items[behaviour_id]
        on = "on" if item.checkState(3) == Qt.CheckState.Checked else "off"
        return (item.text(0), item.text(1), item.text(2), on)

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

    def use_suggested_selected(self) -> None:
        if self._selected is not None:
            self._control(lambda bid: self._context.rewards.reset_to_suggested(bid))

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
            for category, behaviours in profile.by_category().items():
                if not behaviours:
                    continue
                top = QTreeWidgetItem([category_text(category, len(behaviours))])
                top.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self._tree.addTopLevelItem(top)
                for b in behaviours:
                    self._items[b.id] = self._add_row(top, b, enabled)
                top.setExpanded(True)
        self._tree.blockSignals(False)
        self._summary.setText(
            self._context.rewards.summary() if profile is not None and enabled else IDLE_TEXT
        )
        self._show_personality(profile)
        self._show_warning()
        self._tree.setEnabled(enabled)
        self._personality.setEnabled(enabled and profile is not None)
        keep = keep if keep is not None else self._selected
        if keep is not None and keep in self._items:
            self.select(keep)
            self._selected = keep
        else:
            self._selected = None
        self._set_controls_enabled(self._selected is not None and enabled)

    # ----------------------------------------------------------- internals

    def _add_row(self, top: QTreeWidgetItem, b: Behaviour, enabled: bool) -> QTreeWidgetItem:
        name, reward, suggested = behaviour_cells(b)
        row = QTreeWidgetItem(top, [name, reward, suggested, ""])
        row.setData(0, Qt.ItemDataRole.UserRole, b.id)
        flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
        if enabled:
            flags |= Qt.ItemFlag.ItemIsEnabled
        row.setFlags(flags)
        row.setCheckState(3, Qt.CheckState.Checked if b.enabled else Qt.CheckState.Unchecked)
        row.setToolTip(0, b.notes or b.trigger.describe())
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
        for w in (self._plus, self._minus, self._suggested):
            w.setEnabled(on)

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
        self.selection_changed.emit(self._selected)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 3:
            return
        bid = item.data(0, Qt.ItemDataRole.UserRole)
        if bid is None:
            return
        on = item.checkState(3) == Qt.CheckState.Checked
        try:
            self._context.rewards.set_enabled(str(bid), on)
        except (UnsupportedCapabilityError, ConfigurationError) as exc:
            self._note.setText(str(exc))

    def _on_personality_activated(self, index: int) -> None:
        pid = str(self._personality.itemData(index))
        if pid == "custom":
            self._show_personality(self.profile)
            return
        self.choose_personality(pid)

    def _on_profile_changed(self, event: RewardProfileChanged) -> None:
        self.refresh()

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
