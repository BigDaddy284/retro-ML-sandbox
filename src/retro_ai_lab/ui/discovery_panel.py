"""The "What the AI is doing" panel: discoveries a child can check and correct.

Shows the engine's current stage, the values it believes it has found with
a plain-words confidence badge, the evidence behind the selected one, the
latest events, and the relationships it has noticed. Three buttons let a
person answer the question DESIGN.md §3.5 asks: "Yes, that's right",
"No, that's wrong" and a rename box. "Try the buttons" runs an experiment
on demand when the emulator supports save states.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from retro_ai_lab.core import ConfigurationError, EmulatorError, UnsupportedCapabilityError
from retro_ai_lab.services import (
    DiscoveryProgress,
    DiscoveryUpdated,
    EmulatorConnected,
    EmulatorDisconnected,
)
from retro_ai_lab.ui.activity_strip import ActivityStrip
from retro_ai_lab.ui.coalesce import Coalescer
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.panels import section_heading
from retro_ai_lab.understanding import Confidence, Knowledge, Moment, Variable, describe_key

IDLE_TEXT = "Connect to a game and the AI will start watching it."
NO_SELECTION = "Pick a discovery to see why the AI thinks so."
MAX_EVENTS = 8
MAX_RELATIONSHIPS = 8
FIND_PLACEHOLDER = "Find by name or address\u2026"
TELL_BUTTONS = (
    (Moment.LOST_LIFE, "I just lost a life", "lost_life"),
    (Moment.GOT_FURTHER, "I just got further", "got_further"),
    (Moment.SCORED, "My score just went up", "scored"),
)
TELL_TIP = (
    "Press this right after it happens. The AI keeps only the bytes that moved the right "
    "way in the last few seconds; after two presses that agree it knows the value."
)

SortKey = tuple[object, ...]


def _by_address(v: Variable) -> SortKey:
    return (v.region, v.offset)


def _by_name(v: Variable) -> SortKey:
    return (v.display_name.lower(), v.region, v.offset)


def _by_sureness(v: Variable) -> SortKey:
    yours = 0 if v.confirmed else 2 if v.rejected else 1  # your word first, your "no" last
    return (yours, -v.confidence.value, v.region, v.offset)


SORTS: tuple[tuple[str, Callable[[Variable], SortKey]], ...] = (
    ("By address", _by_address),
    ("By name", _by_name),
    ("Most sure first", _by_sureness),
)

BADGES = {
    Confidence.UNKNOWN: "Not sure",
    Confidence.LOW: "A guess",
    Confidence.MEDIUM: "Probably",
    Confidence.HIGH: "Pretty sure",
    Confidence.CONFIRMED: "Confirmed by you",
}


def badge(variable: Variable) -> str:
    if variable.rejected:
        return "You said no"
    return BADGES[variable.confidence]


def variable_line(variable: Variable) -> str:
    where = f"{variable.region}[{variable.offset}]"
    return f"{variable.display_name}  ({variable.kind.label}, {where})  —  {badge(variable)}"


def _paint(item: QListWidgetItem, variable: Variable) -> None:
    """A rejected guess is greyed; one taken back keeps the usual colour."""
    if variable.rejected:
        item.setForeground(Qt.GlobalColor.gray)
    else:
        item.setData(Qt.ItemDataRole.ForegroundRole, None)


class DiscoveryPanel(QWidget):
    selection_changed = Signal(object)  # variable id or None

    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("discovery_panel")
        self._context = context
        self._selected: str | None = None

        self._stage = QLabel(IDLE_TEXT, self)
        self._stage.setObjectName("discovery_stage")
        self._stage.setWordWrap(True)
        self._summary = QLabel("", self)
        self._summary.setObjectName("discovery_summary")

        self._find = QLineEdit(self)  # T-898: hundreds of guesses need a way in
        self._find.setObjectName("discovery_find")
        self._find.setPlaceholderText(FIND_PLACEHOLDER)
        self._find.setClearButtonEnabled(True)
        self._find.textChanged.connect(self._apply_filter)
        self._found = QLabel("", self)
        self._found.setObjectName("discovery_found")
        self._sort = QComboBox(self)
        self._sort.setObjectName("discovery_sort")
        for label, _ in SORTS:
            self._sort.addItem(label)
        self._sort.currentIndexChanged.connect(self._on_sort_changed)

        self._list = QListWidget(self)
        self._list.setObjectName("discovery_list")
        self._list.currentItemChanged.connect(self._on_current_changed)

        self._evidence = QLabel(NO_SELECTION, self)
        self._evidence.setObjectName("discovery_evidence")
        self._evidence.setWordWrap(True)
        self._evidence.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._yes = QPushButton("Yes, that's right", self)
        self._yes.setObjectName("confirm_button")
        self._no = QPushButton("No, that's wrong", self)
        self._no.setObjectName("reject_button")
        self._name = QLineEdit(self)
        self._name.setObjectName("rename_edit")
        self._name.setPlaceholderText("Call it…")
        self._rename = QPushButton("Rename", self)
        self._rename.setObjectName("rename_button")
        self._experiment = QPushButton("Try the buttons", self)
        self._experiment.setObjectName("experiment_button")
        self._yes.clicked.connect(self.confirm_selected)
        self._no.clicked.connect(self.reject_selected)
        self._rename.clicked.connect(self._on_rename_clicked)
        self._name.returnPressed.connect(self._on_rename_clicked)
        self._experiment.clicked.connect(self.run_experiment)
        # T-901: the cheat-finder trick, child-sized: say when it happened.
        self._tell_buttons: dict[Moment, QPushButton] = {}
        for moment, label, key in TELL_BUTTONS:
            button = QPushButton(label, self)
            button.setObjectName(f"tell_{key}")
            button.setToolTip(TELL_TIP)
            button.clicked.connect(lambda _=False, m=moment: self.tell(m))
            self._tell_buttons[moment] = button
        self._export = QPushButton("Share what it learned…", self)
        self._export.setObjectName("export_button")
        self._import = QPushButton("Load shared knowledge…", self)
        self._import.setObjectName("import_button")
        self._export.clicked.connect(self._on_export_clicked)
        self._import.clicked.connect(self._on_import_clicked)
        self._note = QLabel("", self)
        self._note.setObjectName("discovery_note")
        self._note.setWordWrap(True)

        self._events = QListWidget(self)
        self._events.setObjectName("discovery_events")
        self._relationships = QListWidget(self)
        self._relationships.setObjectName("discovery_relationships")

        verdict = QHBoxLayout()
        verdict.addWidget(self._yes)
        verdict.addWidget(self._no)
        rename = QHBoxLayout()
        rename.addWidget(self._name, 1)
        rename.addWidget(self._rename)

        self._activity = ActivityStrip(context, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._activity)
        layout.addWidget(self._stage)
        layout.addWidget(self._summary)
        layout.addWidget(section_heading("What the AI thinks it has found", self))
        find = QHBoxLayout()
        find.addWidget(self._find, 1)
        find.addWidget(self._found)
        find.addWidget(self._sort)
        layout.addLayout(find)
        # T-900: the explanation and the verdict buttons sit above the list, so they
        # stay in view in a short panel while the list scrolls on its own.
        layout.addWidget(self._evidence)
        layout.addLayout(verdict)
        layout.addLayout(rename)
        layout.addWidget(self._list, 2)
        tell = QHBoxLayout()
        tell.addWidget(QLabel("Tell the AI:", self))
        for button in self._tell_buttons.values():
            tell.addWidget(button)
        tell.addStretch(1)
        layout.addLayout(tell)
        actions = QHBoxLayout()  # T-883: the three "do something" buttons on one row
        actions.addWidget(self._experiment)
        actions.addWidget(self._export)
        actions.addWidget(self._import)
        layout.addLayout(actions)
        layout.addWidget(self._note)
        layout.addWidget(section_heading("What just happened", self))
        layout.addWidget(self._events, 1)
        layout.addWidget(section_heading("What leads to what", self))
        layout.addWidget(self._relationships, 1)

        # Updates arrive on most training steps; the list follows a few times a
        # second at most (T-843), touching only the rows that changed (T-896).
        self._refresher = Coalescer(self._refresh_from_update, parent=self)
        self._items: dict[str, QListWidgetItem] = {}  # variable id -> its row
        self._keys: list[SortKey] = []  # the sort key of each row, in order
        self._touched: dict[str, Variable] = {}  # added or changed since the last update

        bridge = context.bridge
        self._unsubscribe = (
            bridge.subscribe(DiscoveryProgress, self._on_progress),
            bridge.subscribe(DiscoveryUpdated, self._on_updated),
            bridge.subscribe(EmulatorConnected, self._on_connected),
            bridge.subscribe(EmulatorDisconnected, self._on_disconnected),
        )
        unsubscribe = self._unsubscribe
        self.destroyed.connect(lambda *_: [u() for u in unsubscribe])
        self._set_enabled(False)
        self.refresh()

    # --------------------------------------------------------------- API

    @property
    def knowledge(self) -> Knowledge | None:
        return self._context.discovery.knowledge

    @property
    def selected_id(self) -> str | None:
        return self._selected

    @property
    def activity(self) -> ActivityStrip:
        return self._activity

    @property
    def stage_text(self) -> str:
        return self._stage.text()

    @property
    def summary_text(self) -> str:
        return self._summary.text()

    @property
    def evidence_text(self) -> str:
        return self._evidence.text()

    def variable_ids(self) -> tuple[str, ...]:
        return tuple(
            str(self._list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._list.count())
        )

    def visible_ids(self) -> tuple[str, ...]:
        """The rows the find box leaves showing, in order."""
        return tuple(
            str(self._list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._list.count())
            if not self._list.item(i).isHidden()
        )

    @property
    def find_text(self) -> str:
        return self._find.text()

    def find_rows(self, text: str) -> None:
        """Show only the rows whose line contains the text (any case); "" shows all."""
        self._find.setText(text)

    @property
    def found_text(self) -> str:
        return self._found.text()

    @property
    def sort_name(self) -> str:
        return self._sort.currentText()

    def sort_by(self, name: str) -> None:
        """Order the rows by one of the :data:`SORTS` labels."""
        index = [label for label, _ in SORTS].index(name)
        self._sort.setCurrentIndex(index)

    def item_text(self, var_id: str) -> str:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == var_id:
                return item.text()
        raise KeyError(var_id)

    def event_texts(self) -> tuple[str, ...]:
        return tuple(self._events.item(i).text() for i in range(self._events.count()))

    def relationship_texts(self) -> tuple[str, ...]:
        return tuple(self._relationships.item(i).text() for i in range(self._relationships.count()))

    def select(self, var_id: str | None) -> None:
        if var_id is None:
            self._list.clearSelection()
            self._list.setCurrentRow(-1)
            return
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == var_id:
                self._list.setCurrentItem(item)
                return
        raise KeyError(var_id)

    @property
    def note_text(self) -> str:
        return self._note.text()

    def confirm_selected(self) -> None:
        k, var_id = self._selection()
        if k is None or var_id is None:
            return
        k.confirm(var_id)
        self._saved()
        self.refresh(keep=var_id)

    def reject_selected(self) -> None:
        k, var_id = self._selection()
        if k is None or var_id is None:
            return
        k.reject(var_id)
        self._saved()
        self.refresh(keep=var_id)

    def export_to(self, path: str | Path) -> None:
        try:
            written = self._context.discovery.export_to(path)
        except ConfigurationError as exc:
            self._note.setText(str(exc))
            return
        self._note.setText(f"Saved what the AI learned to {written.name}.")

    def import_from(self, path: str | Path) -> None:
        try:
            imported = self._context.discovery.import_from(path)
        except ConfigurationError as exc:
            self._note.setText(str(exc))
            return
        self._note.setText(
            f"Loaded {len(imported)} discoveries about {imported.title or 'this game'}."
        )
        self.refresh()

    def rename_selected(self, name: str) -> None:
        """Give the selected discovery a name; naming counts as confirming it."""
        k, var_id = self._selection()
        if k is None or var_id is None or not name.strip():
            return
        k.correct(var_id, name=name.strip())
        self._name.clear()
        self._saved()
        self.refresh(keep=var_id)

    def tell(self, moment: Moment) -> None:
        """Narrow down the byte behind what the person says just happened (T-901)."""
        try:
            narrowing = self._context.discovery.tell(moment)
        except (ConfigurationError, UnsupportedCapabilityError) as exc:
            self._note.setText(str(exc))
            return
        self._note.setText(narrowing.text)

    def run_experiment(self) -> None:
        try:
            self._context.discovery.experiment()
        except UnsupportedCapabilityError as exc:
            self._set_stage(str(exc))
        except EmulatorError:
            pass  # the connection service reports link loss
        self.refresh(keep=self._selected)

    def refresh(self, *, keep: str | None = None) -> None:
        """Rebuild the lists from the knowledge."""
        k = self.knowledge
        self._list.blockSignals(True)
        self._list.clear()
        self._items = {}
        self._keys = []
        self._touched = {}
        key = self._sort_key
        if k is not None:
            for v in sorted(k, key=key):
                item = QListWidgetItem(variable_line(v))
                item.setData(Qt.ItemDataRole.UserRole, v.id)
                _paint(item, v)
                self._list.addItem(item)
                self._items[v.id] = item
                self._keys.append(key(v))
        self._list.blockSignals(False)
        self._apply_filter()
        self._summary.setText(self._context.discovery.summary())
        self._experiment.setEnabled(self._context.discovery.can_experiment())
        self._fill_events()
        self._fill_relationships()
        if keep is not None and keep in self.variable_ids():
            self.select(keep)
        else:
            self._selected = None
            self._show_evidence()

    def update_rows(self, variables: Iterable[Variable]) -> None:
        """Bring the rows of these variables up to date without rebuilding the list (T-896).

        A variable the knowledge no longer holds loses its row; a new one gets a
        row in (region, offset) order; a changed one has its line repainted. The
        selection and the scroll position stay where they are.
        """
        k = self.knowledge
        key = self._sort_key
        for v in variables:
            item = self._items.get(v.id)
            if k is None or v.id not in k:
                if item is not None:
                    self._remove_row(item, v.id)
                continue
            current = k.get(v.id)
            if item is None:
                item = QListWidgetItem(variable_line(current))
                item.setData(Qt.ItemDataRole.UserRole, current.id)
                self._insert_row(item, current.id, key(current))
            else:
                line = variable_line(current)
                if item.text() != line:
                    item.setText(line)
                if self._keys[self._list.row(item)] != key(current):  # it sorts elsewhere now
                    selected = item is self._list.currentItem()
                    self._list.blockSignals(True)
                    self._remove_row(item, current.id)
                    self._insert_row(item, current.id, key(current))
                    if selected:
                        self._list.setCurrentItem(item)
                    self._list.blockSignals(False)
            _paint(item, current)
            item.setHidden(not self._matches(item))
            if current.id == self._selected:
                self._show_evidence()
        self._show_found()
        if k is not None and len(k) != len(self._items):  # a change nobody announced
            self.refresh(keep=self._selected)

    # ----------------------------------------------------------- internals

    def _insert_row(self, item: QListWidgetItem, var_id: str, key: SortKey) -> None:
        row = bisect_left(self._keys, key)
        self._list.insertItem(row, item)
        self._keys.insert(row, key)
        self._items[var_id] = item

    def _remove_row(self, item: QListWidgetItem, var_id: str) -> None:
        row = self._list.row(item)
        self._list.takeItem(row)
        del self._keys[row]
        del self._items[var_id]

    @property
    def _sort_key(self) -> Callable[[Variable], SortKey]:
        return SORTS[self._sort.currentIndex()][1]

    def _on_sort_changed(self, _index: int) -> None:
        self.refresh(keep=self._selected)

    def _matches(self, item: QListWidgetItem) -> bool:
        wanted = self._find.text().strip().lower()
        return not wanted or wanted in item.text().lower()

    def _apply_filter(self) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(not self._matches(item))
        self._show_found()

    def _show_found(self) -> None:
        if not self._find.text().strip():
            self._found.setText("")
            return
        shown = sum(1 for i in range(self._list.count()) if not self._list.item(i).isHidden())
        self._found.setText(f"{shown} of {self._list.count()}")

    def _selection(self) -> tuple[Knowledge | None, str | None]:
        k = self.knowledge
        if k is None or self._selected is None or self._selected not in k:
            return None, None
        return k, self._selected

    def _on_current_changed(self, current: QListWidgetItem | None, _previous: object) -> None:
        self._selected = (
            str(current.data(Qt.ItemDataRole.UserRole)) if current is not None else None
        )
        self._show_evidence()
        self.selection_changed.emit(self._selected)

    def _show_evidence(self) -> None:
        k, var_id = self._selection()
        if k is None or var_id is None:
            self._evidence.setText(NO_SELECTION)
            self._set_verdict_enabled(False)
            return
        v = k.get(var_id)
        lines = [f"{v.display_name}: {badge(v)}."]
        lines.extend(f"• {e.detail}" for e in v.evidence if e.detail)
        if not v.evidence:
            lines.append("• No evidence yet.")
        self._evidence.setText("\n".join(lines))
        self._set_verdict_enabled(True)

    def _set_verdict_enabled(self, enabled: bool) -> None:
        for w in (self._yes, self._no, self._name, self._rename):
            w.setEnabled(enabled)

    def _set_enabled(self, enabled: bool) -> None:
        self._list.setEnabled(enabled)
        for button in self._tell_buttons.values():
            button.setEnabled(enabled)
        self._find.setEnabled(enabled)
        self._sort.setEnabled(enabled)
        self._export.setEnabled(enabled)
        self._import.setEnabled(enabled)
        if not enabled:
            self._set_verdict_enabled(False)
            self._experiment.setEnabled(False)

    def _fill_events(self) -> None:
        self._events.clear()
        for e in reversed(self._context.discovery.events.recent(MAX_EVENTS)):
            self._events.addItem(f"Frame {e.frame}: {e.detail}")

    def _fill_relationships(self) -> None:
        self._relationships.clear()
        k = self.knowledge
        if k is None:
            return
        rels = sorted(k.confident_relationships(), key=lambda r: (-r.confidence.value, -r.support))
        for r in rels[:MAX_RELATIONSHIPS]:
            text = f"{describe_key(r.cause, k)} leads to {describe_key(r.effect, k)}"
            self._relationships.addItem(
                f"{text}  —  {BADGES[r.confidence]} (seen {r.support} times)"
            )

    def _on_rename_clicked(self) -> None:
        self.rename_selected(self._name.text())

    def _saved(self) -> None:
        path = self._context.discovery.save()
        if path is not None:
            self._note.setText("Remembered for next time.")

    def _on_export_clicked(self) -> None:
        k = self.knowledge
        suggested = f"{k.title or k.game_id}-knowledge.json" if k else "knowledge.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Share what the AI learned", suggested, "Knowledge files (*.json)"
        )
        if path:
            self.export_to(path)

    def _on_import_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load shared knowledge", "", "Knowledge files (*.json)"
        )
        if path:
            self.import_from(path)

    # -------------------------------------------------------------- events

    def _on_progress(self, event: DiscoveryProgress) -> None:
        summary = self._context.discovery.summary()
        self._set_stage(event.detail, repeats_summary=event.detail == summary)
        self._summary.setText(summary)

    def _set_stage(self, text: str, *, repeats_summary: bool = False) -> None:
        """Show the stage line, unless it is only a copy of the summary (T-883).

        While watching, the engine puts its summary in the stage line; the summary
        line under it keeps updating, so the copy would only repeat it, then go stale.
        """
        self._stage.setText(text)
        self._stage.setVisible(not repeats_summary)

    def _on_updated(self, event: DiscoveryUpdated) -> None:
        for v in (*event.added, *event.changed):
            self._touched[v.id] = v
        self._refresher.request()

    def _refresh_from_update(self) -> None:
        touched, self._touched = self._touched, {}
        if touched:
            self.update_rows(touched.values())
        self._summary.setText(self._context.discovery.summary())
        self._experiment.setEnabled(self._context.discovery.can_experiment())
        self._fill_events()
        self._fill_relationships()

    @property
    def refresher(self) -> Coalescer:
        return self._refresher

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._set_enabled(True)
        self.refresh()

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._set_enabled(False)
        self._set_stage(IDLE_TEXT)
        self.refresh()


__all__ = [
    "BADGES",
    "IDLE_TEXT",
    "NO_SELECTION",
    "SORTS",
    "DiscoveryPanel",
    "badge",
    "variable_line",
]
