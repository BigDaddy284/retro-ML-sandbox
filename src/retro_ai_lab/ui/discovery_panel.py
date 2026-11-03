"""The "What the AI is doing" panel: discoveries a child can check and correct.

Shows the engine's current stage, the values it believes it has found with
a plain-words confidence badge, the evidence behind the selected one, the
latest events, and the relationships it has noticed. Three buttons let a
person answer the question DESIGN.md §3.5 asks: "Yes, that's right",
"No, that's wrong" and a rename box. "Try the buttons" runs an experiment
on demand when the emulator supports save states.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
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
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.understanding import Confidence, Knowledge, Variable, describe_key

IDLE_TEXT = "Connect to a game and the AI will start watching it."
NO_SELECTION = "Pick a discovery to see why the AI thinks so."
MAX_EVENTS = 8
MAX_RELATIONSHIPS = 8

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
        layout.addWidget(QLabel("What the AI thinks it has found", self))
        layout.addWidget(self._list, 2)
        layout.addWidget(self._evidence)
        layout.addLayout(verdict)
        layout.addLayout(rename)
        layout.addWidget(self._experiment)
        files = QHBoxLayout()
        files.addWidget(self._export)
        files.addWidget(self._import)
        layout.addLayout(files)
        layout.addWidget(self._note)
        layout.addWidget(QLabel("What just happened", self))
        layout.addWidget(self._events, 1)
        layout.addWidget(QLabel("What leads to what", self))
        layout.addWidget(self._relationships, 1)

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

    def run_experiment(self) -> None:
        try:
            self._context.discovery.experiment()
        except UnsupportedCapabilityError as exc:
            self._stage.setText(str(exc))
        except EmulatorError:
            pass  # the connection service reports link loss
        self.refresh(keep=self._selected)

    def refresh(self, *, keep: str | None = None) -> None:
        """Rebuild the lists from the knowledge."""
        k = self.knowledge
        self._list.blockSignals(True)
        self._list.clear()
        if k is not None:
            for v in sorted(k, key=lambda v: (v.region, v.offset)):
                item = QListWidgetItem(variable_line(v))
                item.setData(Qt.ItemDataRole.UserRole, v.id)
                if v.rejected:
                    item.setForeground(Qt.GlobalColor.gray)
                self._list.addItem(item)
        self._list.blockSignals(False)
        self._summary.setText(self._context.discovery.summary())
        self._experiment.setEnabled(self._context.discovery.can_experiment())
        self._fill_events()
        self._fill_relationships()
        if keep is not None and keep in self.variable_ids():
            self.select(keep)
        else:
            self._selected = None
            self._show_evidence()

    # ----------------------------------------------------------- internals

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
        self._stage.setText(event.detail)
        self._summary.setText(self._context.discovery.summary())

    def _on_updated(self, event: DiscoveryUpdated) -> None:
        self.refresh(keep=self._selected)

    def _on_connected(self, event: EmulatorConnected) -> None:
        self._set_enabled(True)
        self.refresh()

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        self._set_enabled(False)
        self._stage.setText(IDLE_TEXT)
        self.refresh()


__all__ = ["BADGES", "IDLE_TEXT", "NO_SELECTION", "DiscoveryPanel", "badge", "variable_line"]
