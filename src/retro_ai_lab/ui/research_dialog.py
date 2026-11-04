"""The Research window: learning settings, the situation now, the step log.

DESIGN.md §16 Phase 8 "Advanced research mode". It is one window off the toolbar,
so the six panels stay as they are for the child; a grown-up or an older child
opens it to change how the AI learns, to see what the AI thinks every move is
worth right now, and to take the step log away as a CSV file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QModelIndex, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
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
from retro_ai_lab.learning import Settings
from retro_ai_lab.services import EmulatorConnected, EmulatorDisconnected
from retro_ai_lab.training import (
    COLUMNS,
    EpisodeFinished,
    SituationReport,
    Status,
    StepRecord,
    TrainingStateChanged,
)
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.panels import name_column
from retro_ai_lab.understanding import Confidence, Variable

REFRESH_MS = 500
MAX_ROWS = 200
IDLE_TEXT = "Connect to a game the AI can learn to see what it is thinking."
NO_SITUATION_TEXT = "Nothing observed yet. Play or train a little first."
SITUATION_COLUMNS: tuple[str, ...] = ("Move", "Value", "Best")
MEMORY_COLUMNS: tuple[str, ...] = ("Address", "Now", "Changes", "The AI thinks", "Name")
MAX_MEMORY_ROWS = 400
NO_MEMORY_TEXT = "Connect to a game that shares its memory to see it here."
MEMORY_WAITING_TEXT = "Watching the memory; play or train a little and the values appear."
SURENESS: dict[Confidence, str] = {
    Confidence.UNKNOWN: "no idea yet",
    Confidence.LOW: "a hunch",
    Confidence.MEDIUM: "fairly sure",
    Confidence.HIGH: "sure",
    Confidence.CONFIRMED: "confirmed",
}
LOG_COLUMNS: tuple[str, ...] = tuple(c.replace("_", " ").capitalize() for c in COLUMNS)
SETTING_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("alpha", "Learning rate (alpha)", "How far a value moves towards what happened, 0 to 1."),
    ("gamma", "Future counts (gamma)", "How much later rewards count against the next one."),
    ("epsilon", "Exploring (epsilon)", "Chance of trying a random move at the start."),
    ("epsilon_min", "Exploring floor", "Exploring never drops below this."),
    ("epsilon_decay", "Exploring decay", "Exploring is multiplied by this after every go."),
    ("explore_steps", "Exploring steps", "Learned steps over which exploring falls to the floor."),
)
WHOLE_NUMBER_FIELDS: dict[str, tuple[int, int]] = {"explore_steps": (1, 1_000_000)}
WHAT_IS_THIS = (
    "For grown-ups and researchers: how the AI learns, what it thinks right now and "
    "every step it took. Changing the settings affects the next goes, not the ones done."
)


def settings_summary(log_count: int, dropped: int) -> str:
    steps = "step" if log_count == 1 else "steps"
    text = f"{log_count:,} {steps} logged"
    if dropped:
        text += f", {dropped:,} older ones dropped"
    return text + "."


class ResearchDialog(QDialog):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("research_dialog")
        self.setWindowTitle("Research")
        self.setModal(False)
        self.resize(760, 640)
        self._context = context
        self._training = context.training
        self._log = context.step_log

        intro = QLabel(WHAT_IS_THIS, self)
        intro.setObjectName("research_intro")
        intro.setWordWrap(True)

        # --- learning settings
        settings_box = QGroupBox("Learning settings", self)
        settings_box.setObjectName("settings_box")
        form = QFormLayout(settings_box)
        self._spins: dict[str, QDoubleSpinBox] = {}
        for key, label, tip in SETTING_FIELDS:
            spin = QDoubleSpinBox(settings_box)
            spin.setObjectName(f"setting_{key}")
            if key in WHOLE_NUMBER_FIELDS:
                low, high = WHOLE_NUMBER_FIELDS[key]
                spin.setRange(low, high)
                spin.setDecimals(0)
                spin.setSingleStep(100)
            else:
                spin.setRange(0.0, 1.0)
                spin.setDecimals(3)
                spin.setSingleStep(0.01)
            spin.setToolTip(tip)
            form.addRow(label, spin)
            self._spins[key] = spin
        self._apply = QPushButton("Apply", settings_box)
        self._apply.setObjectName("apply_settings_button")
        self._apply.clicked.connect(self.apply_settings)
        self._revert = QPushButton("Back to what it is now", settings_box)
        self._revert.setObjectName("revert_settings_button")
        self._revert.clicked.connect(self._show_settings)
        buttons = QHBoxLayout()
        buttons.addWidget(self._apply)
        buttons.addWidget(self._revert)
        buttons.addStretch(1)
        form.addRow(buttons)

        # --- the situation now
        situation_box = QGroupBox("The situation now", self)
        situation_box.setObjectName("situation_box")
        self._situation = QLabel(IDLE_TEXT, situation_box)
        self._situation.setObjectName("situation_text")
        self._situation.setWordWrap(True)
        self._values = QTreeWidget(situation_box)
        self._values.setObjectName("situation_table")
        self._values.setColumnCount(len(SITUATION_COLUMNS))
        self._values.setHeaderLabels(list(SITUATION_COLUMNS))
        name_column(self._values)
        self._values.setRootIsDecorated(False)
        situation_layout = QVBoxLayout(situation_box)
        situation_layout.addWidget(self._situation)
        situation_layout.addWidget(self._values, 1)

        # --- the memory (T-862)
        memory_box = QGroupBox("Memory", self)
        memory_box.setObjectName("memory_box")
        self._memory_summary = QLabel(NO_MEMORY_TEXT, memory_box)
        self._memory_summary.setObjectName("memory_summary")
        self._memory_summary.setWordWrap(True)
        self._memory_filter = QLineEdit(memory_box)
        self._memory_filter.setObjectName("memory_filter")
        self._memory_filter.setPlaceholderText("Filter by address (ram[12], 0x0c) or name")
        self._memory_filter.textChanged.connect(lambda _t: self._show_memory())
        self._memory_table = QTreeWidget(memory_box)
        self._memory_table.setObjectName("memory_table")
        self._memory_table.setColumnCount(len(MEMORY_COLUMNS))
        self._memory_table.setHeaderLabels(list(MEMORY_COLUMNS))
        self._memory_table.setRootIsDecorated(False)
        self._memory_table.header().setSectionResizeMode(
            len(MEMORY_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch
        )
        self._memory_table.currentItemChanged.connect(self._on_memory_current_changed)
        self._memory_selected: str | None = None  # "region[offset]"
        self._memory_name = QLineEdit(memory_box)
        self._memory_name.setObjectName("memory_name_edit")
        self._memory_name.setPlaceholderText("Name the picked value, in your words")
        self._memory_name_button = QPushButton("Name it", memory_box)
        self._memory_name_button.setObjectName("memory_name_button")
        self._memory_name_button.clicked.connect(self._on_memory_name_clicked)
        self._memory_name.returnPressed.connect(self._on_memory_name_clicked)
        naming = QHBoxLayout()
        naming.addWidget(self._memory_name, 1)
        naming.addWidget(self._memory_name_button)
        memory_layout = QVBoxLayout(memory_box)
        memory_layout.addWidget(self._memory_summary)
        memory_layout.addWidget(self._memory_filter)
        memory_layout.addWidget(self._memory_table, 1)
        memory_layout.addLayout(naming)

        # --- the step log
        log_box = QGroupBox("Step log", self)
        log_box.setObjectName("log_box")
        self._log_summary = QLabel("", log_box)
        self._log_summary.setObjectName("log_summary")
        self._log_table = QTreeWidget(log_box)
        self._log_table.setObjectName("log_table")
        self._log_table.setColumnCount(len(LOG_COLUMNS))
        self._log_table.setHeaderLabels(list(LOG_COLUMNS))
        self._log_table.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._log_table.setRootIsDecorated(False)
        self._export = QPushButton("Save as CSV…", log_box)
        self._export.setObjectName("export_log_button")
        self._export.clicked.connect(self._ask_export)
        self._clear = QPushButton("Clear", log_box)
        self._clear.setObjectName("clear_log_button")
        self._clear.clicked.connect(self.clear_log)
        log_buttons = QHBoxLayout()
        log_buttons.addWidget(self._log_summary, 1)
        log_buttons.addWidget(self._export)
        log_buttons.addWidget(self._clear)
        log_layout = QVBoxLayout(log_box)
        log_layout.addLayout(log_buttons)
        log_layout.addWidget(self._log_table, 1)

        self._note = QLabel("", self)
        self._note.setObjectName("research_note")
        self._note.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(settings_box)
        layout.addWidget(situation_box, 1)
        layout.addWidget(memory_box, 2)
        layout.addWidget(log_box, 2)
        layout.addWidget(self._note)

        self._ticker = QTimer(self)
        self._ticker.setObjectName("research_ticker")
        self._ticker.setInterval(REFRESH_MS)
        self._ticker.timeout.connect(self._refresh_live)

        bridge = context.bridge
        unsubscribe = (
            bridge.subscribe(TrainingStateChanged, self._on_changed),
            bridge.subscribe(EpisodeFinished, self._on_changed),
            bridge.subscribe(EmulatorConnected, self._on_changed),
            bridge.subscribe(EmulatorDisconnected, self._on_changed),
        )
        ticker = self._ticker

        def release(*_: object) -> None:
            ticker.stop()
            for off in unsubscribe:
                off()

        self.destroyed.connect(release)
        self.refresh()

    # ------------------------------------------------------------- API

    @property
    def note_text(self) -> str:
        return self._note.text()

    @property
    def situation_text(self) -> str:
        return self._situation.text()

    @property
    def log_summary(self) -> str:
        return self._log_summary.text()

    @property
    def ticking(self) -> bool:
        return self._ticker.isActive()

    def settings_values(self) -> dict[str, float]:
        return {key: spin.value() for key, spin in self._spins.items()}

    def set_settings_values(self, **values: float) -> None:
        for key, value in values.items():
            if key not in self._spins:
                raise ConfigurationError(f"There is no setting called {key!r}.")
            self._spins[key].setValue(float(value))

    def apply_settings(self) -> bool:
        """Make the boxes the live settings; False (with a note) when refused."""
        try:
            values: dict[str, Any] = dict(self.settings_values())
            settings = Settings(**values)
            self._training.set_settings(settings)
        except (UnsupportedCapabilityError, ConfigurationError) as exc:
            self._note.setText(str(exc))
            return False
        self._note.setText("Settings applied. The next goes learn with them.")
        self.refresh()
        return True

    @property
    def memory_summary(self) -> str:
        return self._memory_summary.text()

    def memory_rows(self) -> tuple[tuple[str, ...], ...]:
        """(address, now, changes, the AI thinks, name) per row shown."""
        out = []
        for i in range(self._memory_table.topLevelItemCount()):
            item = self._memory_table.topLevelItem(i)
            if item is not None:
                out.append(tuple(item.text(c) for c in range(len(MEMORY_COLUMNS))))
        return tuple(out)

    def set_memory_filter(self, text: str) -> None:
        self._memory_filter.setText(text)

    def select_memory(self, address: str | None) -> None:
        """Pick a row by its address text, such as "ram[12]"."""
        if address is None:
            self._memory_table.clearSelection()
            self._memory_table.setCurrentIndex(QModelIndex())
            return
        for i in range(self._memory_table.topLevelItemCount()):
            item = self._memory_table.topLevelItem(i)
            if item is not None and item.text(0) == address:
                self._memory_table.setCurrentItem(item)
                return
        raise ConfigurationError(f"No row at {address}.")

    def name_memory(self, name: str) -> bool:
        """Name the picked address; the note says how it went."""
        address = self._memory_selected
        if address is None:
            self._note.setText("Pick a value in the memory table first.")
            return False
        region, _, rest = address.partition("[")
        try:
            var = self._context.discovery.name_value(region, int(rest.rstrip("]")), name)
        except (UnsupportedCapabilityError, ConfigurationError, ValueError) as exc:
            self._note.setText(str(exc))
            return False
        self._note.setText(f"{var.display_name}: remembered for next time.")
        self._memory_name.clear()
        self._show_memory()
        return True

    def situation_rows(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (item.text(0), item.text(1), item.text(2))
            for item in (
                self._values.topLevelItem(i) for i in range(self._values.topLevelItemCount())
            )
            if item is not None
        )

    def log_rows(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(item.text(c) for c in range(len(LOG_COLUMNS)))
            for item in (
                self._log_table.topLevelItem(i) for i in range(self._log_table.topLevelItemCount())
            )
            if item is not None
        )

    def export_to(self, path: Path) -> int:
        """Write the log to ``path``; returns rows written (a note says where it went)."""
        try:
            count = self._log.write_csv(Path(path))
        except OSError as exc:
            self._note.setText(f"Couldn't save the log: {exc}")
            return 0
        self._note.setText(f"Saved {count:,} steps to {Path(path).name}.")
        return count

    def clear_log(self) -> None:
        self._log.clear()
        self._show_log()

    def refresh(self) -> None:
        self._show_settings()
        self._refresh_live()

    # ------------------------------------------------------------- Qt

    def showEvent(self, event: object) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)  # type: ignore[arg-type]
        self.refresh()
        self._ticker.start()

    def hideEvent(self, event: object) -> None:  # noqa: N802 (Qt override)
        self._ticker.stop()
        super().hideEvent(event)  # type: ignore[arg-type]

    # ----------------------------------------------------------- internals

    def _show_settings(self) -> None:
        t = self._training
        editable = t.enabled and t.status is Status.IDLE
        for key, value in t.settings.to_dict().items():
            spin = self._spins[key]
            if spin.value() != value:
                spin.setValue(value)
            spin.setEnabled(editable)
        self._apply.setEnabled(editable)
        self._revert.setEnabled(editable)

    def _refresh_live(self) -> None:
        self._show_situation()
        self._show_memory()
        self._show_log()

    def _memory_addresses(self) -> tuple[tuple[str, int], ...]:
        """Bytes that behave like game state, plus every address the AI has a guess at."""
        observation = self._context.observation
        knowledge = self._context.discovery.knowledge
        keys: set[tuple[str, int]] = set(observation.tracker.classify()["state"])
        if knowledge is not None:
            for v in knowledge:
                if not v.rejected:
                    keys.add((v.region, v.offset))
        return tuple(sorted(keys))

    def _show_memory(self) -> None:
        keep = self._memory_selected
        table = self._memory_table
        table.blockSignals(True)
        table.clear()
        snapshot = self._context.observation.latest
        enabled = self._context.discovery.enabled and snapshot is not None
        if not enabled or snapshot is None:
            self._memory_summary.setText(NO_MEMORY_TEXT)
            table.blockSignals(False)
            self._memory_selected = None
            self._memory_name_button.setEnabled(False)
            return
        knowledge = self._context.discovery.knowledge
        tracker = self._context.observation.tracker
        wanted = self._memory_filter.text().strip().lower()
        addresses = self._memory_addresses()
        shown = 0
        named = 0
        chosen: QTreeWidgetItem | None = None
        for region, offset in addresses:
            var = knowledge.at(region, offset) if knowledge is not None else None
            if var is not None and var.rejected:
                var = None
            address = f"{region}[{offset}]"
            name = var.name if var is not None else ""
            if name:
                named += 1
            if wanted and not _memory_matches(wanted, address, offset, name, var):
                continue
            if shown >= MAX_MEMORY_ROWS:
                shown += 1
                continue
            try:
                now = snapshot.byte(region, offset)
            except (ConfigurationError, KeyError, IndexError):
                continue
            stats = tracker.stats(region, offset)
            item = QTreeWidgetItem(
                [
                    address,
                    f"{now} (0x{now:02x})",
                    f"{stats.change_rate:.0%} of steps" if stats.samples else "not seen yet",
                    _thinks(var),
                    name,
                ]
            )
            table.addTopLevelItem(item)
            if address == keep:
                chosen = item
            shown += 1
        table.blockSignals(False)
        if chosen is not None:
            table.setCurrentItem(chosen)
        else:
            self._memory_selected = None
        self._memory_name_button.setEnabled(self._memory_selected is not None)
        if not addresses:
            self._memory_summary.setText(MEMORY_WAITING_TEXT)
            return
        text = f"{len(addresses):,} values that change like game state or carry a guess"
        if named:
            text += f"; {named} named"
        if shown > MAX_MEMORY_ROWS:
            text += f"; showing the first {MAX_MEMORY_ROWS:,}, use the filter"
        self._memory_summary.setText(text + ".")

    def _on_memory_current_changed(self, current: QTreeWidgetItem | None, _p: object) -> None:
        self._memory_selected = current.text(0) if current is not None else None
        self._memory_name_button.setEnabled(self._memory_selected is not None)

    def _on_memory_name_clicked(self) -> None:
        self.name_memory(self._memory_name.text())

    def _show_situation(self) -> None:
        t = self._training
        self._values.clear()
        if not t.enabled or t.agent is None:
            self._situation.setText(IDLE_TEXT)
            return
        state = t.current_state()
        if state is None:
            self._situation.setText(NO_SITUATION_TEXT)
            return
        report = SituationReport.of(t.agent, state)
        knowledge = self._context.discovery.knowledge
        self._situation.setText(report.describe(knowledge))
        for value in report.values:
            item = QTreeWidgetItem([value.label, f"{value.value:+.3f}", "★" if value.best else ""])
            self._values.addTopLevelItem(item)

    def _show_log(self) -> None:
        self._log_summary.setText(settings_summary(len(self._log), self._log.dropped))
        self._export.setEnabled(len(self._log) > 0)
        self._clear.setEnabled(len(self._log) > 0)
        records = self._log.records[-MAX_ROWS:]
        self._log_table.clear()
        for record in reversed(records):  # newest first
            self._log_table.addTopLevelItem(QTreeWidgetItem(list(_cells(record))))

    def _ask_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the step log", "steps.csv", "CSV files (*.csv)"
        )
        if path:
            self.export_to(Path(path))

    def _on_changed(self, event: object) -> None:
        self.refresh()


def _thinks(var: Variable | None) -> str:
    if var is None:
        return ""
    return f"{var.kind.label} ({SURENESS[var.confidence]})"


def _memory_matches(
    wanted: str, address: str, offset: int, name: str, var: Variable | None
) -> bool:
    if wanted in address.lower() or wanted in name.lower():
        return True
    if var is not None and wanted in var.kind.label.lower():
        return True
    if wanted.startswith("0x"):
        try:
            return int(wanted, 16) == offset
        except ValueError:
            return False
    return wanted.isdigit() and int(wanted) == offset


def _cells(record: StepRecord) -> tuple[str, ...]:
    cells = list(record.row())
    cells[6] = "" if record.value is None else f"{record.value:+.3f}"
    cells[7] = f"{record.reward:+.2f}"
    return tuple(cells)


__all__ = [
    "IDLE_TEXT",
    "LOG_COLUMNS",
    "MAX_MEMORY_ROWS",
    "MAX_ROWS",
    "MEMORY_COLUMNS",
    "NO_SITUATION_TEXT",
    "REFRESH_MS",
    "SETTING_FIELDS",
    "SITUATION_COLUMNS",
    "ResearchDialog",
    "settings_summary",
]
