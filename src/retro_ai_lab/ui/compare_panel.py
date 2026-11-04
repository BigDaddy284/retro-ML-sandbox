"""The "Compare AIs" panel: save the AI by name, compare, load, delete (DESIGN.md §11)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from retro_ai_lab.comparison import (
    MEASURES,
    AiComparison,
    BehaviourDifference,
    Challenge,
    CompetitionFinished,
    SavedAi,
    SavedAiChanged,
    Scorecard,
    behaviour_differences,
    describe,
)
from retro_ai_lab.core import ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.services import EmulatorConnected, EmulatorDisconnected
from retro_ai_lab.training import LearningCurve, Status, TrainingStateChanged
from retro_ai_lab.training.records import EpisodeRecord
from retro_ai_lab.ui.comparison_graph import ComparisonGraph
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.panels import name_column

IDLE_TEXT = "Connect to a game the AI can learn, then save your AI here by name."
NO_SAVED_TEXT = "No saved AIs for this game yet. Train one, give it a name and press Save."
LEADER_MARK = "★ "
COLUMNS: tuple[str, ...] = ("AI", "Personality", *(m.label for m in MEASURES))


def row_cells(card: Scorecard, leaders: dict[str, Scorecard | None]) -> tuple[str, ...]:
    cells = [card.name, card.personality or "custom"]
    for m in MEASURES:
        value = card.value(m.key)
        if value is None:
            text = "-"
        elif m.key in ("average_reward", "best_reward"):
            text = f"{value:+.1f}"
        elif m.key == "successes":
            text = f"{card.successes} of {card.attempts}"
        elif m.key == "average_steps":
            text = f"{value:.0f}"
        else:
            text = f"{value:g}"
        leader = leaders.get(m.key)
        if leader is not None and leader.name == card.name:
            text = LEADER_MARK + text
        cells.append(text)
    return tuple(cells)


class ComparePanel(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("compare_panel")
        self._context = context
        self._saved: dict[str, SavedAi] = {}

        self._name = QLineEdit(self)
        self._name.setObjectName("ai_name_edit")
        self._name.setPlaceholderText("A name for this AI, like AI A")
        self._name.returnPressed.connect(self._on_save_clicked)
        self._save = QPushButton("Save this AI as", self)
        self._save.setObjectName("save_ai_button")
        self._save.clicked.connect(self._on_save_clicked)

        self._table = QTreeWidget(self)
        self._table.setObjectName("comparison_table")
        self._table.setColumnCount(len(COLUMNS))
        self._table.setHeaderLabels(list(COLUMNS))
        name_column(self._table)
        self._table.setRootIsDecorated(False)
        self._table.currentItemChanged.connect(lambda *_: self._update_buttons())

        self._load = QPushButton("Load", self)
        self._load.setObjectName("load_ai_button")
        self._load.clicked.connect(self.load_selected)
        self._delete = QPushButton("Delete", self)
        self._delete.setObjectName("delete_ai_button")
        self._delete.clicked.connect(self.delete_selected)
        self._share = QPushButton("Share…", self)
        self._share.setObjectName("share_ai_button")
        self._share.clicked.connect(self._on_share_clicked)
        self._import = QPushButton("Load a shared AI…", self)
        self._import.setObjectName("import_ai_button")
        self._import.clicked.connect(self._on_import_clicked)
        self._race = QPushButton("Race them!", self)
        self._race.setObjectName("race_button")
        self._race.setToolTip(Challenge().describe())
        self._race.clicked.connect(self.race)
        self._scoreboard = QLabel("", self)
        self._scoreboard.setObjectName("scoreboard_text")
        self._scoreboard.setWordWrap(True)

        self._text = QLabel(IDLE_TEXT, self)
        self._text.setObjectName("comparison_text")
        self._text.setWordWrap(True)
        self._graph = ComparisonGraph(self)
        self._note = QLabel("", self)
        self._note.setObjectName("compare_note")
        self._note.setWordWrap(True)

        save_row = QHBoxLayout()
        save_row.addWidget(self._name, 1)
        save_row.addWidget(self._save)
        buttons = QHBoxLayout()
        buttons.addWidget(self._load)
        buttons.addWidget(self._delete)
        buttons.addWidget(self._share)
        buttons.addWidget(self._race)
        buttons.addStretch(1)
        buttons.addWidget(self._import)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(save_row)
        layout.addWidget(self._table, 2)
        layout.addLayout(buttons)
        layout.addWidget(self._text)
        layout.addWidget(self._scoreboard)
        layout.addWidget(self._graph, 1)
        layout.addWidget(self._note)

        bridge = context.bridge
        unsubscribe = (
            bridge.subscribe(SavedAiChanged, self._on_changed),
            bridge.subscribe(CompetitionFinished, self._on_competition),
            bridge.subscribe(EmulatorConnected, self._on_connection),
            bridge.subscribe(EmulatorDisconnected, self._on_connection),
            bridge.subscribe(TrainingStateChanged, lambda _e: self._update_buttons()),
        )
        self.destroyed.connect(lambda *_: [u() for u in unsubscribe])
        self.refresh()

    # --------------------------------------------------------------- API

    @property
    def name_text(self) -> str:
        return self._name.text()

    def set_name(self, name: str) -> None:
        self._name.setText(name)

    @property
    def note_text(self) -> str:
        return self._note.text()

    @property
    def comparison_text(self) -> str:
        return self._text.text()

    @property
    def scoreboard_text(self) -> str:
        return self._scoreboard.text()

    def race(self, challenge: Challenge | None = None) -> bool:
        """Race every saved AI of this game under ``challenge`` (the default: 3 goes)."""
        names = self.saved_names()
        self._note.setText("Racing…")
        try:
            board = self._context.saved_ais.compete(names, challenge)
        except (ConfigurationError, UnsupportedCapabilityError) as exc:
            self._note.setText(str(exc))
            return False
        winner = board.winner()
        self._note.setText(
            f"{winner.name} won the race." if winner is not None else "The race was a tie."
        )
        return True

    @property
    def graph(self) -> ComparisonGraph:
        return self._graph

    def saved_names(self) -> tuple[str, ...]:
        return tuple(self._saved)

    def table_rows(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(item.text(c) for c in range(len(COLUMNS))) for item in self._items())

    @property
    def selected_name(self) -> str | None:
        item = self._table.currentItem()
        return item.text(0) if item is not None else None

    def select(self, name: str | None) -> None:
        if name is None:
            self._table.setCurrentItem(None)  # type: ignore[call-overload]
            return
        for item in self._items():
            if item.text(0) == name:
                self._table.setCurrentItem(item)
                return
        raise ValueError(f"No saved AI {name!r} in the table.")

    def save_now(self, name: str | None = None) -> bool:
        """Save the connected AI under ``name`` (or the name box). True on success."""
        text = self._name.text() if name is None else name
        try:
            saved = self._context.saved_ais.save_as(text)
        except (ConfigurationError, UnsupportedCapabilityError) as exc:
            self._note.setText(str(exc))
            return False
        self._name.setText("")
        self._note.setText(f"Saved as {saved.name}.")
        self.select(saved.name)
        return True

    def load_selected(self) -> bool:
        name = self.selected_name
        if name is None:
            self._note.setText("Pick a saved AI to load.")
            return False
        try:
            saved = self._context.saved_ais.load(name)
        except (ConfigurationError, UnsupportedCapabilityError) as exc:
            self._note.setText(str(exc))
            return False
        self._note.setText(f"Loaded {saved.name}: its rewards, goal and brain are the AI's now.")
        return True

    def delete_selected(self) -> bool:
        name = self.selected_name
        if name is None:
            self._note.setText("Pick a saved AI to delete.")
            return False
        try:
            removed = self._context.saved_ais.delete(name)
        except (ConfigurationError, UnsupportedCapabilityError) as exc:
            self._note.setText(str(exc))
            return False
        self._note.setText(f"Deleted {name}." if removed else f"{name} was already gone.")
        return removed

    def export_selected_to(self, path: str | Path) -> bool:
        """Write the selected saved AI to ``path`` for a friend."""
        name = self.selected_name
        if name is None:
            self._note.setText("Pick a saved AI to share.")
            return False
        try:
            written = self._context.saved_ais.export_to(name, path)
        except (ConfigurationError, UnsupportedCapabilityError) as exc:
            self._note.setText(str(exc))
            return False
        self._note.setText(f"Shared {name} as {written.name}.")
        return True

    def import_from(self, path: str | Path) -> bool:
        """Keep a friend's shared AI for this game."""
        try:
            saved = self._context.saved_ais.import_from(path)
        except (ConfigurationError, UnsupportedCapabilityError) as exc:
            self._note.setText(str(exc))
            return False
        self._note.setText(f"Loaded the shared AI as {saved.name}.")
        self.select(saved.name)
        return True

    def refresh(self) -> None:
        service = self._context.saved_ais
        keep = self.selected_name
        self._saved = {}
        if service.enabled:
            for name in service.names():
                try:
                    self._saved[name] = service.get(name)
                except ConfigurationError:
                    continue  # a damaged file is left out rather than breaking the panel
        self._fill_table()
        self._show_comparison(service.enabled)
        if keep in self._saved:
            self.select(keep)
        self._update_buttons()

    # ----------------------------------------------------------- internals

    def _items(self) -> tuple[QTreeWidgetItem, ...]:
        items = (self._table.topLevelItem(i) for i in range(self._table.topLevelItemCount()))
        return tuple(item for item in items if item is not None)

    def _fill_table(self) -> None:
        self._table.clear()
        comparison = AiComparison.of(tuple(self._saved.values()))
        leaders = comparison.leaders()
        for card in comparison.cards:
            self._table.addTopLevelItem(QTreeWidgetItem(list(row_cells(card, leaders))))

    def _show_comparison(self, enabled: bool) -> None:
        saved = tuple(self._saved.values())
        if not enabled:
            self._text.setText(IDLE_TEXT)
            self._graph.set_curves(())
            return
        if not saved:
            self._text.setText(NO_SAVED_TEXT)
            self._graph.set_curves(())
            return
        differences: tuple[BehaviourDifference, ...] = ()
        if len(saved) == 2:
            try:
                differences = behaviour_differences(saved[0], saved[1])
            except ConfigurationError:
                differences = ()
        self._text.setText(
            describe(
                AiComparison.of(saved),
                differences=differences,
                space=self._context.training.space,
                knowledge=self._context.discovery.knowledge,
            )
        )
        self._graph.set_curves([(s.name, _curve_of(s)) for s in saved])

    def _update_buttons(self) -> None:
        enabled = self._context.saved_ais.enabled
        chosen = self.selected_name is not None
        self._name.setEnabled(enabled)
        self._save.setEnabled(enabled)
        self._load.setEnabled(enabled and chosen)
        self._delete.setEnabled(enabled and chosen)
        self._share.setEnabled(enabled and chosen)
        self._import.setEnabled(enabled)
        idle = self._context.training.status is Status.IDLE
        self._race.setEnabled(enabled and idle and len(self._saved) >= 2)

    def _on_save_clicked(self) -> None:
        self.save_now()

    def _on_share_clicked(self) -> None:
        name = self.selected_name
        if name is None:
            self._note.setText("Pick a saved AI to share.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Share this AI", f"{name}-ai.json", "Shared AI files (*.json)"
        )
        if path:
            self.export_selected_to(path)

    def _on_import_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load a shared AI", "", "Shared AI files (*.json)"
        )
        if path:
            self.import_from(path)

    def _on_changed(self, event: SavedAiChanged) -> None:
        self.refresh()

    def _on_connection(self, event: object) -> None:
        self._note.setText("")
        self._scoreboard.setText("")
        self.refresh()

    def _on_competition(self, event: CompetitionFinished) -> None:
        self._scoreboard.setText(event.scoreboard.describe())


def _curve_of(saved: SavedAi) -> LearningCurve:
    raw = saved.brain.get("goes", [])
    goes = []
    if isinstance(raw, list):
        for item in raw:
            try:
                goes.append(EpisodeRecord.from_dict(item))
            except ConfigurationError:
                continue
    return LearningCurve(goes)


__all__ = ["COLUMNS", "IDLE_TEXT", "LEADER_MARK", "NO_SAVED_TEXT", "ComparePanel", "row_cells"]
