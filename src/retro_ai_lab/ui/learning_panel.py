"""The Learning panel: start the AI, watch it, pause it, stop it (DESIGN.md §8 to §10).

The child picks a mode (the AI on its own, with the child's help, or the
child playing while the AI watches), whether to watch the AI at game speed
or let it practise fast, and presses Start. A timer steps the training
service while the AI plays; in manual mode the play clock drives instead.
The status line says what the AI is doing and pressing, and the list below
shows how its recent goes went. Everything goes through
:class:`TrainingService`; the panel only draws what it holds.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
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
from retro_ai_lab.learning import algorithm_choices, describe_algorithm
from retro_ai_lab.services import EmulatorConnected, EmulatorDisconnected
from retro_ai_lab.training import (
    EpisodeFinished,
    EpisodeRecord,
    Mode,
    Speed,
    Status,
    TrainingStateChanged,
    TrainingStepped,
)
from retro_ai_lab.training.insights import describe_changes, describe_experiments
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.learning_graph import LearningGraph

IDLE_TEXT = "Connect to a game, then press Start to let the AI learn it."
PIN_TEXT = "Practise from here"
UNPIN_TEXT = "Forget this spot"
PINNED_NOTE = "From now on every go starts here. Press again to forget this spot."
UNPINNED_NOTE = "Spot forgotten: goes start wherever the game is."
SPOT_LINE = "Every go starts from the spot you picked."
COLUMNS = ("Go", "How", "Steps", "Points", "Ended by")
MAX_ROWS = 30
FRAME_MS = 1000 / 60
FAST_SLICE_SECONDS = 0.02  # how long each fast tick may hog the GUI
FAST_TICK_MS = 1
ENDED_BY = {
    "reset": "the game started over",
    "max_steps": "time was up",
    "stuck": "the screen stopped changing",
    "disconnected": "the game went away",
    "stopped": "you pressed Stop",
    "mode change": "you changed the mode",
}


def go_cells(record: EpisodeRecord) -> tuple[str, str, str, str, str]:
    r = record.result
    return (
        str(record.number),
        record.mode.label,
        str(r.steps),
        f"{r.reward:+g}",
        ENDED_BY.get(r.ended_by, r.ended_by),
    )


class LearningPanel(QWidget):
    def __init__(self, context: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("learning_panel")
        self._context = context
        self._training = context.training

        self._mode = QComboBox(self)
        self._mode.setObjectName("mode_combo")
        for m in Mode:
            self._mode.addItem(m.label, m.value)
            self._mode.setItemData(self._mode.count() - 1, m.description, 3)  # tooltip role
        self._mode.activated.connect(self._on_mode_activated)
        self._speed = QComboBox(self)
        self._speed.setObjectName("speed_combo")
        for sp in Speed:
            self._speed.addItem(sp.label, sp.value)
        self._speed.activated.connect(self._on_speed_activated)
        self._algorithm = QComboBox(self)
        self._algorithm.setObjectName("algorithm_combo")
        for name in algorithm_choices():
            label, text = describe_algorithm(name)
            self._algorithm.addItem(label, name)
            self._algorithm.setItemData(self._algorithm.count() - 1, text, 3)  # tooltip role
        self._algorithm.activated.connect(self._on_algorithm_activated)

        self._start = QPushButton("Start", self)
        self._start.setObjectName("start_button")
        self._pause = QPushButton("Pause", self)
        self._pause.setObjectName("pause_button")
        self._stop = QPushButton("Stop", self)
        self._stop.setObjectName("stop_button")
        self._forget = QPushButton("Start from scratch", self)
        self._forget.setObjectName("forget_button")
        self._forget.setToolTip("Forget everything the AI has learned about this game.")
        self._start.clicked.connect(self.start)
        self._pause.clicked.connect(self.pause_or_resume)
        self._stop.clicked.connect(self.stop)
        self._forget.clicked.connect(self.forget)
        self._rehearse = QPushButton("Learn from my moves again", self)
        self._rehearse.setObjectName("rehearse_button")
        self._rehearse.setToolTip("The AI goes over every move you showed it one more time.")
        self._rehearse.clicked.connect(self.rehearse)
        self._spot = QPushButton(PIN_TEXT, self)
        self._spot.setObjectName("spot_button")
        self._spot.setToolTip("Practise this part of the game: every go starts from here.")
        self._spot.clicked.connect(self.toggle_spot)

        self._status = QLabel(IDLE_TEXT, self)
        self._status.setObjectName("learning_status")
        self._status.setWordWrap(True)

        self._graph = LearningGraph(self)
        self._insights = QLabel("", self)
        self._insights.setObjectName("learning_insights")
        self._insights.setWordWrap(True)

        self._tree = QTreeWidget(self)
        self._tree.setObjectName("goes_tree")
        self._tree.setColumnCount(len(COLUMNS))
        self._tree.setHeaderLabels(list(COLUMNS))
        self._tree.header().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._tree.setRootIsDecorated(False)

        self._note = QLabel("", self)
        self._note.setObjectName("learning_note")
        self._note.setWordWrap(True)

        pickers = QHBoxLayout()
        pickers.addWidget(QLabel("Who plays", self))
        pickers.addWidget(self._mode, 1)
        pickers.addWidget(QLabel("Speed", self))
        pickers.addWidget(self._speed, 1)
        pickers.addWidget(QLabel("How the AI learns", self))
        pickers.addWidget(self._algorithm, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self._start)
        buttons.addWidget(self._pause)
        buttons.addWidget(self._stop)
        buttons.addWidget(self._spot)
        buttons.addStretch(1)
        buttons.addWidget(self._rehearse)
        buttons.addWidget(self._forget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(pickers)
        layout.addLayout(buttons)
        layout.addWidget(self._status)
        layout.addWidget(self._graph, 2)
        layout.addWidget(self._insights)
        layout.addWidget(self._tree, 1)
        layout.addWidget(self._note)

        self._ticker = QTimer(self)
        self._ticker.setObjectName("learning_ticker")
        self._ticker.timeout.connect(self.tick)

        bridge = context.bridge
        unsubscribe = (
            bridge.subscribe(TrainingStateChanged, self._on_state_changed),
            bridge.subscribe(EpisodeFinished, self._on_episode_finished),
            bridge.subscribe(TrainingStepped, self._on_stepped),
            bridge.subscribe(EmulatorConnected, self._on_connection_changed),
            bridge.subscribe(EmulatorDisconnected, self._on_connection_changed),
        )
        ticker = self._ticker

        def release(*_: object) -> None:
            ticker.stop()
            for off in unsubscribe:
                off()

        self.destroyed.connect(release)
        self.refresh()

    # --------------------------------------------------------------- API

    @property
    def status_text(self) -> str:
        return self._status.text()

    @property
    def note_text(self) -> str:
        return self._note.text()

    @property
    def mode(self) -> Mode:
        return Mode(self._mode.currentData())

    @property
    def speed(self) -> Speed:
        return Speed(self._speed.currentData())

    @property
    def algorithm(self) -> str:
        return str(self._algorithm.currentData())

    @property
    def graph(self) -> LearningGraph:
        return self._graph

    @property
    def insights_text(self) -> str:
        return self._insights.text()

    @property
    def ticking(self) -> bool:
        return self._ticker.isActive()

    @property
    def tick_interval_ms(self) -> int:
        return self._ticker.interval()

    def go_rows(self) -> tuple[tuple[str, ...], ...]:
        out = []
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item is not None:
                out.append(tuple(item.text(c) for c in range(len(COLUMNS))))
        return tuple(out)

    def choose_mode(self, mode: Mode) -> None:
        self._call(lambda: self._training.set_mode(mode))
        self.refresh()

    def choose_speed(self, speed: Speed) -> None:
        self._call(lambda: self._training.set_speed(speed))
        self.refresh()

    def choose_algorithm(self, algorithm: str) -> None:
        note = ""

        def change() -> None:
            nonlocal note
            note = self._training.set_algorithm(algorithm)

        if self._call(change):
            self._note.setText(note)
        self.refresh()

    def start(self) -> None:
        if self._call(lambda: self._training.start(self.mode)):
            self._note.setText("")
        self.refresh()

    def pause_or_resume(self) -> None:
        if self._training.status is Status.PAUSED:
            self._call(self._training.resume)
        else:
            self._call(self._training.pause)
        self.refresh()

    def stop(self) -> None:
        self._call(self._training.stop)
        self.refresh()

    def toggle_spot(self) -> None:
        """Pin the game's position as the spot to practise, or forget the pinned one."""
        t = self._training
        if t.spot_pinned:
            if self._call(t.unpin_spot):
                self._note.setText(UNPINNED_NOTE)
        elif self._call(t.pin_spot):
            self._note.setText(PINNED_NOTE)
        self.refresh()

    def forget(self) -> None:
        if self._call(self._training.forget):
            self._note.setText("The AI has forgotten this game and will start fresh.")
        self.refresh()

    def rehearse(self) -> None:
        steps = 0

        def go() -> None:
            nonlocal steps
            steps = self._training.rehearse()

        if self._call(go):
            self._note.setText(
                f"The AI learned from your {steps} move{'s' if steps != 1 else ''} again."
            )
        self.refresh()

    def tick(self) -> int:
        """One timer tick: a step (LIVE) or a slice of steps (FAST). Returns the steps."""
        if self._training.status is not Status.RUNNING or not self._training.mode.ai_plays:
            self._arm_ticker()
            return 0
        if self._training.speed is Speed.FAST:
            taken = self._training.run_for(FAST_SLICE_SECONDS)
        else:
            taken = 1 if self._training.step() else 0
        self._show_status()
        return taken

    def refresh(self) -> None:
        t = self._training
        enabled = t.enabled
        idle = t.status is Status.IDLE
        self._set_combo(self._mode, t.mode.value)
        self._set_combo(self._speed, t.speed.value)
        self._set_combo(self._algorithm, t.algorithm)
        self._mode.setEnabled(enabled)
        self._speed.setEnabled(enabled)
        self._algorithm.setEnabled(enabled and idle)
        self._start.setEnabled(enabled and idle)
        self._pause.setEnabled(enabled and not idle)
        self._pause.setText("Resume" if t.status is Status.PAUSED else "Pause")
        self._stop.setEnabled(enabled and not idle)
        self._forget.setEnabled(enabled and idle)
        self._rehearse.setEnabled(enabled and idle and t.demonstrations.count > 0)
        self._spot.setText(UNPIN_TEXT if t.spot_pinned else PIN_TEXT)
        harness = t.harness
        self._spot.setEnabled(enabled and idle and harness is not None and harness.available())
        self._show_status()
        self._show_goes()
        self._arm_ticker()

    # ----------------------------------------------------------- internals

    def _call(self, action: object) -> bool:
        try:
            action()  # type: ignore[operator]
        except (UnsupportedCapabilityError, ConfigurationError) as exc:
            self._note.setText(str(exc))
            return False
        return True

    def _set_combo(self, combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0 and index != combo.currentIndex():
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def _arm_ticker(self) -> None:
        t = self._training
        want = t.enabled and t.status is Status.RUNNING and t.mode.ai_plays
        if not want:
            self._ticker.stop()
            return
        interval = FAST_TICK_MS if t.speed is Speed.FAST else int(FRAME_MS * t.frames_per_step)
        if self._ticker.interval() != interval:
            self._ticker.setInterval(max(1, interval))
        if not self._ticker.isActive():
            self._ticker.start()

    def _show_status(self) -> None:
        t = self._training
        if not t.enabled:
            self._status.setText(IDLE_TEXT)
            return
        lines = [t.summary()]
        episode = t.episode
        if episode is not None and t.mode.ai_plays and episode.steps > 0:
            pressed = " + ".join(sorted(episode.last_buttons)) or "nothing"
            lines.append(f"Pressing: {pressed}.")
        if t.agent is not None:
            lines.append(
                f"Learning with the {describe_algorithm(t.agent.algorithm)[0].lower()}. "
                + t.agent.stats()
            )
        shown = t.demonstrations.shown
        if shown and t.mode is not Mode.MANUAL:
            lines.append(f"You have shown the AI {shown} move{'s' if shown != 1 else ''}.")
        if t.fresh_start:
            lines.append(t.fresh_start)
        if t.spot_pinned:
            lines.append(SPOT_LINE)
        self._status.setText("\n".join(lines))

    def _show_insights(self) -> None:
        t = self._training
        if not t.enabled or t.space is None:
            self._insights.setText("")
            return
        changes, goes_ago, known = t.behaviour_changes()
        knowledge = self._context.discovery.knowledge
        rewards = self._context.rewards
        current = rewards.current_warning
        experiments = t.experiments(
            current.warning if current is not None else None,
            self._context.goals.goal,
            rewards.profile,
        )
        self._insights.setText(
            describe_changes(changes, known, goes_ago, knowledge, t.space)
            + "\n"
            + describe_experiments(experiments)
        )

    def _show_goes(self) -> None:
        self._graph.set_curve(self._training.curve())
        self._show_insights()
        self._tree.clear()
        for record in reversed(self._training.history[-MAX_ROWS:]):
            self._tree.addTopLevelItem(QTreeWidgetItem(list(go_cells(record))))

    def _on_mode_activated(self, index: int) -> None:
        self.choose_mode(Mode(self._mode.itemData(index)))

    def _on_speed_activated(self, index: int) -> None:
        self.choose_speed(Speed(self._speed.itemData(index)))

    def _on_algorithm_activated(self, index: int) -> None:
        self.choose_algorithm(str(self._algorithm.itemData(index)))

    def _on_state_changed(self, event: TrainingStateChanged) -> None:
        if event.reason == "disconnected":
            self._note.setText("The game went away, so training stopped.")
        elif event.reason == "promoted":
            self._note.setText(self._training.last_change)
        self.refresh()

    def _on_stepped(self, event: TrainingStepped) -> None:
        if not self._training.mode.ai_plays:  # the child's steps don't come through tick()
            self._show_status()

    def _on_episode_finished(self, event: EpisodeFinished) -> None:
        self._show_goes()
        self._show_status()

    def _on_connection_changed(self, event: object) -> None:
        if isinstance(event, EmulatorConnected):
            self._note.setText("")
        self.refresh()


__all__ = [
    "COLUMNS",
    "ENDED_BY",
    "IDLE_TEXT",
    "MAX_ROWS",
    "PINNED_NOTE",
    "PIN_TEXT",
    "SPOT_LINE",
    "UNPINNED_NOTE",
    "UNPIN_TEXT",
    "LearningPanel",
    "go_cells",
]
