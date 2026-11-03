"""T-409: the Goals panel."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.services import DiscoveryUpdated
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.goals_panel import BY_HAND, IDLE_TEXT, GoalsPanel
from retro_ai_lab.ui.main_window import MainWindow
from retro_ai_lab.understanding import EventKind, Evidence, GameEvent, Knowledge, VariableKind

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]


@pytest.fixture
def context(tmp_path) -> AppContext:  # type: ignore[no-untyped-def]
    return AppContext(data_dir=tmp_path)


@pytest.fixture
def panel(qtbot: QtBot, context: AppContext) -> GoalsPanel:
    p = GoalsPanel(context)
    qtbot.addWidget(p)
    p.show()
    return p


def teach(context: AppContext) -> Knowledge:
    k = context.discovery.knowledge
    assert k is not None
    added = (
        k.propose(RAM, LAYOUT.lives, kind=VariableKind.LIVES, name="Lives", evidence=SURE),
        k.propose(
            RAM, LAYOUT.checkpoint, kind=VariableKind.PROGRESS, name="Checkpoint", evidence=SURE
        ),
        k.propose(RAM, LAYOUT.lap, kind=VariableKind.PROGRESS, name="Lap", evidence=SURE),
    )
    context.bus.publish(DiscoveryUpdated(added, (), (), ()))
    return k


def widget(panel: GoalsPanel, name: str) -> object:
    return panel.findChild(object, name)


def ids(panel: GoalsPanel) -> tuple[str, str]:
    a, b = panel.objective_ids()[:2]
    return a, b


def test_idle_panel(panel: GoalsPanel) -> None:
    assert panel.summary_text == IDLE_TEXT and panel.title_text == ""
    assert panel.objective_ids() == () and panel.trigger_keys() == (None, "reset:*")
    for name in ("add_button", "remove_button", "done_button", "reset_button", "goal_title"):
        assert not widget(panel, name).isEnabled(), name  # type: ignore[attr-defined]


def test_steps_appear_with_detection_status_and_badges(
    panel: GoalsPanel, context: AppContext
) -> None:
    context.connection.connect("fake", seed=2)
    assert panel.title_text == "Get as far as you can in Toy Racer"
    assert panel.summary_text.endswith("no steps yet.")
    teach(context)
    cp, lap = ids(panel)
    assert panel.row_text(cp) == (
        "Get further in Checkpoint",
        "+25",
        "Checkpoint goes up (Pretty sure)",
        "Ready",
        "on",
    )
    assert panel.row_text(lap)[3] == "Waiting for: Get further in Checkpoint"
    assert panel.summary_text.endswith("0 of 2 steps done. Next: Get further in Checkpoint.")
    assert "Lap goes up" in [
        widget(panel, "trigger_combo").itemText(i)  # type: ignore[attr-defined]
        for i in range(widget(panel, "trigger_combo").count())  # type: ignore[attr-defined]
    ]
    panel.select(cp)
    k = context.discovery.knowledge
    assert k is not None
    checkpoint = k.at(RAM, LAYOUT.checkpoint)
    assert checkpoint
    context.bus.publish(
        DiscoveryUpdated((), (), (GameEvent(EventKind.PROGRESS, 9, checkpoint.id, ""),), ())
    )
    assert panel.selected_id == cp  # kept across the update
    assert panel.row_text(cp)[3] == "Done" and panel.row_text(lap)[3] == "Ready"
    assert panel.note_text == "Done: Get further in Checkpoint!"
    assert widget(panel, "done_button").text() == "Not done"  # type: ignore[attr-defined]


def test_add_remove_move_and_done(panel: GoalsPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    teach(context)
    panel.add_step("   ", None)
    assert "description" in panel.note_text and len(panel.objective_ids()) == 2
    panel.add_step("Beat my record", None)
    assert panel.note_text == "Step added." and len(panel.objective_ids()) == 3
    mine = panel.objective_ids()[-1]
    assert panel.row_text(mine) == ("Beat my record", "+25", BY_HAND, "Ready", "on")
    lap_key = next(k for k in panel.trigger_keys() if k and k.startswith("progress:"))
    panel.add_step("Another lap", lap_key)
    other = panel.objective_ids()[-1]
    assert panel.row_text(other)[2].startswith("Checkpoint goes up")
    panel.add_step("Mystery", "flag_on:ghost")
    assert "doesn't know" in panel.note_text
    panel.select(other)
    panel.move_selected(-1)
    assert panel.objective_ids().index(other) == 2 and panel.selected_id == other
    panel.move_selected(-1)
    panel.move_selected(-1)
    panel.move_selected(-1)  # already at the top: no-op
    assert panel.objective_ids()[0] == other
    panel.move_selected(1)
    assert panel.objective_ids()[1] == other
    panel.toggle_done_selected()
    assert panel.row_text(other)[3] == "Done" and panel.note_text == "Done!"
    panel.toggle_done_selected()
    assert panel.row_text(other)[3] == "Ready"
    panel.remove_selected()
    assert other not in panel.objective_ids() and panel.selected_id is None
    assert context.goals.store is not None and context.goals.store.exists("toyracer-2")


def test_after_tick_reset_and_title(panel: GoalsPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    teach(context)
    cp, lap = ids(panel)
    panel.select(lap)
    assert panel.after_choices() == (None, cp)
    combo = widget(panel, "after_combo")
    assert combo.currentData() == cp  # type: ignore[attr-defined]
    panel.set_selected_after(None)
    assert panel.row_text(lap)[3] == "Ready" and panel.note_text == "Order remembered."
    panel.select(cp)
    panel.set_selected_after(lap)
    assert panel.row_text(cp)[3] == "Waiting for: Get further in Lap"
    panel.select(lap)
    combo.setCurrentIndex(combo.findData(cp))  # type: ignore[attr-defined]
    combo.activated.emit(combo.currentIndex())  # type: ignore[attr-defined]
    assert "depend on each other" in panel.note_text
    assert panel.row_text(lap)[3] == "Ready"
    panel.set_selected_enabled(False)
    assert panel.row_text(lap)[3:] == ("Off", "off")
    item = panel._items[lap]
    item.setCheckState(4, Qt.CheckState.Checked)
    assert panel.note_text == "Switched on."  # the service heard it at once
    QApplication.processEvents()  # the tree is rebuilt a tick later, never inside the signal
    assert panel.row_text(lap)[4] == "on" and panel._items[lap] is not item
    panel.toggle_done_selected()
    panel.reset_progress()
    assert panel.row_text(lap)[3] == "Ready" and "Starting again" in panel.note_text
    title = widget(panel, "goal_title")
    title.setText("Win the cup")  # type: ignore[attr-defined]
    title.returnPressed.emit()  # type: ignore[attr-defined]
    goal = panel.goal
    assert goal is not None and goal.title == "Win the cup"
    assert panel.summary_text.startswith("Win the cup:")
    panel.set_title(" ")
    assert "needs a name" in panel.note_text


def test_disconnect_keeps_rows_but_disables(panel: GoalsPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    teach(context)
    panel.select(panel.objective_ids()[0])
    context.connection.disconnect()
    assert panel.objective_ids() != () and panel.summary_text == IDLE_TEXT
    for name in ("add_button", "remove_button", "done_button", "reset_button", "goal_title"):
        assert not widget(panel, name).isEnabled(), name  # type: ignore[attr-defined]


def test_main_window_hosts_the_panel(qtbot: QtBot, context: AppContext) -> None:
    window = MainWindow(context)
    qtbot.addWidget(window)
    assert isinstance(window.goals_panel, GoalsPanel)
    assert window.goals_panel.parent() is window.panel("progress").content
