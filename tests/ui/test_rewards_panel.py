"""T-406: the Rewards panel."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import Capabilities
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.rewards import ExploitWarning
from retro_ai_lab.services import DiscoveryUpdated, ExploitSuspected
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.main_window import MainWindow
from retro_ai_lab.ui.rewards_panel import IDLE_TEXT, YOU_SET_THIS, RewardsPanel
from retro_ai_lab.understanding import (
    Encoding,
    EventKind,
    Evidence,
    GameEvent,
    Knowledge,
    VariableKind,
)

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]


@pytest.fixture
def context(tmp_path) -> AppContext:  # type: ignore[no-untyped-def]
    return AppContext(data_dir=tmp_path)


@pytest.fixture
def panel(qtbot: QtBot, context: AppContext) -> RewardsPanel:
    p = RewardsPanel(context)
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
        k.propose(
            RAM,
            LAYOUT.score,
            encoding=Encoding.U16LE,
            kind=VariableKind.SCORE,
            name="Score",
            evidence=SURE,
        ),
    )
    context.bus.publish(DiscoveryUpdated(added, (), (), ()))
    return k


def button(panel: RewardsPanel, name: str) -> object:
    return panel.findChild(object, name)


def test_idle_panel(panel: RewardsPanel) -> None:
    assert panel.summary_text == IDLE_TEXT
    assert panel.behaviour_ids() == () and panel.category_texts() == ()
    assert panel.selected_id is None and panel.warning_text == ""
    for name in ("plus_button", "minus_button", "suggested_button", "personality_combo"):
        assert not button(panel, name).isEnabled(), name  # type: ignore[attr-defined]
    assert panel.personality_id == "custom"


def test_rows_appear_under_their_categories(panel: RewardsPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    assert panel.behaviour_ids() == ("reset:*",)
    assert panel.category_texts() == ("Survival (1)",)
    assert panel.summary_text == "1 reward switched on; +0.0 earned."
    panel.select("reset:*")
    teach(context)
    panel.refresher.flush()  # suggestions may wait for the refresh window (T-843)
    assert panel.selected_id == "reset:*"  # kept across the update
    assert panel.category_texts() == ("Collection (1)", "Survival (3)", "Progression (1)")
    k = context.discovery.knowledge
    assert k is not None
    lives = k.at(RAM, LAYOUT.lives)
    assert lives
    assert panel.row_text(f"setback:{lives.id}") == ("Losing Lives", "-10", "-10", "on")
    assert panel.row_text("reset:*") == ("Starting over", "-5", "-5", "on")
    assert button(panel, "personality_combo").isEnabled()  # type: ignore[attr-defined]


def test_plus_minus_suggested_and_tick(panel: RewardsPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    k = teach(context)
    panel.refresher.flush()
    checkpoint = k.at(RAM, LAYOUT.checkpoint)
    assert checkpoint
    bid = f"progress:{checkpoint.id}"
    panel.increase_selected()  # nothing selected: no-op
    panel.select(bid)
    assert button(panel, "plus_button").isEnabled()  # type: ignore[attr-defined]
    panel.increase_selected()
    assert panel.row_text(bid)[1] == f"+11  ({YOU_SET_THIS})"
    assert panel.note_text == "Remembered for next time."
    panel.decrease_selected()
    panel.decrease_selected()
    assert panel.row_text(bid)[1] == f"+9  ({YOU_SET_THIS})"
    panel.use_suggested_selected()
    assert panel.row_text(bid)[1] == "+10"
    assert panel.selected_id == bid
    panel.set_selected_enabled(False)
    assert panel.row_text(bid)[3] == "off"
    profile = panel.profile
    assert profile is not None and not profile.get(bid).enabled
    # The tick box itself drives the service too.
    item = panel._items[bid]
    item.setCheckState(3, Qt.CheckState.Checked)
    assert profile.get(bid).enabled  # the service heard it at once
    QApplication.processEvents()  # the tree is rebuilt a tick later, never inside the signal
    assert panel.row_text(bid)[3] == "on"
    assert panel._items[bid] is not item  # rebuilt, and nothing exploded
    assert context.rewards.store is not None and context.rewards.store.exists("toyracer-2")


def test_personality_picker(panel: RewardsPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    k = teach(context)
    lives = k.at(RAM, LAYOUT.lives)
    assert lives
    panel.choose_personality("survivor")
    assert panel.personality_id == "survivor"
    assert panel.row_text(f"setback:{lives.id}")[1] == "-20"
    assert panel.note_text.startswith("Survivor:")
    panel.select(f"setback:{lives.id}")
    panel.increase_selected()
    assert panel.personality_id == "survivor"  # tweaking keeps the preset's name
    combo = button(panel, "personality_combo")
    combo.setCurrentIndex(combo.findData("explorer"))  # type: ignore[attr-defined]
    combo.activated.emit(combo.currentIndex())  # type: ignore[attr-defined]
    assert panel.personality_id == "explorer"
    assert panel.row_text(f"setback:{lives.id}")[1] == f"-19  ({YOU_SET_THIS})"
    panel.choose_personality("ghost")
    assert "No personality" in panel.note_text


def test_total_and_warning(panel: RewardsPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    k = teach(context)
    score = k.at(RAM, LAYOUT.score)
    assert score
    context.bus.publish(
        DiscoveryUpdated((), (), (GameEvent(EventKind.SCORE_GAIN, 5, score.id, ""),), ())
    )
    assert panel.summary_text == "5 rewards switched on; +5.0 earned."
    warning = ExploitWarning("farming", (f"score_gain:{score.id}",), 1.0, 100, "Why.", "Do this.")
    context.rewards._current = ExploitSuspected(5, warning)  # as the service would set it
    context.bus.publish(ExploitSuspected(5, warning))
    assert panel.warning_text == "Hmm. Why.\nDo this."
    context.rewards._current = None
    context.bus.publish(
        DiscoveryUpdated((), (), (GameEvent(EventKind.SCORE_GAIN, 500, score.id, ""),), ())
    )
    assert panel.warning_text == ""


def test_disconnect_keeps_rows_but_disables_controls(
    panel: RewardsPanel, context: AppContext
) -> None:
    context.connection.connect("fake", seed=2)
    teach(context)
    panel.select("reset:*")
    context.connection.disconnect()
    assert panel.behaviour_ids() != () and panel.summary_text == IDLE_TEXT
    assert not button(panel, "plus_button").isEnabled()  # type: ignore[attr-defined]
    assert not button(panel, "personality_combo").isEnabled()  # type: ignore[attr-defined]
    context.connection.connect(
        "fake", capabilities=Capabilities(screen_capture=True, frame_advance=True)
    )
    assert panel.behaviour_ids() == ()


def test_main_window_hosts_the_panel(qtbot: QtBot, context: AppContext) -> None:
    window = MainWindow(context)
    qtbot.addWidget(window)
    assert isinstance(window.rewards_panel, RewardsPanel)
    assert window.rewards_panel.parent() is window.panel("rewards").content


def test_many_profile_changes_rebuild_the_tree_once(
    panel: RewardsPanel, context: AppContext, qtbot: QtBot
) -> None:
    """T-843: suggestions change on many training steps; the tree follows a few times a second."""
    context.connection.connect("fake", seed=3)
    k = teach(context)
    rebuilds: list[int] = []
    original = panel.refresh

    def counted(*, keep: str | None = None) -> None:
        rebuilds.append(1)
        original(keep=keep)

    panel.refresh = counted  # type: ignore[method-assign]
    panel.refresher.flush()
    qtbot.waitUntil(lambda: not panel.refresher.pending, timeout=2000)
    qtbot.wait(300)  # out of the window the teach() refresh opened
    rebuilds.clear()
    for i in range(12):  # each new value brings new suggested rewards
        v = k.propose(RAM, 16 + i, kind=VariableKind.SCORE, name=f"S{i}", evidence=SURE)
        context.bus.publish(DiscoveryUpdated((v,), (), (), ()))
    assert len(rebuilds) == 1 and panel.refresher.pending  # the leading one only
    qtbot.waitUntil(lambda: not panel.refresher.pending, timeout=2000)
    assert len(rebuilds) == 2 and len(panel.behaviour_ids()) >= 12
    # A person's own change is never made to wait.
    bid = panel.behaviour_ids()[0]
    context.rewards.set_value(bid, 7.0)
    assert len(rebuilds) == 3 and not panel.refresher.pending


def test_the_name_column_can_be_dragged_wider(qtbot: QtBot, context: AppContext) -> None:
    """T-856: the first column of every table is wide and Interactive, the last stretches."""
    from PySide6.QtWidgets import QHeaderView, QTreeWidget

    from retro_ai_lab.ui.compare_panel import ComparePanel
    from retro_ai_lab.ui.goals_panel import GoalsPanel
    from retro_ai_lab.ui.panels import NAME_COLUMN_WIDTH
    from retro_ai_lab.ui.research_dialog import ResearchDialog

    widgets = [
        RewardsPanel(context),
        GoalsPanel(context),
        ComparePanel(context),
        ResearchDialog(context),
    ]
    names = ["rewards_tree", "goals_tree", "comparison_table", "situation_table"]
    for widget, name in zip(widgets, names, strict=True):
        qtbot.addWidget(widget)
        tree = widget.findChild(QTreeWidget, name)
        assert tree is not None, name
        header = tree.header()
        assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Interactive, name
        assert header.sectionSize(0) >= NAME_COLUMN_WIDTH, name
        assert header.stretchLastSection(), name
