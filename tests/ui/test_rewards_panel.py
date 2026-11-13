"""T-406: the Rewards panel."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QTreeWidget
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import Capabilities
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.rewards import Category, ExploitWarning
from retro_ai_lab.services import DiscoveryUpdated, ExploitSuspected
from retro_ai_lab.ui import coalesce
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


def test_the_controls_group_lists_every_button(panel: RewardsPanel, context: AppContext) -> None:
    """T-874: a row per button and one for repeats, in a Controls group, worth 0 to start."""
    from retro_ai_lab.rewards import AGAIN_ID, press_id

    context.connection.connect("fake", seed=2)
    panel.refresher.flush()
    names = [b.name for b in context.connection.adapter.buttons()]  # type: ignore[union-attr]
    assert f"Controls ({len(names) + 1})" in panel.category_texts()
    assert panel.row_text(press_id("up"))[:2] == ("Pressing Up", "+0")
    assert panel.row_text(AGAIN_ID)[:2] == ("Pressing the same button again", "+0")
    panel.select(press_id("up"))
    assert "Each step the button is held" in panel.explain_text
    panel.set_selected_value(-2)
    assert panel.row_text(press_id("up"))[1] == f"-2  ({YOU_SET_THIS})"


def test_rows_appear_under_their_categories(panel: RewardsPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    ids = panel.behaviour_ids()
    assert ids[0] == "reset:*" and all(i.startswith("button:") for i in ids[1:])  # T-874
    buttons = len(ids) - 1
    assert panel.category_texts() == ("Survival (1)", f"Controls ({buttons})")
    assert panel.summary_text == "1 reward switched on; +0.0 earned."  # button rows sit at 0
    panel.select("reset:*")
    teach(context)
    panel.refresher.flush()  # suggestions may wait for the refresh window (T-843)
    assert panel.selected_id == "reset:*"  # kept across the update
    assert panel.category_texts() == (
        "Collection (1)",
        "Survival (3)",
        "Progression (1)",
        f"Controls ({buttons})",
    )
    k = context.discovery.knowledge
    assert k is not None
    lives = k.at(RAM, LAYOUT.lives)
    assert lives
    lives_at = f"ram[{LAYOUT.lives}] (0x{LAYOUT.lives:04X})"  # the Address column (T-879)
    assert panel.row_text(f"setback:{lives.id}") == ("Losing Lives", "-10", "-10", "on", lives_at)
    assert panel.row_text("reset:*") == ("Starting over", "-5", "-5", "on", "")
    assert panel.row_text("button:press:up")[4] == ""
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


def test_the_points_box_takes_any_number(panel: RewardsPanel, context: AppContext) -> None:
    """T-872: a suggestion of 0.65 could not be got rid of with + and -; type the points instead."""
    context.connection.connect("fake", seed=2)
    k = teach(context)
    panel.refresher.flush()
    checkpoint = k.at(RAM, LAYOUT.checkpoint)
    assert checkpoint
    bid = f"progress:{checkpoint.id}"
    box = button(panel, "points_spin")
    assert not box.isEnabled() and panel.points_value == 0.0  # type: ignore[attr-defined]
    panel.select(bid)
    assert box.isEnabled() and panel.points_value == 10.0  # type: ignore[attr-defined]
    box.setValue(0.65)  # type: ignore[attr-defined]
    assert panel.row_text(bid)[1] == f"+0.65  ({YOU_SET_THIS})"
    assert panel.note_text == "Remembered for next time."
    panel.set_selected_value(-3)
    assert panel.row_text(bid)[1] == f"-3  ({YOU_SET_THIS})"
    assert panel.points_value == -3.0
    panel.increase_selected()  # + and - still work from the typed number
    assert panel.row_text(bid)[1] == f"-2  ({YOU_SET_THIS})" and panel.points_value == -2.0
    box.setValue(1000)  # type: ignore[attr-defined]  # clamped by the box itself
    assert panel.points_value == 100.0 and panel.row_text(bid)[1] == f"+100  ({YOU_SET_THIS})"
    panel.select(None)
    assert not box.isEnabled() and panel.points_value == 0.0  # type: ignore[attr-defined]


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
    label = panel.findChild(QLabel, "rewards_warning")
    assert label is not None and label.property("warning") is True  # themed, not inline (T-876)
    assert label.styleSheet() == ""
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
    panel: RewardsPanel, context: AppContext, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
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
    # The twelve changes come within one window. The window is 250 ms of real time,
    # and on a loaded machine one rebuild can take that long: the next change is then
    # rightly shown at once, and this test counted two (T-1018, seen on CI). So the
    # refresher's clock stands still while they come; its timer runs on as it does.
    now = time.monotonic()
    monkeypatch.setattr(coalesce, "time", SimpleNamespace(monotonic=lambda: now))
    for i in range(12):  # each new confirmed value takes the score's seat (T-858)
        v = k.propose(RAM, 27 - i, kind=VariableKind.SCORE, name=f"S{i}", evidence=SURE)
        k.confirm(v.id)
        context.bus.publish(DiscoveryUpdated((v,), (), (), ()))
    assert len(rebuilds) == 1 and panel.refresher.pending  # the leading one only
    qtbot.waitUntil(lambda: not panel.refresher.pending, timeout=2000)
    assert len(rebuilds) == 2 and f"score_gain:{v.id}" in panel.behaviour_ids()
    # A person's own change is never made to wait.
    bid = panel.behaviour_ids()[0]
    context.rewards.set_value(bid, 7.0)
    assert len(rebuilds) == 3 and not panel.refresher.pending


def test_picking_a_reward_explains_it(panel: RewardsPanel, context: AppContext) -> None:
    """T-859: a sentence under the table says what the picked reward means and is worth."""
    from retro_ai_lab.rewards import PICK_ONE, describe_category

    assert panel.explain_text == PICK_ONE
    context.connection.connect("fake", seed=2)
    k = teach(context)
    panel.refresher.flush()
    lives = k.at(RAM, LAYOUT.lives)
    assert lives
    bid = f"setback:{lives.id}"
    assert panel.explain_text == PICK_ONE
    panel.select(bid)
    assert panel.explain_text == "Losing Lives: the AI loses 10 points each time."
    panel.increase_selected()  # the sentence follows the value
    assert panel.explain_text.startswith("Losing Lives: the AI loses 9 points each time.")
    assert "You set the value; the AI suggested -10." in panel.explain_text
    panel.set_selected_enabled(False)
    assert panel.explain_text.endswith("It is switched off, so it counts for nothing right now.")
    panel.select(None)
    assert panel.explain_text == PICK_ONE
    tree = panel.findChild(QTreeWidget, "rewards_tree")
    assert tree is not None
    tops = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    assert any(t is not None and t.toolTip(0) == describe_category(Category.SURVIVAL) for t in tops)
    assert panel._items[bid].toolTip(0).startswith("Losing Lives: the AI loses 9 points each time.")


def test_adding_and_removing_a_reward_of_my_own(panel: RewardsPanel, context: AppContext) -> None:
    """T-861: pick a value, a direction and the points; the row appears and can be removed."""
    from retro_ai_lab.rewards import OWN_NOTE

    assert panel.own_choices() == ()
    assert not button(panel, "add_own_button").isEnabled()  # type: ignore[attr-defined]
    context.connection.connect("fake", seed=2)
    k = teach(context)
    lane = k.propose(RAM, LAYOUT.lane, kind=VariableKind.POSITION, evidence=SURE)
    k.propose(RAM, LAYOUT.speed, kind=VariableKind.SPEED, evidence=SURE, name="Speed")
    context.bus.publish(DiscoveryUpdated((lane,), (), (), ()))
    panel.refresher.flush()
    choices = dict(panel.own_choices())
    assert choices[lane.id] == "Your position" and "Speed" in choices.values()
    assert button(panel, "add_own_button").isEnabled()  # type: ignore[attr-defined]
    lives = k.at(RAM, LAYOUT.lives)
    assert lives
    panel.select(f"setback:{lives.id}")
    assert button(panel, "remove_own_button").isEnabled()  # type: ignore[attr-defined]
    panel.add_own(lane.id, True, 2.0)
    assert panel.note_text == "Reward added."
    up = f"mine:up:{lane.id}"
    assert panel.selected_id == up and panel.row_text(up) == (
        "Your position goes up",
        f"+2  ({YOU_SET_THIS})",
        "+2",
        "on",
        f"ram[{LAYOUT.lane}] (0x{LAYOUT.lane:04X})",
    )
    assert panel.explain_text.startswith("Your position goes up: the AI gets 2 points each time.")
    assert OWN_NOTE in panel.explain_text
    assert button(panel, "remove_own_button").isEnabled()  # type: ignore[attr-defined]
    panel.add_own(lane.id, False, -2.0)
    down = f"mine:down:{lane.id}"
    assert panel.row_text(down)[:2] == ("Your position goes down", f"-2  ({YOU_SET_THIS})")
    assert "Progression (3)" in panel.category_texts()  # the checkpoint and the two of mine
    panel.add_own(lane.id, True, 1.0)
    assert panel.note_text == "That reward is already there."
    # The AI's own suggestions leave the rows alone; Remove takes one out.
    context.rewards.refresh_suggestions()
    panel.refresher.flush()
    assert up in panel.behaviour_ids() and down in panel.behaviour_ids()
    panel.select(down)
    panel.remove_selected()
    assert panel.note_text == "Reward removed." and down not in panel.behaviour_ids()
    assert up in panel.behaviour_ids()
    context.connection.disconnect()
    assert not button(panel, "add_own_button").isEnabled()  # type: ignore[attr-defined]


def test_remove_takes_any_row_out_and_it_stays_out(
    panel: RewardsPanel, context: AppContext
) -> None:
    """T-937 gave Remove every row; T-943 makes a removal stick: the user found removed
    rows "auto readded" by the AI's next guess, and wanted to remove everything."""
    assert not button(panel, "remove_own_button").isEnabled()  # type: ignore[attr-defined]
    assert not button(panel, "suggest_again_button").isEnabled()  # type: ignore[attr-defined]
    context.connection.connect("fake", seed=2)
    k = teach(context)
    panel.refresher.flush()
    lives = k.at(RAM, LAYOUT.lives)
    assert lives
    row = f"setback:{lives.id}"
    assert row in panel.behaviour_ids()
    panel.select(row)
    assert button(panel, "remove_own_button").isEnabled()  # type: ignore[attr-defined]
    panel.remove_selected()
    panel.refresher.flush()
    assert panel.note_text.startswith('Removed "Losing Lives". It stays out for this game')
    assert not k.get(lives.id).rejected and row not in panel.behaviour_ids()
    assert button(panel, "suggest_again_button").isEnabled()  # type: ignore[attr-defined]
    # The AI's next life count brings no "Losing" row back.
    hearts = k.propose(RAM, LAYOUT.lap, kind=VariableKind.LIVES, name="Hearts", evidence=SURE)
    k.update(lives.id, evidence=())
    context.rewards.refresh_suggestions()
    panel.refresher.flush()
    ids = panel.behaviour_ids()
    assert f"setback:{hearts.id}" not in ids and f"recovery:{hearts.id}" in ids
    # Rows with nothing behind them go too.
    for bid in ("reset:*", "button:press:up"):
        panel.select(bid)
        panel.remove_selected()
        panel.refresher.flush()
        assert bid not in panel.behaviour_ids() and "stays out" in panel.note_text
    button(panel, "suggest_again_button").click()  # type: ignore[attr-defined]
    panel.refresher.flush()
    assert panel.note_text == "Brought back 3 suggested rewards."
    ids = panel.behaviour_ids()
    assert f"setback:{hearts.id}" in ids and "reset:*" in ids and "button:press:up" in ids
    assert not button(panel, "suggest_again_button").isEnabled()  # type: ignore[attr-defined]
    panel.select(None)
    assert not button(panel, "remove_own_button").isEnabled()  # type: ignore[attr-defined]


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


def test_the_address_column_shows_a_range(panel: RewardsPanel, context: AppContext) -> None:
    """T-941: a value over several bytes shows every byte it takes."""
    context.connection.connect("fake", seed=2)
    k = teach(context)
    panel.refresher.flush()
    score = k.at(RAM, LAYOUT.score)
    assert score is not None and score.width == 2
    first, last = LAYOUT.score, LAYOUT.score + 1
    assert panel.row_text(f"score_gain:{score.id}")[4] == (
        f"ram[{first}..{last}] (0x{first:04X}..0x{last:04X})"
    )


def test_the_table_rests_while_the_ai_practises_fast(
    panel: RewardsPanel, context: AppContext
) -> None:
    """T-944: the table follows the knowledge panel: once every 2 s while nobody watches."""
    from retro_ai_lab.training import Speed
    from retro_ai_lab.ui.coalesce import DEFAULT_DELAY_MS, REST_DELAY_MS

    context.connection.connect("fake", seed=2)
    t = context.training
    assert panel.refresher.delay_ms == DEFAULT_DELAY_MS
    t.set_speed(Speed.FAST)
    t.start()
    assert panel.refresher.resting and panel.refresher.delay_ms == REST_DELAY_MS
    t.stop()
    assert not panel.refresher.resting and panel.refresher.delay_ms == DEFAULT_DELAY_MS
