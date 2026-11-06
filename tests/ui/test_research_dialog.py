"""T-809: the Research window (settings, the situation now, the step log)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import Settings
from retro_ai_lab.training import COLUMNS, Mode, StepLog
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.main_window import MainWindow
from retro_ai_lab.ui.research_dialog import (
    IDLE_TEXT,
    LOG_COLUMNS,
    MAX_ROWS,
    MEMORY_COLUMNS,
    NO_MEMORY_TEXT,
    NO_SITUATION_TEXT,
    SETTING_FIELDS,
    ResearchDialog,
    settings_summary,
)
from tests.ui.test_learning_panel import connect


@pytest.fixture
def context(tmp_path: Path) -> AppContext:
    return AppContext(data_dir=tmp_path)


@pytest.fixture
def dialog(qtbot: QtBot, context: AppContext) -> ResearchDialog:
    d = ResearchDialog(context)
    qtbot.addWidget(d)
    d.show()
    return d


def test_settings_follow_the_live_ones_and_apply(
    dialog: ResearchDialog, context: AppContext
) -> None:
    t = context.training
    assert dialog.settings_values() == t.settings.to_dict()
    assert [k for k, _, _ in SETTING_FIELDS] == list(t.settings.to_dict())
    assert not dialog.findChild(object, "apply_settings_button").isEnabled()  # type: ignore[union-attr]
    assert not dialog.findChild(object, "setting_alpha").isEnabled()  # type: ignore[union-attr]
    connect(context)
    assert dialog.findChild(object, "apply_settings_button").isEnabled()  # type: ignore[union-attr]
    dialog.set_settings_values(alpha=0.5, gamma=0.8, epsilon=0.4, epsilon_min=0.1)
    assert dialog.apply_settings()
    assert t.settings == Settings(alpha=0.5, gamma=0.8, epsilon=0.4, epsilon_min=0.1)
    assert t.agent is not None and t.agent.epsilon == 0.4
    assert "applied" in dialog.note_text
    # A bad combination is refused in words and the live settings stay.
    dialog.set_settings_values(epsilon=0.05)
    assert not dialog.apply_settings()
    assert "epsilon_min can't be above epsilon" in dialog.note_text
    assert t.settings.epsilon == 0.4
    dialog.findChild(object, "revert_settings_button").click()  # type: ignore[union-attr]
    assert dialog.settings_values()["epsilon"] == 0.4
    with pytest.raises(Exception, match="no setting called"):
        dialog.set_settings_values(speed=1.0)
    # Exploring steps is a whole number from 1 up (T-875).
    assert dialog.settings_values()["explore_steps"] == 6000
    dialog.set_settings_values(explore_steps=0)
    assert dialog.settings_values()["explore_steps"] == 1
    dialog.set_settings_values(explore_steps=1234.6)
    assert dialog.settings_values()["explore_steps"] == 1235
    assert dialog.apply_settings() and t.settings.explore_steps == 1235
    # Exploring rebound is a share from 0 to 1 (T-932).
    assert dialog.settings_values()["rebound"] == 0.25
    dialog.set_settings_values(rebound=0.0)
    assert dialog.apply_settings() and t.settings.rebound == 0.0
    # Not while training runs.
    t.start()
    assert not dialog.findChild(object, "setting_alpha").isEnabled()  # type: ignore[union-attr]
    assert not dialog.apply_settings() and "Stop training" in dialog.note_text
    t.stop()
    assert dialog.findChild(object, "apply_settings_button").isEnabled()  # type: ignore[union-attr]


def test_situation_table_shows_every_move(dialog: ResearchDialog, context: AppContext) -> None:
    assert dialog.situation_text == IDLE_TEXT and dialog.situation_rows() == ()
    connect(context)
    dialog.refresh()
    assert dialog.situation_text == NO_SITUATION_TEXT
    t = context.training
    t.train(1)
    rows = dialog.situation_rows()
    assert t.agent is not None and len(rows) == len(t.agent.space)
    assert [r[0] for r in rows] == list(t.agent.space.labels)
    assert sum(r[2] == "★" for r in rows) >= 1 and all(r[1][0] in "+-" for r in rows)
    # The knowledge's names are used (discovery may add variables of its own as it goes).
    text = dialog.situation_text
    assert text.startswith("In the situation ") and "Lane=" in text
    context.connection.disconnect()
    assert dialog.situation_text == IDLE_TEXT and dialog.situation_rows() == ()


def test_memory_view_lists_names_and_filters(dialog: ResearchDialog, context: AppContext) -> None:
    """T-862: the memory table shows state bytes and guesses, filters, and names a value."""
    from retro_ai_lab.emulator.fake.toy_racer import LAYOUT

    assert dialog.memory_summary == NO_MEMORY_TEXT and dialog.memory_rows() == ()
    assert len(MEMORY_COLUMNS) == 5
    connect(context)
    context.training.train(2)  # a few steps: the tracker has seen the bytes move
    dialog.refresh()
    rows = {r[0]: r for r in dialog.memory_rows()}
    lane = f"ram[{LAYOUT.lane}]"
    assert lane in rows and rows[lane][3].startswith("Position (") and rows[lane][4] == "Lane"
    assert "(0x" in rows[lane][1] and rows[lane][2].endswith("of steps")
    assert "values that change like game state" in dialog.memory_summary
    assert "named" in dialog.memory_summary
    # The filter narrows by name, by address and by hex offset.
    dialog.set_memory_filter("lane")
    assert [r[0] for r in dialog.memory_rows()] == [lane]
    dialog.set_memory_filter(f"0x{LAYOUT.score:02x}")
    assert [r[0] for r in dialog.memory_rows()] == [f"ram[{LAYOUT.score}]"]
    dialog.set_memory_filter("")
    # Naming: nothing picked, then a guess renamed and confirmed, then a bare address named.
    assert not dialog.name_memory("Track")
    assert "Pick a value" in dialog.note_text
    dialog.select_memory(lane)
    assert dialog.name_memory("Track position")
    assert dialog.note_text.startswith("Track position:")
    rows = {r[0]: r for r in dialog.memory_rows()}
    assert rows[lane][4] == "Track position" and "(confirmed)" in rows[lane][3]
    # A state byte the AI has no guess at. Which bytes it has guessed after two
    # random steps varies from run to run, so when it guessed them all, take one
    # guess away: the address is then exactly as it is before the AI proposes it.
    bare_rows = [r[0] for r in dialog.memory_rows() if r[3] == ""]
    if not bare_rows:
        knowledge = context.discovery.knowledge
        assert knowledge is not None
        state = set(context.observation.tracker.classify()["state"])
        spare = next(
            v for v in knowledge if (v.region, v.offset) in state and v.offset != LAYOUT.lane
        )
        knowledge.remove(spare.id)
        dialog.refresh()
        bare_rows = [r[0] for r in dialog.memory_rows() if r[3] == ""]
    bare = bare_rows[0]
    dialog.select_memory(bare)
    assert not dialog.name_memory("   ") and "needs a name" in dialog.note_text
    assert dialog.name_memory("Mystery")
    rows = {r[0]: r for r in dialog.memory_rows()}
    assert rows[bare][4] == "Mystery" and rows[bare][3] == "Unknown (confirmed)"
    with pytest.raises(ConfigurationError):
        dialog.select_memory("ram[9999]")
    context.connection.disconnect()
    dialog.refresh()
    assert dialog.memory_summary == NO_MEMORY_TEXT and dialog.memory_rows() == ()


def test_log_count_rows_csv_and_clear(
    dialog: ResearchDialog, context: AppContext, tmp_path: Path
) -> None:
    assert (
        settings_summary(0, 0) == "0 steps logged." and settings_summary(1, 0) == "1 step logged."
    )
    assert settings_summary(1200, 3) == "1,200 steps logged, 3 older ones dropped."
    assert dialog.log_summary == "0 steps logged." and dialog.log_rows() == ()
    assert not dialog.findChild(object, "export_log_button").isEnabled()  # type: ignore[union-attr]
    connect(context)
    t = context.training
    records = t.train(2)
    steps = sum(r.result.steps for r in records)
    assert len(context.step_log) == steps
    assert dialog.log_summary == settings_summary(steps, 0)
    rows = dialog.log_rows()
    assert len(rows) == min(steps, MAX_ROWS) and LOG_COLUMNS[0] == "Go"
    assert rows[0][:2] == ("2", str(records[1].result.steps))  # newest first
    assert rows[-1][0] == "1" if steps <= MAX_ROWS else True
    assert all(len(r) == len(COLUMNS) for r in rows)
    path = tmp_path / "out" / "steps.csv"
    assert dialog.export_to(path) == steps and "Saved" in dialog.note_text
    assert StepLog.read_csv(path) == context.step_log.records
    assert dialog.export_to(tmp_path) == 0 and "Couldn't save" in dialog.note_text  # a folder
    dialog.findChild(object, "clear_log_button").click()  # type: ignore[union-attr]
    assert (
        len(context.step_log) == 0
        and dialog.log_rows() == ()
        and dialog.log_summary.startswith("0 ")
    )
    # The child's own moves are logged too, without a value.
    t.start(Mode.MANUAL)
    for _ in range(3):
        context.observation.record_input(["a"])
        context.connection.call(lambda a: a.advance_frames(4))
        context.observation.advance(4)
    t.stop()
    assert len(context.step_log) >= 1 and dialog.log_rows()[0][6] == ""


def test_toolbar_button_opens_the_window_and_logging_outlives_it(
    qtbot: QtBot, context: AppContext
) -> None:
    window = MainWindow(context)
    qtbot.addWidget(window)
    window.show()
    dialog = window.research_dialog
    assert not dialog.isVisible() and not dialog.ticking
    window.findChild(object, "research_button").click()  # type: ignore[union-attr]
    assert dialog.isVisible() and dialog.ticking
    dialog.close()
    assert not dialog.isVisible() and not dialog.ticking
    connect(context)
    context.training.train(1)
    assert len(context.step_log) > 0
    window.open_research()
    assert dialog.isVisible() and dialog.log_rows()


# --------------------------------------------------------------- T-880: tabs


def _tabs(dialog: ResearchDialog):  # type: ignore[no-untyped-def]
    from PySide6.QtWidgets import QTabWidget

    tabs = dialog.findChild(QTabWidget, "research_tabs")
    assert tabs is not None
    return tabs


def test_research_sections_are_tabs(dialog: ResearchDialog) -> None:
    from retro_ai_lab.ui.research_dialog import TAB_TITLES

    tabs = _tabs(dialog)
    assert TAB_TITLES == (
        "Learning settings",
        "The situation now",
        "Memory",
        "Step log",
        "Search values",
    )
    assert [tabs.tabText(i) for i in range(tabs.count())] == list(TAB_TITLES)
    for i, box in enumerate(("settings_box", "situation_box", "memory_box", "log_box")):
        widget = dialog.findChild(object, box)
        assert widget is not None and tabs.widget(i).isAncestorOf(widget)  # type: ignore[arg-type]
    # The intro and the note sit outside the tabs, so they show whatever tab is picked.
    for name in ("research_intro", "research_note"):
        widget = dialog.findChild(object, name)
        assert widget is not None and not tabs.isAncestorOf(widget)  # type: ignore[arg-type]
    assert tabs.currentIndex() == 0


def test_settings_explain_themselves(dialog: ResearchDialog) -> None:
    from PySide6.QtWidgets import QLabel

    _tabs(dialog).setCurrentIndex(0)
    for key, _label, tip in SETTING_FIELDS:
        hint = dialog.findChild(QLabel, f"setting_{key}_hint")
        assert hint is not None and hint.text() == tip and hint.isVisible()


def test_tab_titles_count_rows(dialog: ResearchDialog, context: AppContext) -> None:
    tabs = _tabs(dialog)
    assert tabs.tabText(2) == "Memory" and tabs.tabText(3) == "Step log"
    connect(context)
    context.training.train(2)
    dialog.refresh()
    steps = len(context.step_log)
    assert steps > 0 and tabs.tabText(3) == f"Step log ({steps:,})"
    memory = tabs.tabText(2)
    assert memory.startswith("Memory (") and memory.endswith(")")
    assert int(memory[len("Memory (") : -1].replace(",", "")) >= len(dialog.memory_rows()) > 0
    dialog.clear_log()
    assert tabs.tabText(3) == "Step log"
    context.connection.disconnect()
    dialog.refresh()
    assert tabs.tabText(2) == "Memory"


def test_picked_tab_stays_across_refresh(dialog: ResearchDialog, context: AppContext) -> None:
    tabs = _tabs(dialog)
    tabs.setCurrentIndex(3)
    connect(context)
    context.training.train(1)
    dialog.refresh()
    assert tabs.currentIndex() == 3


# ------------------------------------------- T-885: search values, then reward


def _poke(context: AppContext, offset: int, value: int) -> None:
    """Set a byte of the practice console's memory and take a snapshot."""
    from retro_ai_lab.emulator.fake import FakeEmulatorAdapter

    adapter = context.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    adapter.game._ram[offset] = value
    assert context.observation.observe() is not None


def test_search_is_off_without_a_game(dialog: ResearchDialog) -> None:
    from retro_ai_lab.ui.research_dialog import SEARCH_IDLE_TEXT

    assert dialog.search_summary == SEARCH_IDLE_TEXT and dialog.search_rows() == ()
    for name in ("search_new_button", "search_up_button", "reward_up_button"):
        assert not dialog.findChild(object, name).isEnabled()  # type: ignore[union-attr]
    assert not dialog.search_start()
    assert dialog.tabs.tabText(4) == "Search values"


def test_search_narrows_and_shows_hex(dialog: ResearchDialog, context: AppContext) -> None:
    from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
    from retro_ai_lab.observation import ScanKind

    connect(context)
    _poke(context, LAYOUT.lives, 3)
    dialog.refresh()
    assert dialog.findChild(object, "search_new_button").isEnabled()  # type: ignore[union-attr]
    assert dialog.search_start()  # every byte
    everything = dialog.search_count
    assert everything > 10 and dialog.tabs.tabText(4) == f"Search values ({everything:,})"
    assert "match" in dialog.search_summary and "frame" in dialog.search_summary
    assert dialog.search_refine(ScanKind.EXACT, "0x03")  # hex: 3
    lives = f"ram[{LAYOUT.lives}]"
    rows = {r[0]: r for r in dialog.search_rows()}
    assert lives in rows and all(r[2] == "3 (0x03)" for r in rows.values())
    assert rows[lives][1] == f"0x{LAYOUT.lives:04X}"
    _poke(context, LAYOUT.lives, 2)
    assert dialog.search_refine(ScanKind.DOWN)
    rows = {r[0]: r for r in dialog.search_rows()}
    assert lives in rows and rows[lives][2:] == ("2 (0x02)", "2 (0x02)", "3 (0x03)")
    # The Now column follows the game between scans; the scans stay as they were.
    _poke(context, LAYOUT.lives, 9)
    dialog.refresh()
    rows = {r[0]: r for r in dialog.search_rows()}
    assert rows[lives][2:] == ("9 (0x09)", "2 (0x02)", "3 (0x03)")
    _poke(context, LAYOUT.lives, 2)
    assert all(int(r[3].split()[0]) < 3 for r in rows.values())
    # A bad number is refused and the search stays.
    count = dialog.search_count
    assert not dialog.search_refine(ScanKind.EXACT, "lots")
    assert "not a number" in dialog.note_text and dialog.search_count == count
    assert not dialog.search_start("100000000") and dialog.search_count == count
    assert "99,999,999" in dialog.note_text
    # A score of 300 is not one byte; it is searched for across bytes (T-936).
    _poke(context, LAYOUT.score, 0x2C)
    _poke(context, LAYOUT.score + 1, 0x01)
    assert dialog.search_start("300") and dialog.note_text == ""
    rows = {r[0]: r for r in dialog.search_rows()}
    assert f"ram[{LAYOUT.score}]" in rows and f"ram[{LAYOUT.score + 1}]" in rows
    # Start over empties it; disconnecting does too.
    dialog.search_reset()
    assert dialog.search_count == 0 and dialog.tabs.tabText(4) == "Search values"
    assert dialog.search_start("$02")
    assert dialog.search_count > 0
    context.connection.disconnect()
    dialog.refresh()
    assert dialog.search_count == 0 and dialog.search_rows() == ()


def test_a_found_value_becomes_a_reward(dialog: ResearchDialog, context: AppContext) -> None:
    from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
    from retro_ai_lab.rewards.model import Category
    from retro_ai_lab.rewards.own import own_id
    from retro_ai_lab.understanding.events import EventKind

    connect(context)
    _poke(context, LAYOUT.lives, 7)
    dialog.refresh()
    assert dialog.search_start("7")
    lives = f"ram[{LAYOUT.lives}]"
    assert not dialog.reward_search(goes_up=True, name="Lives", points=1.0)
    assert "Pick a value" in dialog.note_text
    dialog.select_search(lives)
    assert not dialog.reward_search(goes_up=True, name="  ", points=1.0)
    assert "name" in dialog.note_text
    assert dialog.reward_search(goes_up=False, name="My lives", points=-2.5)
    knowledge = context.discovery.knowledge
    assert knowledge is not None
    var = knowledge.at("ram", LAYOUT.lives)
    assert var is not None and var.confirmed and var.name == "My lives"
    profile = context.rewards.profile
    assert profile is not None
    row = profile.find(own_id(var.id, False))
    assert row is not None and row.value == -2.5 and row.name == "My lives goes down"
    assert row.trigger.kind is EventKind.DOWN and row.category is Category.PROGRESSION
    assert "My lives goes down" in dialog.note_text and "-2.5" in dialog.note_text
    assert dialog.reward_search(goes_up=True, name="My lives", points=0.65)
    up = profile.find(own_id(var.id, True))
    assert up is not None and up.value == 0.65
    assert not dialog.reward_search(goes_up=True, name="My lives", points=1.0)
    assert "already" in dialog.note_text
    # The buttons use the boxes.
    dialog.findChild(object, "search_name_edit").setText("Spare")  # type: ignore[union-attr]
    dialog.findChild(object, "search_points").setValue(3.25)  # type: ignore[union-attr]
    assert dialog.search_start()  # every byte again, so there is another to pick
    other = next(r[0] for r in dialog.search_rows() if r[0] != lives)
    dialog.select_search(other)
    dialog.findChild(object, "reward_up_button").click()  # type: ignore[union-attr]
    assert "Spare goes up" in dialog.note_text
