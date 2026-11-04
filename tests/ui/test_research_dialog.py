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
    bare = next(r[0] for r in dialog.memory_rows() if r[3] == "")
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
