"""T-307: the "What the AI is doing" panel."""

from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import Capabilities
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, LAYOUT
from retro_ai_lab.services import DiscoveryService, ObservationService
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.discovery_panel import IDLE_TEXT, NO_SELECTION, DiscoveryPanel, badge
from retro_ai_lab.ui.main_window import MainWindow
from retro_ai_lab.understanding import Confidence, Evidence, VariableKind

BY_NAME = {b.name: b for b in BUTTONS}


@pytest.fixture
def context() -> AppContext:
    ctx = AppContext()
    # Faster schedules so a short drive produces discoveries.
    ctx.discovery = DiscoveryService(
        ctx.connection, ctx.observation, ctx.bus, review_every=60, experiment_every=None
    )
    return ctx


@pytest.fixture
def panel(qtbot: QtBot, context: AppContext) -> DiscoveryPanel:
    p = DiscoveryPanel(context)
    qtbot.addWidget(p)
    p.show()
    return p


def drive(context: AppContext, frames: int) -> None:
    adapter = context.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    obs: ObservationService = context.observation
    for i in range(frames):
        held = [BY_NAME["start"]] if adapter.game.state.game_over else [BY_NAME["up"]]
        if i % 50 in range(10, 13):
            held.append(BY_NAME["left"])
        context.connection.call(lambda a, held=held: a.set_input(held))
        context.connection.call(lambda a: a.advance_frames(1))
        obs.advance(1)


def test_idle_panel(panel: DiscoveryPanel) -> None:
    assert panel.stage_text == IDLE_TEXT
    assert panel.variable_ids() == () and panel.evidence_text == NO_SELECTION
    assert panel.summary_text == "Not watching a game."
    assert not panel.findChild(object, "confirm_button").isEnabled()  # type: ignore[union-attr]
    assert not panel.findChild(object, "experiment_button").isEnabled()  # type: ignore[union-attr]


def test_discoveries_appear_with_badges_and_evidence(
    panel: DiscoveryPanel, context: AppContext
) -> None:
    context.connection.connect("fake", seed=2)
    assert "Toy Racer" in panel.stage_text
    drive(context, 180)
    ids = panel.variable_ids()
    assert len(ids) >= 5
    k = panel.knowledge
    assert k is not None
    score = k.best(VariableKind.SCORE)
    assert score is not None
    assert badge(score) in ("A guess", "Probably")
    assert panel.item_text(score.id).endswith(badge(score))
    assert "Score" in panel.item_text(score.id)
    panel.select(score.id)
    assert panel.selected_id == score.id
    assert "Only ever goes up" in panel.evidence_text
    assert panel.findChild(object, "confirm_button").isEnabled()  # type: ignore[union-attr]
    assert panel.summary_text.startswith("Watching")


def test_confirm_reject_and_rename(panel: DiscoveryPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    k = panel.knowledge
    assert k is not None
    v = k.propose(
        "ram", LAYOUT.lives, kind=VariableKind.LIVES, evidence=[Evidence("t", "few", 1.0)]
    )
    w = k.propose("ram", LAYOUT.lap, kind=VariableKind.PROGRESS)
    panel.refresh()
    panel.select(v.id)
    panel.confirm_selected()
    assert k.get(v.id).confidence is Confidence.CONFIRMED
    assert panel.item_text(v.id).endswith("Confirmed by you")
    assert panel.selected_id == v.id  # selection survives the refresh
    panel.select(w.id)
    panel.reject_selected()
    assert k.get(w.id).rejected and panel.item_text(w.id).endswith("You said no")
    panel.select(v.id)
    panel.rename_selected("Lives left")
    assert k.get(v.id).name == "Lives left" and k.get(v.id).confirmed
    assert panel.item_text(v.id).startswith("Lives left")
    panel.rename_selected("   ")  # blank names are ignored
    assert k.get(v.id).name == "Lives left"
    panel.select(None)
    panel.confirm_selected()  # nothing selected: a no-op
    assert panel.evidence_text == NO_SELECTION


def test_rename_box_and_buttons_drive_the_same_actions(
    qtbot: QtBot, panel: DiscoveryPanel, context: AppContext
) -> None:
    context.connection.connect("fake", seed=2)
    k = panel.knowledge
    assert k is not None
    v = k.propose("ram", LAYOUT.score, kind=VariableKind.SCORE)
    panel.refresh()
    panel.select(v.id)
    edit = panel.findChild(object, "rename_edit")
    edit.setText("Points")  # type: ignore[union-attr]
    panel.findChild(object, "rename_button").click()  # type: ignore[union-attr]
    assert k.get(v.id).name == "Points"
    panel.findChild(object, "reject_button").click()  # type: ignore[union-attr]
    assert k.get(v.id).rejected
    panel.findChild(object, "confirm_button").click()  # type: ignore[union-attr]
    assert k.get(v.id).confirmed and not k.get(v.id).rejected


def test_events_and_relationships_are_listed(panel: DiscoveryPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=2)
    drive(context, 700)
    assert panel.event_texts(), "events should be listed once the AI believes in a few values"
    assert all(t.startswith("Frame ") for t in panel.event_texts())
    assert len(panel.event_texts()) <= 8


def test_experiment_button(panel: DiscoveryPanel, context: AppContext) -> None:
    context.connection.connect("fake", seed=5)
    button = panel.findChild(object, "experiment_button")
    assert button.isEnabled()  # type: ignore[union-attr]
    button.click()  # type: ignore[union-attr]
    assert context.discovery.experiments == 1
    k = panel.knowledge
    assert k is not None and k.at("ram", LAYOUT.input) is not None
    assert "Controller input" in panel.item_text(k.at("ram", LAYOUT.input).id)  # type: ignore[union-attr]
    context.connection.disconnect()
    assert panel.stage_text == IDLE_TEXT
    assert not button.isEnabled()  # type: ignore[union-attr]
    context.connection.connect(
        "fake", capabilities=Capabilities(memory_read=True, input=True, frame_advance=True)
    )
    assert not button.isEnabled()  # type: ignore[union-attr]
    panel.run_experiment()  # unavailable: explained in the stage line, no crash
    assert "experiments" in panel.stage_text


def test_main_window_hosts_the_panel(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    assert isinstance(window.discovery_panel, DiscoveryPanel)
    assert window.discovery_panel.parent() is not None
    assert window.panel("activity").findChild(DiscoveryPanel) is window.discovery_panel


# ------------------------------------------------------- persistence (T-308)


def test_panel_saves_verdicts_and_shares_files(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    from pathlib import Path

    base = Path(str(tmp_path))
    monkeypatch.setenv("RETRO_AI_LAB_DATA_DIR", str(base / "data"))
    ctx = AppContext()
    assert ctx.data_dir == base / "data"
    panel = DiscoveryPanel(ctx)
    qtbot.addWidget(panel)
    assert not panel.findChild(object, "export_button").isEnabled()  # type: ignore[union-attr]
    ctx.connection.connect("fake", seed=2)
    k = panel.knowledge
    assert k is not None
    v = k.propose("ram", LAYOUT.score, kind=VariableKind.SCORE)
    panel.refresh()
    panel.select(v.id)
    panel.confirm_selected()
    assert panel.note_text == "Remembered for next time."
    assert ctx.knowledge_store.load("toyracer-2").get(v.id).confirmed
    panel.export_to(base / "share.json")
    assert "share.json" in panel.note_text
    panel.import_from(base / "share.json")
    assert panel.note_text.startswith("Loaded 1 discoveries")
    panel.import_from(base / "missing.json")
    assert "Can't read" in panel.note_text
    panel.export_to(base / "nope" / "deep" / "x.json")
    assert "Can't write" in panel.note_text
    ctx.connection.disconnect()
    assert not panel.findChild(object, "import_button").isEnabled()  # type: ignore[union-attr]
