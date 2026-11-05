"""T-307: the "What the AI is doing" panel."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import Capabilities
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, LAYOUT
from retro_ai_lab.services import DiscoveryService, DiscoveryUpdated, ObservationService
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.discovery_panel import (
    IDLE_TEXT,
    NO_SELECTION,
    SORTS,
    DiscoveryPanel,
    badge,
)
from retro_ai_lab.ui.main_window import MainWindow
from retro_ai_lab.understanding import Confidence, Evidence, VariableKind
from retro_ai_lab.understanding.events import EventKind, GameEvent

SURE = [Evidence("test", "", 6.0)]

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
    panel.refresher.flush()  # the review's update may still be inside the refresh window
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
    panel.refresher.flush()
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


def test_many_updates_rebuild_the_list_once(
    panel: DiscoveryPanel, context: AppContext, qtbot: QtBot
) -> None:
    """T-843: training publishes an update on most steps; the list follows a few times a second."""
    context.connection.connect("fake", seed=3)
    k = context.discovery.knowledge
    assert k is not None
    rebuilds: list[int] = []
    original = panel.refresh

    def counted(*, keep: str | None = None) -> None:
        rebuilds.append(1)
        original(keep=keep)

    panel.refresh = counted  # type: ignore[method-assign]
    runs = panel.refresher.runs
    v = k.propose("ram", 0, kind=VariableKind.SCORE, name="Score", evidence=SURE)
    context.bus.publish(DiscoveryUpdated((v,), (), (), ()))
    assert panel.refresher.runs == runs + 1 and v.id in panel.variable_ids()  # shows at once
    for i in range(1, 11):
        w = k.propose("ram", i, kind=VariableKind.PROGRESS, name=f"P{i}", evidence=SURE)
        context.bus.publish(DiscoveryUpdated((w,), (), (), ()))
    assert panel.refresher.runs == runs + 1 and panel.refresher.pending
    qtbot.waitUntil(lambda: not panel.refresher.pending, timeout=2000)
    assert panel.refresher.runs == runs + 2 and len(panel.variable_ids()) == 11
    assert rebuilds == []  # T-896: rows are added, the list is never rebuilt
    # Events alone refresh the lists under the variables, not the variables.
    event = GameEvent(EventKind.PROGRESS, 7, "v2", "P1 went up to 1.", 0, 1)
    context.discovery.events.extend((event,))
    panel.refresher.flush()
    context.bus.publish(DiscoveryUpdated((), (), (event,), ()))
    qtbot.waitUntil(lambda: not panel.refresher.pending, timeout=2000)
    assert rebuilds == [] and any("P1 went up" in t for t in panel.event_texts())


# ------------------------------------------- T-896: rows change, the list stays


def test_updates_touch_only_the_rows_that_changed(
    panel: DiscoveryPanel, context: AppContext, qtbot: QtBot
) -> None:
    """A review's update adds, repaints or drops rows in place; selection and order hold."""
    context.connection.connect("fake", seed=3)
    k = context.discovery.knowledge
    assert k is not None
    first = k.propose("ram", 4, kind=VariableKind.SCORE, name="Score", evidence=SURE)
    last = k.propose("ram", 9, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
    panel.refresh()
    panel.select(last.id)
    items = {i: panel._list.item(n) for n, i in enumerate(panel.variable_ids())}
    # A new one lands between them, in (region, offset) order, without a rebuild.
    panel.refresher.flush()
    mid = k.propose("ram", 6, kind=VariableKind.PROGRESS, name="Level", evidence=SURE)
    context.bus.publish(DiscoveryUpdated((mid,), (), (), ()))
    panel.refresher.flush()
    assert panel.variable_ids() == (first.id, mid.id, last.id)
    assert panel.selected_id == last.id and panel._list.item(2) is items[last.id]
    # A changed one keeps its row and gets a fresh line, badge and evidence.
    k.reject(first.id)
    panel.refresher.flush()
    context.bus.publish(DiscoveryUpdated((), (k.get(first.id),), (), ()))
    panel.refresher.flush()
    assert panel._list.item(0) is items[first.id] and "You said no" in panel.item_text(first.id)
    assert panel._list.item(0).foreground().color() == Qt.GlobalColor.gray
    k.correct(first.id, name="Points")  # taken back: the usual colour again
    panel.refresher.flush()
    context.bus.publish(DiscoveryUpdated((), (k.get(first.id),), (), ()))
    panel.refresher.flush()
    assert panel.item_text(first.id).startswith("Points") and (
        panel._list.item(0).foreground().color() != Qt.GlobalColor.gray
    )
    k.support(last.id, Evidence("test", "Went down when the car crashed.", 1.0))
    panel.refresher.flush()
    context.bus.publish(DiscoveryUpdated((), (k.get(last.id),), (), ()))
    panel.refresher.flush()
    assert "Went down when the car crashed." in panel.evidence_text  # the selected one
    # One the knowledge no longer holds loses its row; the selection moves with Qt.
    k.remove(mid.id)
    panel.refresher.flush()
    context.bus.publish(DiscoveryUpdated((), (mid,), (), ()))
    panel.refresher.flush()
    assert panel.variable_ids() == (first.id, last.id) and panel.selected_id == last.id
    # A change nobody announced still shows: the list falls back to a rebuild.
    quiet = k.propose("ram", 1, kind=VariableKind.FLAG, name="Quiet", evidence=SURE)
    panel.refresher.flush()
    context.bus.publish(DiscoveryUpdated((), (k.get(last.id),), (), ()))
    panel.refresher.flush()
    assert panel.variable_ids() == (quiet.id, first.id, last.id) and panel.selected_id == last.id
    # Direct use, for the benchmark and the panel's own verdict buttons.
    panel.update_rows(())
    assert len(panel.variable_ids()) == 3


# ------------------------------------------- T-883: grouped, without repeats


def test_the_watching_line_is_not_shown_twice(panel: DiscoveryPanel, context: AppContext) -> None:
    from retro_ai_lab.services import DiscoveryProgress

    stage = panel.findChild(object, "discovery_stage")
    summary = panel.findChild(object, "discovery_summary")
    assert stage is not None and summary is not None
    assert stage.isVisible() and summary.isVisible()  # type: ignore[attr-defined]
    context.connection.connect("fake")
    assert "Toy Racer" in panel.stage_text and stage.isVisible()  # type: ignore[attr-defined]
    # The engine copies its summary into the stage line while it watches.
    copy = context.discovery.summary()
    context.bus.publish(DiscoveryProgress("watching", copy))
    assert panel.stage_text == copy == panel.summary_text
    assert not stage.isVisible() and summary.isVisible()  # type: ignore[attr-defined]
    drive(context, 200)
    panel.refresher.flush()
    assert not stage.isVisible()  # type: ignore[attr-defined]
    context.bus.publish(DiscoveryProgress("experimenting", "Trying each button."))
    assert panel.stage_text == "Trying each button." and stage.isVisible()  # type: ignore[attr-defined]
    context.connection.disconnect()
    assert panel.stage_text == IDLE_TEXT and stage.isVisible()  # type: ignore[attr-defined]


def test_sections_have_headings(panel: DiscoveryPanel) -> None:
    from PySide6.QtWidgets import QLabel

    headings = [
        label.text() for label in panel.findChildren(QLabel) if label.property("heading") is True
    ]
    assert headings == [
        "What the AI thinks it has found",
        "What just happened",
        "What leads to what",
    ]


def test_action_buttons_share_a_row(panel: DiscoveryPanel) -> None:
    names = ("experiment_button", "export_button", "import_button")
    buttons = [panel.findChild(object, n) for n in names]
    assert all(b is not None for b in buttons)
    tops = {b.geometry().top() for b in buttons}  # type: ignore[union-attr]
    assert len(tops) == 1


# ------------------------------------------- T-898: find and sort


def test_the_find_box_narrows_the_list(panel: DiscoveryPanel, context: AppContext) -> None:
    """Hundreds of guesses: typing part of a name or address leaves only the matches."""
    assert not panel.findChild(object, "discovery_find").isEnabled()  # type: ignore[union-attr]
    context.connection.connect("fake", seed=3)
    assert panel.findChild(object, "discovery_find").isEnabled()  # type: ignore[union-attr]
    k = context.discovery.knowledge
    assert k is not None
    lives = k.propose("ram", 4, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
    lifes = k.propose("ram", 9, kind=VariableKind.PROGRESS, name="Lifes", evidence=SURE)
    score = k.propose("ram", 12, kind=VariableKind.SCORE, name="Score", evidence=SURE)
    panel.refresh()
    total = len(panel.variable_ids())
    assert panel.found_text == "" and len(panel.visible_ids()) == total
    panel.find_rows("LIF")  # any case, anywhere in the line
    assert panel.visible_ids() == (lifes.id,) and panel.found_text == f"1 of {total}"
    panel.find_rows("ram[12]")  # the address works too
    assert panel.visible_ids() == (score.id,)
    panel.find_rows("Confirmed by you")  # and the badge
    assert panel.visible_ids() == ()
    panel.find_rows("Liv")
    assert panel.visible_ids() == (lives.id,)
    # A row arriving while the box is in use obeys it; the selection is untouched.
    panel.select(lives.id)
    panel.refresher.flush()
    later = k.propose("ram", 20, kind=VariableKind.LIVES, name="Lives (2nd guess)", evidence=SURE)
    context.bus.publish(DiscoveryUpdated((later,), (), (), ()))
    panel.refresher.flush()
    assert panel.visible_ids() == (lives.id, later.id) and panel.selected_id == lives.id
    assert panel.found_text == f"2 of {total + 1}"
    panel.find_rows("")
    assert len(panel.visible_ids()) == total + 1 and panel.found_text == ""


def test_sorting_puts_your_own_verdicts_first(panel: DiscoveryPanel, context: AppContext) -> None:
    """ "Most sure first" shows what a person confirmed at the top and rejected at the bottom."""
    assert [label for label, _ in SORTS] == ["By address", "By name", "Most sure first"]
    context.connection.connect("fake", seed=3)
    k = context.discovery.knowledge
    assert k is not None
    for v in list(k):
        k.remove(v.id)
    zed = k.propose("ram", 1, kind=VariableKind.SCORE, name="Zed", evidence=SURE)
    mid = k.propose("ram", 5, kind=VariableKind.LIVES, name="Mid", evidence=SURE)
    low = k.propose("ram", 7, kind=VariableKind.FLAG, name="Apple")  # no evidence: low
    sworn = k.propose("ram", 9, kind=VariableKind.PROGRESS, name="Lifes", evidence=SURE)
    k.confirm(sworn.id)
    k.reject(zed.id)
    panel.refresh()
    assert panel.sort_name == "By address"
    assert panel.variable_ids() == (zed.id, mid.id, low.id, sworn.id)
    panel.select(mid.id)
    panel.sort_by("By name")
    assert (
        panel.variable_ids() == (low.id, sworn.id, mid.id, zed.id) and panel.selected_id == mid.id
    )
    panel.sort_by("Most sure first")
    assert panel.variable_ids() == (sworn.id, mid.id, low.id, zed.id)
    # A guess whose standing changes moves to its new place without a rebuild.
    rebuilds: list[int] = []
    panel.refresh = lambda *, keep=None: rebuilds.append(1)  # type: ignore[method-assign]
    k.confirm(low.id)
    panel.refresher.flush()
    context.bus.publish(DiscoveryUpdated((), (k.get(low.id),), (), ()))
    panel.refresher.flush()
    assert panel.variable_ids() == (low.id, sworn.id, mid.id, zed.id)
    assert rebuilds == [] and panel.selected_id == mid.id
    k.reject(mid.id)  # the selected one moves and stays selected
    panel.refresher.flush()
    context.bus.publish(DiscoveryUpdated((), (k.get(mid.id),), (), ()))
    panel.refresher.flush()
    assert panel.variable_ids() == (low.id, sworn.id, zed.id, mid.id)
    assert rebuilds == [] and panel.selected_id == mid.id
    assert "You said no" in panel.evidence_text
