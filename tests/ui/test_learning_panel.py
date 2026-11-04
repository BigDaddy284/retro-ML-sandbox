"""T-507: the Learning panel and the game controls yielding to the AI."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import Capabilities
from retro_ai_lab.emulator.fake.adapter import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.services import DiscoveryUpdated
from retro_ai_lab.training import Mode, Speed, Status
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.controller_pad import AI_HAS_CONTROLS, KEY_HINT
from retro_ai_lab.ui.learning_panel import ENDED_BY, IDLE_TEXT, LearningPanel, go_cells
from retro_ai_lab.ui.main_window import MainWindow
from retro_ai_lab.understanding import Encoding, Evidence, Knowledge, VariableKind

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]


@pytest.fixture
def context(tmp_path: Path) -> AppContext:
    return AppContext(data_dir=tmp_path)


@pytest.fixture
def panel(qtbot: QtBot, context: AppContext) -> LearningPanel:
    p = LearningPanel(context)
    qtbot.addWidget(p)
    p.show()
    return p


def connect(context: AppContext, **kw: object) -> Knowledge:
    context.connection.connect("fake", seed=3, **kw)  # type: ignore[arg-type]
    k = context.discovery.knowledge
    assert k is not None
    if k.at(RAM, LAYOUT.lane) is not None:
        return k  # taught earlier in this test and saved by the knowledge store
    added = (
        k.propose(RAM, LAYOUT.lane, kind=VariableKind.POSITION, name="Lane", evidence=SURE),
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


def test_idle_until_connected(panel: LearningPanel, context: AppContext) -> None:
    assert panel.status_text == IDLE_TEXT and panel.go_rows() == ()
    for name in ("start_button", "pause_button", "stop_button", "forget_button"):
        assert not panel.findChild(object, name).isEnabled()  # type: ignore[union-attr]
    assert not panel.ticking
    panel.start()  # refused, explained, nothing raised
    assert "Can't train yet" in panel.note_text
    context.connection.connect(
        "fake", capabilities=Capabilities(screen_capture=True, frame_advance=True, input=True)
    )
    assert panel.status_text == IDLE_TEXT  # no memory: nothing to learn from
    context.connection.disconnect()
    connect(context)
    assert panel.status_text.startswith("Ready to train.")
    assert panel.mode is Mode.GUIDED and panel.speed is Speed.LIVE
    assert panel.findChild(object, "start_button").isEnabled()  # type: ignore[union-attr]
    assert not panel.findChild(object, "stop_button").isEnabled()  # type: ignore[union-attr]
    assert panel.findChild(object, "forget_button").isEnabled()  # type: ignore[union-attr]


def test_start_tick_pause_resume_stop_and_the_goes_list(
    panel: LearningPanel, context: AppContext
) -> None:
    connect(context)
    t = context.training
    panel.choose_mode(Mode.INDEPENDENT)
    assert t.mode is Mode.INDEPENDENT and panel.mode is Mode.INDEPENDENT
    panel.start()
    assert t.status is Status.RUNNING and panel.ticking and panel.note_text == ""
    assert panel.tick_interval_ms == int(1000 / 60 * t.frames_per_step)
    assert not panel.findChild(object, "start_button").isEnabled()  # type: ignore[union-attr]
    assert not panel.findChild(object, "forget_button").isEnabled()  # type: ignore[union-attr]
    assert panel.tick() == 1 and panel.tick() == 1
    assert t.episode is not None and t.episode.steps == 2
    assert "Go 1: playing, step 2" in panel.status_text
    assert "Pressing: " in panel.status_text and "situation" in panel.status_text
    panel.pause_or_resume()
    assert t.status is Status.PAUSED and not panel.ticking
    assert panel.findChild(object, "pause_button").text() == "Resume"  # type: ignore[union-attr]
    assert panel.tick() == 0 and t.episode.steps == 2
    panel.pause_or_resume()
    assert t.status is Status.RUNNING and panel.ticking
    assert panel.findChild(object, "pause_button").text() == "Pause"  # type: ignore[union-attr]
    while t.episodes < 1:
        panel.tick()
    assert len(panel.go_rows()) == 1
    panel.tick()
    panel.stop()
    assert t.status is Status.IDLE and not panel.ticking
    rows = panel.go_rows()
    assert len(rows) == 2 and rows[0][0] == "2" and rows[1][0] == "1"  # newest first
    assert rows[0][1] == "On its own" and rows[0][4] == ENDED_BY["stopped"]
    assert rows[1][4] in (ENDED_BY["max_steps"], ENDED_BY["reset"], ENDED_BY["stuck"])
    assert panel.status_text.startswith("Resting after 2 goes.")
    assert go_cells(t.history[0])[2] == str(t.history[0].result.steps)


def test_fast_speed_takes_many_steps_per_tick(panel: LearningPanel, context: AppContext) -> None:
    connect(context)
    panel.choose_speed(Speed.FAST)
    assert context.training.speed is Speed.FAST and panel.speed is Speed.FAST
    panel.start()
    assert panel.tick_interval_ms == 1
    assert panel.tick() > 1
    panel.choose_speed(Speed.LIVE)
    assert panel.ticking and panel.tick_interval_ms > 1 and panel.tick() == 1
    panel.stop()


def test_game_controls_yield_to_the_ai_and_come_back(qtbot: QtBot, context: AppContext) -> None:
    w = MainWindow(context)
    qtbot.addWidget(w)
    w.show()
    game, learning = w.game_panel, w.learning_panel
    assert learning.parent() is w.panel("learning").content
    connect(context)
    assert game.clock.running and game.clock.playing and game.pad.press("up")
    game.pad.release("up")
    learning.start()
    assert context.training.ai_in_control
    assert not game.clock.running and not game.clock.playing
    assert game.pad.locked and not game.pad.press("up") and game.pad.message == AI_HAS_CONTROLS
    assert not game.play_button.isEnabled()
    assert learning.tick() == 1  # the AI plays while the clock is stopped
    learning.stop()
    assert not game.pad.locked and game.pad.message == KEY_HINT
    assert game.clock.running and game.play_button.isEnabled()
    assert game.pad.press("up")
    game.pad.release("up")
    # If the child had paused the game, it stays paused after the AI is done.
    game.clock.pause()
    learning.start()
    learning.stop()
    assert not game.clock.running and game.play_button.isEnabled()
    w.close()


def test_pictures_rest_while_the_ai_practises_fast(qtbot: QtBot, context: AppContext) -> None:
    """T-836: at "Practise fast" the pump drops to a few pictures a second."""
    from retro_ai_lab.ui.game_panel import DEFAULT_FPS, PRACTISE_FPS

    w = MainWindow(context)
    qtbot.addWidget(w)
    w.show()
    game, learning = w.game_panel, w.learning_panel
    connect(context)
    normal = game.pump.interval_ms
    assert normal == 1000 // DEFAULT_FPS
    learning.choose_speed(Speed.FAST)
    assert game.pump.interval_ms == normal  # not running yet
    learning.start()
    assert game.pump.running and game.pump.interval_ms == 1000 // PRACTISE_FPS
    learning.choose_speed(Speed.LIVE)
    assert game.pump.interval_ms == normal
    learning.choose_speed(Speed.FAST)
    assert game.pump.interval_ms == 1000 // PRACTISE_FPS
    learning.pause_or_resume()
    assert game.pump.interval_ms == normal
    learning.stop()
    assert game.pump.interval_ms == normal
    w.close()


def test_manual_mode_leaves_the_controls_with_the_child(qtbot: QtBot, context: AppContext) -> None:
    w = MainWindow(context)
    qtbot.addWidget(w)
    w.show()
    game, learning = w.game_panel, w.learning_panel
    connect(context)
    learning.choose_mode(Mode.MANUAL)
    learning.start()
    t = context.training
    assert t.status is Status.RUNNING and not t.ai_in_control
    assert game.clock.running and not game.pad.locked and not learning.ticking
    assert learning.tick() == 0
    assert "watching you play" in learning.status_text
    agent = t.agent
    assert agent is not None and agent.updates == 0
    assert game.pad.press("up")
    for _ in range(t.frames_per_step * 3):
        game.clock.tick()
    game.pad.release("up")
    assert agent.updates == 3
    learning.choose_mode(Mode.GUIDED)  # mid-go: closes the go, the AI takes over
    assert game.pad.locked and len(learning.go_rows()) == 1
    assert learning.go_rows()[0][1] == "You play"
    assert learning.go_rows()[0][4] == ENDED_BY["mode change"]
    learning.stop()
    w.close()


def test_forget_and_link_loss(panel: LearningPanel, context: AppContext) -> None:
    connect(context)
    t = context.training
    panel.start()
    panel.forget()
    assert "Stop training" in panel.note_text and t.episodes == 0
    panel.stop()
    panel.forget()
    assert "forgotten" in panel.note_text
    assert t.agent is not None and t.agent.updates == 0 and panel.go_rows() == ()
    panel.start()
    for _ in range(3):
        assert panel.tick() == 1
    assert t.episode is not None  # a go is running when the link goes
    adapter = context.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    adapter.drop_link()
    while context.connection.is_connected:
        panel.tick()
    assert t.status is Status.IDLE and not panel.ticking
    assert panel.status_text == IDLE_TEXT and "went away" in panel.note_text
    assert not panel.findChild(object, "start_button").isEnabled()  # type: ignore[union-attr]
    connect(context)
    assert panel.note_text == "" and panel.go_rows()  # the brain came back from disk
    assert panel.go_rows()[0][4] == ENDED_BY["disconnected"]


def test_panel_unsubscribes_when_destroyed(qtbot: QtBot, context: AppContext) -> None:
    p = LearningPanel(context)
    qtbot.addWidget(p)
    before = context.bridge.handler_count()
    p.deleteLater()
    qtbot.waitUntil(lambda: context.bridge.handler_count() < before)
    connect(context)
    context.connection.disconnect()


def test_how_the_ai_learns_picker(panel: LearningPanel, context: AppContext) -> None:
    from retro_ai_lab.learning import AUTO, DQN, Q_TABLE, DQNLearner, QLearningAgent

    combo = panel.findChild(object, "algorithm_combo")
    assert combo is not None and not combo.isEnabled()  # type: ignore[attr-defined]
    labels = [combo.itemText(i) for i in range(combo.count())]  # type: ignore[attr-defined]
    assert labels == ["Choose for me", "Simple table", "Neural network", "Skills per goal step"]
    assert panel.algorithm == AUTO
    connect(context)
    t = context.training
    assert combo.isEnabled() and isinstance(t.agent, QLearningAgent)  # type: ignore[attr-defined]
    assert "simple table" in panel.status_text
    panel.choose_mode(Mode.INDEPENDENT)
    panel.start()
    assert not combo.isEnabled()  # type: ignore[attr-defined]
    panel.choose_algorithm(DQN)  # refused while training, explained
    assert "Stop training" in panel.note_text and isinstance(t.agent, QLearningAgent)
    while panel.tick():
        if t.episodes:
            break
    panel.stop()
    panel.choose_algorithm(DQN)
    assert panel.algorithm == DQN and t.algorithm == DQN
    assert isinstance(t.agent, DQNLearner) and "neural network" in panel.note_text
    assert "neural network" in panel.status_text
    panel.choose_algorithm(Q_TABLE)
    assert isinstance(t.agent, QLearningAgent) and "simple table" in panel.note_text
    panel.choose_algorithm(AUTO)
    assert panel.algorithm == AUTO and isinstance(t.agent, QLearningAgent)


def test_learn_from_my_moves_again(qtbot: QtBot, context: AppContext) -> None:
    w = MainWindow(context)
    qtbot.addWidget(w)
    w.show()
    game, learning = w.game_panel, w.learning_panel
    button = learning.findChild(object, "rehearse_button")
    assert button is not None and not button.isEnabled()  # type: ignore[attr-defined]
    connect(context)
    t = context.training
    assert not button.isEnabled()  # type: ignore[attr-defined]  # nothing shown yet
    learning.choose_mode(Mode.MANUAL)
    learning.start()
    assert game.pad.press("up")
    for _ in range(t.frames_per_step * 3):
        game.clock.tick()
    game.pad.release("up")
    assert t.demonstrations.count == 3 and "shown the AI 3 moves" in learning.status_text
    assert not button.isEnabled()  # type: ignore[attr-defined]  # not while training
    learning.stop()
    assert button.isEnabled()  # type: ignore[attr-defined]
    agent = t.agent
    assert agent is not None
    before = agent.updates
    learning.rehearse()
    assert agent.updates == before + 3 and "learned from your 3 moves again" in learning.note_text
    w.close()


def test_practise_from_here_pins_and_forgets_the_spot(
    panel: LearningPanel, context: AppContext
) -> None:
    from retro_ai_lab.ui.learning_panel import (
        PIN_TEXT,
        PINNED_NOTE,
        SPOT_LINE,
        UNPIN_TEXT,
        UNPINNED_NOTE,
    )

    button = panel.findChild(object, "spot_button")
    assert button is not None and button.text() == PIN_TEXT and not button.isEnabled()
    connect(context)
    t = context.training
    assert button.isEnabled()
    panel.toggle_spot()
    assert t.spot_pinned and button.text() == UNPIN_TEXT
    assert panel.note_text == PINNED_NOTE and SPOT_LINE in panel.status_text
    panel.start()
    assert not button.isEnabled()
    panel.tick()
    panel.stop()
    assert button.isEnabled() and button.text() == UNPIN_TEXT
    panel.toggle_spot()
    assert not t.spot_pinned and button.text() == PIN_TEXT
    assert panel.note_text == UNPINNED_NOTE and SPOT_LINE not in panel.status_text
    context.connection.disconnect()
    connect(
        context,
        capabilities=Capabilities(
            memory_read=True, screen_capture=True, frame_advance=True, input=True
        ),
    )
    assert not button.isEnabled()


def test_the_note_says_when_the_game_ignores_the_ai(
    panel: LearningPanel, context: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-852: training that stopped because the game is deaf explains itself."""
    import retro_ai_lab.training.service as module

    monkeypatch.setattr(module, "find_wake_button", lambda *a, **k: None)
    connect(context)
    adapter = context.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    while not adapter.game.state.game_over:
        adapter.game.step(["up"])
    panel.start()
    assert panel.tick() == 0
    assert context.training.status is Status.IDLE and not panel.ticking
    assert "no button gets it going" in panel.note_text
    assert panel.findChild(object, "start_button").isEnabled()  # type: ignore[union-attr]


def test_the_go_timers_are_editable_in_seconds(panel: LearningPanel, context: AppContext) -> None:
    """T-873: how long a go lasts, and how long before it gives up, in seconds a kid can type."""
    from retro_ai_lab.training import DEFAULT_MAX_STEPS, DEFAULT_UNSTICK_AFTER
    from retro_ai_lab.ui.learning_panel import seconds_to_steps, steps_to_seconds

    assert seconds_to_steps(40, 4) == 600 and steps_to_seconds(600, 4) == 40.0
    assert seconds_to_steps(0.01, 4) == 1
    go = panel.findChild(object, "go_seconds_spin")
    give_up = panel.findChild(object, "give_up_seconds_spin")
    assert go is not None and give_up is not None
    assert not go.isEnabled() and not give_up.isEnabled()  # type: ignore[attr-defined]
    connect(context)
    t = context.training
    fps = t.frames_per_step
    assert go.isEnabled() and panel.go_seconds == steps_to_seconds(DEFAULT_MAX_STEPS, fps)  # type: ignore[attr-defined]
    assert panel.give_up_seconds == round(steps_to_seconds(DEFAULT_UNSTICK_AFTER, fps), 1)
    go.setValue(30)  # type: ignore[attr-defined]
    assert t.max_steps == seconds_to_steps(30, fps) and "30 seconds" in panel.note_text
    give_up.setValue(3.0)  # type: ignore[attr-defined]
    assert t.unstick_after == seconds_to_steps(3.0, fps) and "3 seconds" in panel.note_text
    panel.choose_mode(Mode.INDEPENDENT)
    panel.start()
    assert not go.isEnabled() and not give_up.isEnabled()  # type: ignore[attr-defined]
    panel.choose_go_seconds(10)  # refused while training, and explained
    assert "Stop training" in panel.note_text and t.max_steps == seconds_to_steps(30, fps)
    panel.stop()
    assert go.isEnabled() and panel.go_seconds == 30.0  # type: ignore[attr-defined]
