"""T-010: the play clock advances frames while playing."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import Capabilities
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.main_window import MainWindow
from retro_ai_lab.ui.play_clock import PlayClock


@pytest.fixture
def context() -> AppContext:
    return AppContext()


@pytest.fixture
def clock(qapp: object, context: AppContext) -> PlayClock:
    return PlayClock(context, fps=100)


def game(context: AppContext) -> FakeEmulatorAdapter:
    a = context.connection.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    return a


def test_idle_until_connected(clock: PlayClock) -> None:
    assert not clock.available
    assert not clock.running
    assert clock.playing
    assert not clock.tick()
    assert clock.frames == 0


def test_runs_on_connect_and_advances_the_game(
    qtbot: QtBot, clock: PlayClock, context: AppContext
) -> None:
    states: list[bool] = []
    clock.state_changed.connect(states.append)
    context.connection.connect("fake")
    assert clock.available and clock.running
    assert clock.interval_ms == 10
    qtbot.waitUntil(lambda: game(context).game.state.frame >= 5)
    assert clock.frames >= 5
    context.connection.disconnect()
    assert not clock.running and not clock.available
    assert states[0] is True and states[-1] is False


def test_pause_and_play(qtbot: QtBot, clock: PlayClock, context: AppContext) -> None:
    context.connection.connect("fake")
    clock.pause()
    assert not clock.running and not clock.playing
    frame = game(context).game.state.frame
    qtbot.wait(50)
    assert game(context).game.state.frame == frame  # time stands still
    assert clock.toggle() is True
    assert clock.running
    qtbot.waitUntil(lambda: game(context).game.state.frame > frame)
    assert clock.toggle() is False
    assert not clock.running


def test_pause_choice_survives_reconnect(clock: PlayClock, context: AppContext) -> None:
    clock.pause()
    context.connection.connect("fake")
    assert not clock.running
    context.connection.disconnect()
    clock.play()
    context.connection.connect("fake")
    assert clock.running
    assert clock.frames == 0


def test_tick_n_frames_directly(clock: PlayClock, context: AppContext) -> None:
    context.connection.connect("fake")
    clock.pause()
    assert clock.tick(7)
    assert game(context).game.state.frame == 7
    assert clock.frames == 7


def test_emulator_without_frame_advance(clock: PlayClock, context: AppContext) -> None:
    context.connection.connect("fake", capabilities=Capabilities(screen_capture=True, input=True))
    assert not clock.available and not clock.running
    assert not clock.tick()


def test_dropped_link_stops_the_clock(clock: PlayClock, context: AppContext) -> None:
    context.connection.connect("fake", drop_after_ops=0)
    assert clock.running
    assert not clock.tick()
    assert not clock.running
    assert not context.connection.is_connected


def test_transient_error_is_counted(clock: PlayClock, context: AppContext) -> None:
    from retro_ai_lab.core import EmulatorError

    context.connection.connect("fake")
    clock.pause()
    adapter = game(context)
    original = adapter._do_advance_frames
    calls = {"n": 0}

    def flaky(n: int) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise EmulatorError("hiccup")
        original(n)

    adapter._do_advance_frames = flaky  # type: ignore[method-assign]
    assert not clock.tick()
    assert clock.errors == 1
    assert clock.tick()
    assert clock.frames == 1


def test_bad_fps(context: AppContext) -> None:
    with pytest.raises(ValueError):
        PlayClock(context, fps=0)


# ------------------------------------------------------ end-to-end in window


def test_keyboard_play_in_main_window_moves_the_car(qtbot: QtBot, context: AppContext) -> None:
    w = MainWindow(context)
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    assert not w.game_panel.play_button.isEnabled()
    context.connection.connect("fake", seed=3)
    assert w.game_panel.play_button.isEnabled()
    assert w.game_panel.play_button.text() == "Pause"
    assert w.game_panel.clock.running

    qtbot.keyPress(w, Qt.Key.Key_Up)  # the window forwards keys to the pad
    qtbot.waitUntil(lambda: game(context).game.state.speed >= 2)
    qtbot.keyRelease(w, Qt.Key.Key_Up)
    assert game(context).held == frozenset()
    assert game(context).game.state.position > 0

    qtbot.mouseClick(w.game_panel.play_button, Qt.MouseButton.LeftButton)
    assert w.game_panel.play_button.text() == "Play"
    assert not w.game_panel.clock.running
    frame = game(context).game.state.frame
    qtbot.wait(40)
    assert game(context).game.state.frame == frame
    qtbot.waitUntil(lambda: w.game_panel.view.frame is not None)
    w.close()


def test_clock_feeds_the_observation_service(clock: PlayClock, context: AppContext) -> None:
    """T-202: every tick records a memory snapshot and a diff."""
    context.connection.connect("fake")
    clock.pause()
    assert context.observation.latest is None
    clock.tick(3)
    obs = context.observation
    assert obs.frame == 3 and obs.latest is not None and obs.latest.frame == 3
    assert obs.latest.get("ram") == game(context).game.ram
    clock.tick(1)
    assert len(obs.ring) == 2 and obs.tracker.transitions == 1


def test_context_owns_a_discovery_service(context: AppContext) -> None:
    """T-306: the app context wires discovery to the observation service."""
    context.connection.connect("fake")
    assert context.discovery.enabled and context.discovery.knowledge is not None
