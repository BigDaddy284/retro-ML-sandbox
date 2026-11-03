"""T-010: the play clock advances frames while playing (T-818: off the GUI thread)."""

from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtCore import Qt, QTimer
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
    qtbot.waitUntil(lambda: clock.frames >= 5)  # the count lands when the step reports back
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
    qtbot.waitUntil(lambda: game(context).held == frozenset())  # the release goes by a worker
    assert game(context).game.state.position > 0

    qtbot.mouseClick(w.game_panel.play_button, Qt.MouseButton.LeftButton)
    assert w.game_panel.play_button.text() == "Play"
    assert not w.game_panel.clock.running
    qtbot.waitUntil(lambda: not w.game_panel.clock.in_flight)  # the step underway finishes
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


# ----------------------------------------------------- T-818 off the GUI thread


class _SlowStep:
    """Makes the fake adapter's step wait for a gate, and records its thread."""

    def __init__(self, adapter: FakeEmulatorAdapter) -> None:
        self.gate = threading.Event()
        self.threads: list[int] = []
        original = adapter._do_advance_frames

        def slow(n: int) -> None:
            self.threads.append(threading.get_ident())
            self.gate.wait(5)
            original(n)

        adapter._do_advance_frames = slow  # type: ignore[method-assign]


def _paused(clock: PlayClock, context: AppContext) -> _SlowStep:
    context.connection.connect("fake")
    clock.pause()
    return _SlowStep(game(context))


def test_pulse_steps_off_the_gui_thread_and_reports_on_it(
    qtbot: QtBot, clock: PlayClock, context: AppContext
) -> None:
    slow = _paused(clock, context)
    slow.gate.set()
    assert clock.pulse()
    assert clock.in_flight and clock.frames == 0
    qtbot.waitUntil(lambda: clock.frames == 1)
    assert not clock.in_flight
    assert slow.threads and slow.threads[0] != threading.get_ident()
    assert game(context).game.state.frame == 1
    obs = context.observation
    assert obs.frame == 1 and obs.latest is not None and obs.latest.frame == 1


def test_pulse_skips_while_a_step_is_in_flight(
    qtbot: QtBot, clock: PlayClock, context: AppContext
) -> None:
    slow = _paused(clock, context)
    assert clock.pulse()
    assert clock.pulse() is False
    assert clock.pulse() is False
    assert clock.skipped == 2
    slow.gate.set()
    qtbot.waitUntil(lambda: clock.frames == 1)
    assert len(slow.threads) == 1 and game(context).game.state.frame == 1
    assert clock.pulse()
    qtbot.waitUntil(lambda: clock.frames == 2)


def test_gui_events_keep_flowing_while_a_step_is_slow(
    qtbot: QtBot, clock: PlayClock, context: AppContext
) -> None:
    """The user's complaint: the window must not wait for the emulator."""
    slow = _paused(clock, context)
    clock.play()
    qtbot.waitUntil(lambda: clock.in_flight)
    fired: list[float] = []
    started = time.perf_counter()
    QTimer.singleShot(0, lambda: fired.append(time.perf_counter() - started))
    qtbot.waitUntil(lambda: bool(fired))
    assert fired[0] < 0.1 and clock.in_flight  # the step is still behind the gate
    slow.gate.set()
    qtbot.waitUntil(lambda: clock.frames >= 1)
    clock.pause()


def test_a_step_finishing_after_pause_is_not_counted(
    qtbot: QtBot, clock: PlayClock, context: AppContext
) -> None:
    slow = _paused(clock, context)
    assert clock.pulse()
    clock.play()  # running again...
    clock.pause()  # ...and paused while the step is still underway
    slow.gate.set()
    qtbot.waitUntil(lambda: not clock.in_flight)
    assert clock.frames == 0  # the emulator finished its frame, but the clock moved on
    assert game(context).game.state.frame == 1
    assert clock.pulse()
    qtbot.waitUntil(lambda: clock.frames == 1)


def test_drop_found_on_the_worker_stops_the_clock(
    qtbot: QtBot, clock: PlayClock, context: AppContext
) -> None:
    context.connection.connect("fake", drop_after_ops=0)
    assert clock.running
    assert clock.pulse()
    qtbot.waitUntil(lambda: not clock.running)
    assert not context.connection.is_connected
    assert clock.frames == 0


def test_transient_error_on_the_worker_is_counted(
    qtbot: QtBot, clock: PlayClock, context: AppContext
) -> None:
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
    assert clock.pulse()
    qtbot.waitUntil(lambda: clock.errors == 1)
    assert clock.frames == 0
    assert clock.pulse()
    qtbot.waitUntil(lambda: clock.frames == 1)
    assert clock.errors == 1


def test_context_owns_a_discovery_service(context: AppContext) -> None:
    """T-306: the app context wires discovery to the observation service."""
    context.connection.connect("fake")
    assert context.discovery.enabled and context.discovery.knowledge is not None


# -------------------------------------------- T-823 one request per frame


def test_pulse_is_one_bundled_request_on_a_socket_backend(qtbot: QtBot) -> None:
    from retro_ai_lab.emulator.bizhawk.simulator import BizHawkSimulator

    with BizHawkSimulator() as sim:
        context = AppContext()
        clock = PlayClock(context, fps=100, screenshot_every=2)
        clock.pause()
        context.connection.connect("bizhawk", port=sim.port, call_timeout_s=2.0, timeout_s=2.0)
        held = frozenset({b for b in context.connection.session.buttons if b.name == "up"})  # type: ignore[union-attr]
        clock.set_inputs(lambda: held)
        frames: list[object] = []
        clock.frame_captured.connect(frames.append)
        before = len(sim.requests)
        assert clock.pulse()
        qtbot.waitUntil(lambda: clock.frames == 1)
        assert len(sim.requests) == before + 1
        request = sim.requests[-1]
        assert request["cmd"] == "step" and request["frames"] == 1
        assert request["buttons"]["Up"] is True and not any(
            held for label, held in request["buttons"].items() if label != "Up"
        )
        assert [r["domain"] for r in request["reads"]] == ["RAM"]  # the emulator's name
        assert request["screenshot"] is True and len(frames) == 1
        obs = context.observation
        assert obs.latest is not None and obs.latest.frame == 1
        assert obs.latest.get("ram") == sim.game.ram
        assert clock.pulse()  # the next pulse: no screenshot, still one request
        qtbot.waitUntil(lambda: clock.frames == 2)
        assert len(sim.requests) == before + 2 and "screenshot" not in sim.requests[-1]
        assert len(frames) == 1 and clock.frames_captured == 1
        assert sim.game.state.speed == 2  # "Up" was held during both frames
        context.connection.disconnect()


def test_step_without_a_pad_sends_no_buttons(qtbot: QtBot) -> None:
    from retro_ai_lab.emulator.bizhawk.simulator import BizHawkSimulator

    with BizHawkSimulator() as sim:
        context = AppContext()
        clock = PlayClock(context, fps=100)
        clock.pause()
        context.connection.connect("bizhawk", port=sim.port, call_timeout_s=2.0, timeout_s=2.0)
        clock.set_inputs(lambda: None)
        assert clock.tick()
        assert sim.requests[-1]["cmd"] == "step" and "buttons" not in sim.requests[-1]
        clock.set_inputs(None)
        assert clock.tick()
        assert "buttons" not in sim.requests[-1]
        context.connection.disconnect()


def test_capabilities_decide_what_rides_along(
    qtbot: QtBot, clock: PlayClock, context: AppContext
) -> None:
    context.connection.connect("fake", capabilities=Capabilities(frame_advance=True))
    clock.pause()
    clock.set_inputs(lambda: frozenset())
    frames: list[object] = []
    clock.frame_captured.connect(frames.append)
    assert clock.tick(2)
    assert clock.pulse()
    qtbot.waitUntil(lambda: clock.frames == 3)
    assert clock.errors == 0 and frames == [] and clock.frames_captured == 0
    assert context.observation.latest is None  # no memory to read
    assert game(context).game.state.frame == 3


def test_screenshots_come_on_every_kth_pulse(qtbot: QtBot, context: AppContext) -> None:
    clock = PlayClock(context, fps=100, screenshot_every=3)
    clock.pause()
    context.connection.connect("fake")
    frames: list[object] = []
    clock.frame_captured.connect(frames.append)
    for _ in range(7):
        assert clock.pulse()
        qtbot.waitUntil(lambda: not clock.in_flight)
    assert clock.frames == 7 and [f.number for f in frames] == [1, 4, 7]  # type: ignore[attr-defined]
    assert clock.frames_captured == 3
    context.connection.disconnect()
    context.connection.connect("fake")  # the count starts over on connect
    assert clock.pulse()
    qtbot.waitUntil(lambda: not clock.in_flight)
    assert clock.frames_captured == 1 and clock.frames == 1


def test_bad_screenshot_every(context: AppContext) -> None:
    with pytest.raises(ValueError):
        PlayClock(context, fps=10, screenshot_every=0)


def test_tick_feeds_the_snapshot_from_the_same_step(clock: PlayClock, context: AppContext) -> None:
    context.connection.connect("fake")
    clock.pause()
    adapter = game(context)
    reads = {"n": 0}
    original = adapter._do_read_many

    def counting(reads_: object) -> list[bytes]:
        reads["n"] += 1
        return original(reads_)  # type: ignore[arg-type]

    adapter._do_read_many = counting  # type: ignore[method-assign]
    assert clock.tick(3)
    obs = context.observation
    assert obs.latest is not None and obs.latest.frame == 3 and reads["n"] == 1
    assert obs.latest.get("ram") == adapter.game.ram


# ----------------------------------------------- T-828 a failing step is said once


def test_a_failing_step_is_reported_once_until_the_message_changes(
    qtbot: QtBot, clock: PlayClock, context: AppContext, caplog: pytest.LogCaptureFixture
) -> None:
    from retro_ai_lab.core import EmulatorError

    context.connection.connect("fake")
    clock.pause()
    adapter = game(context)
    messages = ["hiccup", "hiccup", "hiccup", "worse", None, "hiccup"]
    original = adapter._do_advance_frames

    def flaky(n: int) -> None:
        message = messages.pop(0)
        if message is not None:
            raise EmulatorError(message)
        original(n)

    adapter._do_advance_frames = flaky  # type: ignore[method-assign]
    said: list[str] = []
    clock.step_failed.connect(said.append)
    with caplog.at_level("WARNING", logger="retro_ai_lab.ui.play_clock"):
        for _ in range(4):  # hiccup three times, then worse
            assert clock.pulse()
            qtbot.waitUntil(lambda: not clock.in_flight)
        assert clock.errors == 4 and said == ["hiccup", "worse"]
        assert clock.last_failure == "worse"
        assert clock.tick()  # a good step clears the slate...
        assert clock.last_failure is None
        assert not clock.tick()  # ...so the same old message is news again
    assert said == ["hiccup", "worse", "hiccup"]
    assert [r.getMessage() for r in caplog.records] == [
        "Frame step failed: hiccup",
        "Frame step failed: worse",
        "Frame step failed: hiccup",
    ]
