"""T-505: the training service (modes, start / pause / stop, history)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import pytest

from retro_ai_lab.core import (
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
)
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.core.worklog import CallStats, app_work
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import Settings
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    GoalService,
    GoStarted,
    ObservationService,
    RewardService,
)
from retro_ai_lab.storage import BrainStore, GoalStore, RewardStore
from retro_ai_lab.training import (
    DEFAULT_MAX_STEPS,
    DEFAULT_UNSTICK_AFTER,
    Brain,
    EpisodeFinished,
    Mode,
    Speed,
    Status,
    TrainingService,
    TrainingStateChanged,
)
from retro_ai_lab.understanding import Encoding, Evidence, Knowledge, Variable, VariableKind
from retro_ai_lab.understanding.events import CYCLES, EventKind, GameEvent
from retro_ai_lab.understanding.wakeup import InputProbe

WENT_HOME = "Lives, Lap all went back to their starting values."

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]


@dataclass
class Lab:
    bus: EventBus
    connection: ConnectionService
    observation: ObservationService
    discovery: DiscoveryService
    rewards: RewardService
    goals: GoalService
    training: TrainingService
    finished: list[EpisodeFinished]
    changes: list[TrainingStateChanged]

    def adapter(self) -> FakeEmulatorAdapter:
        a = self.connection.adapter
        assert isinstance(a, FakeEmulatorAdapter)
        return a

    def connect(self, **kw: object) -> Knowledge:
        self.connection.connect("fake", seed=3, **kw)  # type: ignore[arg-type]
        return teach(self)

    def play(self, buttons: list[str], frames: int) -> None:
        """What the play clock and the pad do when the child plays."""
        self.observation.record_input(buttons)
        a = self.adapter()
        held = [b for b in a.buttons() if b.name in buttons]
        a.set_input(held)
        for _ in range(frames):
            a.advance_frames(1)
            self.observation.advance(1)


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
    rewards = RewardService(connection, discovery, bus, store=RewardStore(tmp_path / "r"))
    goals = GoalService(connection, discovery, bus, store=GoalStore(tmp_path / "g"))
    training = TrainingService(
        connection,
        observation,
        discovery,
        bus,
        frames_per_step=4,
        max_steps=50,
        settings=Settings(epsilon=0.5),
        seed=1,
        store=BrainStore(tmp_path / "b"),
        lookahead=1,  # these tests count updates step by step; the lookahead has its own (T-963)
        rewards=rewards,
    )
    finished: list[EpisodeFinished] = []
    changes: list[TrainingStateChanged] = []
    bus.subscribe(EpisodeFinished, finished.append)
    bus.subscribe(TrainingStateChanged, changes.append)
    return Lab(bus, connection, observation, discovery, rewards, goals, training, finished, changes)


def teach(lab: Lab) -> Knowledge:
    k = lab.discovery.knowledge
    assert k is not None
    added = (
        k.propose(RAM, LAYOUT.lane, kind=VariableKind.POSITION, name="Lane", evidence=SURE),
        k.propose(RAM, LAYOUT.speed, kind=VariableKind.SPEED, name="Speed", evidence=SURE),
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
        k.propose(RAM, LAYOUT.game_over, kind=VariableKind.FLAG, name="Game over", evidence=SURE),
    )
    lab.bus.publish(DiscoveryUpdated(added, (), (), ()))
    return k


# ------------------------------------------------------------ idle & refusals


def test_idle_until_a_capable_game_is_connected(lab: Lab) -> None:
    t = lab.training
    assert not t.enabled and t.status is Status.IDLE and t.agent is None
    assert t.summary() == "Connect a game to start training."
    assert not t.step() and t.episode is None and t.last is None
    with pytest.raises(ConfigurationError, match="Can't train yet"):
        t.start()
    lab.connection.connect(
        "fake", capabilities=Capabilities(screen_capture=True, frame_advance=True, input=True)
    )
    assert not t.enabled  # no memory: nothing to learn from
    lab.connection.disconnect()
    lab.connect()
    assert t.enabled and t.agent is not None and t.space is not None and t.encoder is not None
    assert t.space.labels[0] == "do nothing"
    assert t.summary() == "Ready to train."
    assert t.mode is Mode.GUIDED and t.speed is Speed.LIVE and not t.ai_in_control


def test_bad_settings_and_misuse_are_refused(lab: Lab) -> None:
    for kw in ({"frames_per_step": 0}, {"max_steps": 0}, {"history": 0}):
        with pytest.raises(ConfigurationError):
            TrainingService(lab.connection, lab.observation, lab.discovery, lab.bus, **kw)  # type: ignore[arg-type]
    t = lab.training
    lab.connect()
    with pytest.raises(ConfigurationError):
        t.pause()
    with pytest.raises(ConfigurationError):
        t.resume()
    with pytest.raises(ConfigurationError):
        t.stop()
    with pytest.raises(ConfigurationError):
        t.set_mode("manual")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        t.set_speed("fast")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        t.start("guided")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        t.run_for(-1)
    with pytest.raises(ConfigurationError):
        t.run_for(1, max_steps=0)
    with pytest.raises(ConfigurationError):
        t.train(0)
    t.set_mode(Mode.MANUAL)
    with pytest.raises(ConfigurationError, match="you are the player"):
        t.train(1)
    t.start()
    with pytest.raises(ConfigurationError, match="already going"):
        t.start()
    with pytest.raises(ConfigurationError):
        t.resume()


# ------------------------------------------------------------ AI modes


def test_independent_run_records_goes_and_stops_cleanly(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    stepped = app_work.stats.get("ai step", CallStats()).count
    t.start(Mode.INDEPENDENT)
    assert t.status is Status.RUNNING and t.mode is Mode.INDEPENDENT and t.ai_in_control
    assert not t.mode.allows_guidance and Mode.GUIDED.allows_guidance
    stages = ("step: choose", "step: emulator", "step: observe", "step: encode", "step: learn")
    staged = {name: app_work.stats.get(name, CallStats()).count for name in stages}
    assert t.step() and t.episode is not None and t.episode.steps == 1
    assert app_work.stats["ai step"].count == stepped + 1  # T-894: timed for the log
    for name in stages:  # T-947: and every stage of it
        assert app_work.stats[name].count == staged[name] + 1, name
    assert "Go 1: playing, step 1" in t.summary()
    while t.episodes < 2:
        assert t.step()
    assert [f.record.number for f in lab.finished] == [1, 2]
    first = lab.finished[0].record
    assert first.mode is Mode.INDEPENDENT and first.result.steps == 50
    assert first.result.ended_by == "max_steps" and first.frame == 200
    assert t.history == tuple(f.record for f in lab.finished)
    assert t.last == lab.finished[1].record and t.last.number == 2
    assert t.total_steps == sum(f.record.result.steps for f in lab.finished)
    t.step()
    t.step()
    assert t.episode is not None and t.episode.steps == 2
    t.stop()
    assert t.status is Status.IDLE and not t.ai_in_control and t.episode is None
    assert t.last is not None and t.last.result.ended_by == "stopped"
    assert t.last.result.steps == 2 and t.episodes == 3
    assert lab.observation.inputs.held_at(lab.observation.frame) == frozenset()
    assert t.summary() == "Resting after 3 goes."
    assert not t.step()
    assert [c.reason for c in lab.changes] == ["start", "stop"]
    assert lab.changes[-1].status is Status.IDLE


def test_pause_holds_the_go_and_resume_carries_on(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    t.start()
    for _ in range(5):
        t.step()
    t.pause()
    assert t.status is Status.PAUSED and t.ai_in_control
    assert "paused, step 5" in t.summary()
    frame = lab.observation.frame
    assert not t.step() and lab.observation.frame == frame
    t.resume()
    assert t.step() and t.episode is not None and t.episode.steps == 6
    assert [c.reason for c in lab.changes] == ["start", "pause", "resume"]


def test_goes_restart_from_the_checkpoint_and_train_runs_whole_goes(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    start = lab.adapter().game.ram
    records = t.train(3)
    assert len(records) == 3 and [r.number for r in records] == [1, 2, 3]
    assert t.status is Status.IDLE and t.episodes == 3
    assert all(r.result.ended_by in ("max_steps", "reset") for r in records)
    assert all(r.result.steps > 1 for r in records)  # the restore itself ends nothing
    assert lab.adapter().game.ram != start
    # Every go begins from the checkpoint taken at start().
    here = lab.adapter().game.ram
    t.start()
    while t.episodes < 4:
        t.step()
    t.step()  # begins go 5 from the checkpoint
    assert t.episode is not None and t.episode.steps == 1
    assert lab.observation.ring[-2].get(RAM) == here
    t.stop()
    # A paused service resumes inside train() and stops afterwards.
    t.start()
    t.pause()
    assert len(t.train(1)) == 1 and t.status is Status.IDLE
    assert t.train(1)[0].number == 7


def test_run_for_honours_its_budget(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    t.set_speed(Speed.FAST)
    assert t.speed is Speed.FAST and Speed.FAST.label == "Practise fast"
    assert t.run_for(1.0) == 0  # not started
    t.start()
    assert t.run_for(5.0, max_steps=7) == 7 and t.total_steps == 7
    assert t.run_for(0.0) == 1  # the deadline is checked after each step
    taken = t.run_for(0.05)
    assert taken >= 1
    t.stop()


def test_the_gui_buttons_wait_for_a_step_on_another_thread(lab: Lab) -> None:
    """The Learning panel steps on a worker (T-903); Stop must not cut a step in half."""
    import threading

    lab.connect()
    t = lab.training
    t.set_speed(Speed.FAST)
    t.start(Mode.INDEPENDENT)
    taken: list[int] = []
    failures: list[BaseException] = []

    def work() -> None:
        try:
            taken.append(t.run_for(0.3))
        except BaseException as exc:  # the test reports it
            failures.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    while t.total_steps == 0:  # the worker is mid-slice
        pass
    t.stop()  # waits for the slice, then closes the go once
    worker.join(timeout=5.0)
    assert not worker.is_alive() and failures == [] and taken and taken[0] >= 1
    assert t.status is Status.IDLE and t.episode is None
    # Stop closed the go that was running, or the slice had just ended one (by its length,
    # or stuck when the slice was long enough for nothing to change for 25 steps): never
    # half a go.
    assert t.episodes == len(lab.finished)
    assert lab.finished[-1].record.result.ended_by in ("stopped", "max_steps", "stuck")
    assert t.total_steps == sum(f.record.result.steps for f in lab.finished)
    assert not t.step()  # stopped: the next step takes nothing


def test_fast_practice_runs_the_emulator_flat_out(lab: Lab) -> None:
    """T-836: "Practise fast" asks the emulator for its top speed, and only then."""
    from retro_ai_lab.training.service import FAST_SPEED

    lab.connect()
    t = lab.training
    a = lab.adapter()
    assert a.speed == 1.0 and FAST_SPEED == 64.0
    t.set_speed(Speed.FAST)
    assert a.speed == 1.0  # not running yet
    assert lab.changes[-1].reason == "speed"
    t.set_speed(Speed.FAST)  # no change, no announcement
    assert lab.changes[-1].reason == "speed" and sum(c.reason == "speed" for c in lab.changes) == 1
    t.start()
    assert a.speed == FAST_SPEED
    t.pause()
    assert a.speed == 1.0
    t.resume()
    assert a.speed == FAST_SPEED
    t.set_speed(Speed.LIVE)
    assert a.speed == 1.0
    t.set_speed(Speed.FAST)
    assert a.speed == FAST_SPEED
    t.set_mode(Mode.MANUAL)  # the child plays: no speeding up
    assert a.speed == 1.0
    t.set_mode(Mode.INDEPENDENT)
    assert a.speed == FAST_SPEED
    t.stop()
    assert a.speed == 1.0
    # Manual mode never speeds the game up, whatever the speed setting says.
    t.start(Mode.MANUAL)
    assert a.speed == 1.0
    t.stop()


def test_a_refused_speed_call_does_not_stop_training(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab.connect()
    t = lab.training
    a = lab.adapter()

    def refuse(multiplier: float) -> None:
        raise EmulatorError("no")

    monkeypatch.setattr(a, "_do_set_speed", refuse)
    t.set_speed(Speed.FAST)
    t.start()
    assert t.status is Status.RUNNING and a.speed == 1.0
    assert t.step()
    t.stop()


def test_an_emulator_without_speed_control_is_left_alone(lab: Lab) -> None:
    caps = replace(Capabilities.all(), speed_control=False)
    lab.connection.connect("fake", seed=3, capabilities=caps)  # type: ignore[arg-type]
    teach(lab)
    t = lab.training
    t.set_speed(Speed.FAST)
    t.start()
    assert t.status is Status.RUNNING and lab.adapter().speed == 1.0
    t.stop()


def test_training_holds_the_button_experiments_back(lab: Lab) -> None:
    """T-839: no automatic experiments over the link while a go is running."""
    lab.connect()
    t = lab.training
    assert lab.discovery.experiments_allowed
    t.start()
    assert not lab.discovery.experiments_allowed
    t.pause()
    assert not lab.discovery.experiments_allowed  # the game is still the AI's
    t.resume()
    t.stop()
    assert lab.discovery.experiments_allowed
    t.start(Mode.MANUAL)  # the child drives the same link
    assert not lab.discovery.experiments_allowed
    t.stop()
    assert lab.discovery.experiments_allowed


def test_due_experiments_run_between_goes_not_mid_go(tmp_path: Path) -> None:
    """T-839: a due experiment runs as the next go begins, never inside a go."""
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    discovery = DiscoveryService(connection, observation, bus, experiment_every=30)
    training = TrainingService(
        connection, observation, discovery, bus, frames_per_step=4, max_steps=8, seed=1
    )
    lab = Lab(bus, connection, observation, discovery, None, None, training, [], [])  # type: ignore[arg-type]
    lab.connect()
    training.start()
    seen = 0
    ran_at: list[int] = []
    for _ in range(200):
        training.step()
        if discovery.experiments > seen:
            seen = discovery.experiments
            episode = training.episode
            assert episode is not None
            ran_at.append(episode.steps)
    training.stop()
    assert seen >= 3, "experiments still happen while training"
    assert ran_at and all(steps == 1 for steps in ran_at), ran_at  # the first step of a go
    assert not discovery.experiment_due or not discovery.experiments_allowed


def test_every_go_announces_its_start(lab: Lab) -> None:
    """T-878: the runner says when a go begins, before any step, so rewards start afresh."""
    lab.connect()
    seen: list[GoStarted] = []
    lab.bus.subscribe(GoStarted, seen.append)
    t = lab.training
    t.start()
    t.step()  # the runner is built on the first step
    assert len(seen) == 1 and t.episodes == 0
    while t.episodes < 2:
        t.step()
    t.step()
    assert len(seen) == 3  # the two finished goes and the one under way
    t.stop()


def test_the_go_log_line_names_the_earners(lab: Lab, caplog: pytest.LogCaptureFixture) -> None:
    """T-857: a go's line says which rewards paid its points, and only when it earned any."""
    k = lab.connect()
    score = k.at("ram", LAYOUT.score)
    profile = lab.rewards.profile
    assert score is not None and profile is not None
    jump = profile.find(f"score_gain:{score.id}")
    assert jump is not None
    t = lab.training
    caplog.set_level(logging.INFO, logger="retro_ai_lab.training.service")
    t.start()
    assert t.step()
    gain = GameEvent(EventKind.SCORE_GAIN, 5, score.id, "")
    lab.bus.publish(DiscoveryUpdated((), (), (gain,), ()))
    lab.bus.publish(DiscoveryUpdated((), (), (gain,), ()))
    lives = k.at("ram", LAYOUT.lives)
    checkpoint = k.at("ram", LAYOUT.checkpoint)
    assert lives is not None and checkpoint is not None
    crash = GameEvent(EventKind.SETBACK, 6, lives.id, "", 3, 2)
    step = GameEvent(EventKind.PROGRESS, 7, checkpoint.id, "", 0, 1)
    lab.bus.publish(DiscoveryUpdated((), (), (crash, step), ()))
    reset = GameEvent(EventKind.RESET, 99, None, WENT_HOME)
    lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))
    t.step()  # the go ends on the reset
    messages = [r.getMessage() for r in caplog.records]
    line = next(m for m in messages if m.startswith("Go 1 ended by reset"))
    assert ", from " in line and f"{jump.name} {2 * jump.value:+,.1f} (2 times), " in line, line
    assert line.endswith(f": {WENT_HOME} (frame 99)")
    # Every earner is named, none folded into "and N more" (T-877): the checkpoint step
    # also completed a goal step, so there are five.
    assert len(t.history[-1].result.earned) == 5 and " more" not in line, line
    for name in (profile.get(f"setback:{lives.id}").name, "Starting over"):
        assert name in line, line
    # Survival in the log (T-934): the losses sit between the earners and the reset's why.
    assert ", lost a life once, the first at step 2: " in line, line
    assert jump.id in [e.id for e in t.history[-1].result.earned]
    while t.episodes < 2:
        t.step()
    line = next(m for m in caplog.records if m.getMessage().startswith("Go 2 ")).getMessage()
    assert (", from " in line) == bool(t.history[-1].result.earned), line
    t.stop()


def test_a_go_s_restore_is_not_counted_as_a_change(lab: Lab) -> None:
    """T-935: the user's lives warning counted the AI's own restores as the lives going up."""
    restores = 0
    real = lab.observation.expect_jump

    def spy() -> None:
        nonlocal restores
        restores += 1
        real()

    lab.observation.expect_jump = spy  # type: ignore[method-assign]
    lab.connect()  # the harness is built at connection, with the spy
    lab.training.start()
    while len(lab.finished) < 12:
        lab.training.step()
    lab.training.stop()
    assert restores >= len(lab.finished) - 1  # every go after the first starts from the checkpoint
    tracker = lab.observation.tracker
    assert tracker.observed - 1 - tracker.transitions >= len(lab.finished) - 1  # skipped
    lives = tracker.stats("ram", LAYOUT.lives)
    assert any(f.record.result.setbacks for f in lab.finished), "no life was lost to put back"
    assert lives.ups == 0 and lives.downs > 0  # lost in play, never "gained" by a restore


def test_the_survival_line_every_ten_goes(lab: Lab, caplog: pytest.LogCaptureFixture) -> None:
    """T-934: from the twentieth go, every tenth go compares the last ten with the ten before."""
    lab.connect()
    t = lab.training
    caplog.set_level(logging.INFO, logger="retro_ai_lab.training.service")
    t.train(21)
    survival = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Survival")]
    assert len(survival) == 1 and survival[0].startswith("Survival over the last 10 goes: ")
    assert "lives lost per go" in survival[0] and "the first life lost after" in survival[0]
    t.train(9)
    survival = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Survival")]
    assert len(survival) == 2


def test_the_log_says_why_each_go_ended(lab: Lab, caplog: pytest.LogCaptureFixture) -> None:
    """T-840: the user's log file shows how each go ended and what a reset blamed."""
    lab.connect()
    t = lab.training
    caplog.set_level(logging.INFO, logger="retro_ai_lab.training.service")
    t.start()
    assert t.step()
    reset = GameEvent(EventKind.RESET, 99, None, WENT_HOME)
    lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))
    t.step()  # the go ends on the reset
    assert t.episodes == 1
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith("Go 1 ended by reset after 2 steps (8 frames), ")
        and m.endswith(f": {WENT_HOME} (frame 99)")
        for m in messages
    ), messages
    while t.episodes < 2:
        t.step()
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith("Go 2 ended by max_steps after 50 steps") and "(frame" not in m
        for m in messages
    ), messages
    t.stop()


def test_checkpoint_can_be_switched_off(lab: Lab) -> None:
    t = TrainingService(
        lab.connection, lab.observation, lab.discovery, lab.bus, max_steps=5, checkpoint=False
    )
    lab.connect()
    t.start()
    t.step()
    assert t._harness is None
    first = lab.adapter().game.ram
    while t.episodes < 1:
        t.step()
    t.step()  # the next go carries on from wherever the game is
    assert lab.adapter().game.ram != first
    t.stop()


# ------------------------------------------------------------ manual mode


def test_manual_mode_learns_from_the_childs_play(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    agent = t.agent
    assert agent is not None
    t.start(Mode.MANUAL)
    assert t.status is Status.RUNNING and not t.ai_in_control
    assert t.episode is None  # begins on the first frame the child plays
    lab.play(["up"], 3)
    assert t.episode is not None and agent.updates == 0  # not a full step yet
    learned = app_work.stats.get("learn", CallStats()).count
    lab.play(["up"], 1)
    assert agent.updates == 1 and t.total_steps == 1 and t.episode.steps == 1
    assert app_work.stats["learn"].count == learned + 1  # T-894: timed for the log
    space = t.space
    assert space is not None and space.matches(["up"]).buttons == frozenset({"up"})
    assert len(agent.table) >= 1  # the situation the child was in is now known
    assert "watching you play, step 1" in t.summary()
    lab.play(["up", "a"], 8)
    assert agent.updates == 3
    assert not any(e.source == "ai" for e in lab.observation.inputs)
    # Paused: the child keeps playing, the AI stops learning.
    t.pause()
    lab.play(["left"], 8)
    assert agent.updates == 3
    t.resume()
    lab.play(["left"], 4)
    assert agent.updates == 4
    # Losing every life restarts the game, which closes the go; the next one begins.
    game = lab.adapter().game
    while not game.state.game_over:  # the child crashes out, off camera
        game.step(["up"])
    lab.play(["up"], 4)  # the engine sees the game-over screen
    for _ in range(10):
        lab.play(["start"], 4)  # and the child starts again
        if lab.finished:
            break
    assert len(lab.finished) == 1
    assert lab.finished[0].record.result.ended_by == "reset"
    assert lab.finished[0].record.result.steps < 50
    assert lab.finished[0].record.mode is Mode.MANUAL
    assert not game.state.game_over
    lab.play(["up"], 4)
    assert t.episode is not None and t.episode.steps == 1
    t.stop()
    assert t.episodes == 2 and t.last is not None and t.last.result.ended_by == "stopped"


def test_switching_mode_mid_go_closes_it_and_carries_on(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    t.start(Mode.GUIDED)
    t.step()
    t.set_mode(Mode.GUIDED)  # no change, no announcement
    t.set_mode(Mode.MANUAL)
    assert t.status is Status.RUNNING and t.mode is Mode.MANUAL and not t.ai_in_control
    assert t.episodes == 1 and t.last is not None and t.last.result.ended_by == "mode change"
    assert not t.step()  # the AI doesn't play in manual mode
    lab.play(["up"], 4)
    assert t.episode is not None and t.episode.steps == 1
    t.set_mode(Mode.INDEPENDENT)
    assert t.episodes == 2 and t.ai_in_control
    assert t.step() and t.episode is not None and t.episode.steps == 1
    t.stop()
    assert [c.reason for c in lab.changes] == ["start", "mode change", "mode change", "stop"]
    assert lab.changes[1].mode is Mode.MANUAL
    # Changing mode while idle is just remembered.
    t.set_mode(Mode.GUIDED)
    assert t.mode is Mode.GUIDED and t.episodes == 3
    assert Mode.MANUAL.label == "You play" and "learns from you" in Mode.MANUAL.description


# ------------------------------------------------------------ link loss


def test_link_loss_closes_the_go_and_reconnect_starts_afresh(lab: Lab) -> None:
    # Enough link for the look for itself on the screen (T-974) and part of a go.
    lab.connect(drop_after_ops=400)
    t = lab.training
    agent = t.agent
    t.start()
    while lab.connection.is_connected:
        t.step()
    assert t.status is Status.IDLE and not t.enabled
    assert t.episodes == 1 and t.last is not None
    assert t.last.result.ended_by == "disconnected"
    assert [c.reason for c in lab.changes] == ["start", "disconnected"]
    assert not t.step()
    with pytest.raises(ConfigurationError):
        t.start()
    assert t.agent is agent  # kept until the next game
    lab.connect()
    assert t.enabled and t.agent is not agent and t.episodes == 1  # loaded from disk (T-506)
    assert t.summary() == "Resting after 1 go."
    # A disconnect while idle is quiet.
    lab.connection.disconnect()
    assert [c.reason for c in lab.changes] == ["start", "disconnected"]


def test_link_dropping_during_the_checkpoint_refuses_to_start(lab: Lab) -> None:
    lab.connect()
    t = lab.training

    class Flaky:
        def available(self) -> bool:
            return True

        def checkpoint(self) -> None:
            lab.connection.disconnect()
            raise EmulatorDisconnectedError()

    t._harness = Flaky()  # type: ignore[assignment]
    with pytest.raises(ConfigurationError, match="went away"):
        t.start()
    assert t.status is Status.IDLE and not t.enabled and lab.changes == []


def test_app_context_wires_the_training_service(tmp_path: Path) -> None:
    from retro_ai_lab.ui.context import AppContext

    context = AppContext(data_dir=tmp_path)
    assert isinstance(context.training, TrainingService)
    assert context.training.store is context.brain_store
    assert context.brain_store.folder == tmp_path / "brains"
    context.connection.connect("fake")
    assert context.training.enabled
    context.connection.disconnect()


# ------------------------------------------------------------ persistence (T-506)


def test_brain_is_saved_as_it_goes_and_restored_on_reconnect(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    store = t.store
    assert store is not None and t.game_id is not None and t.saves == 0
    assert not store.exists(t.game_id) and t.fresh_start == ""
    assert t.brain() is not None and t.brain().episodes == 0  # type: ignore[union-attr]
    t.train(2)
    assert t.saves == 2 and store.exists(t.game_id)
    t.start()
    t.step()
    assert t.save() == store.path_for(t.game_id) and t.saves == 3
    t.stop()  # closes a go, which saves
    assert t.saves == 4
    agent = t.agent
    assert agent is not None
    before = agent.to_dict()
    lab.connection.disconnect()
    assert t.saves == 5 and t.save() is None  # nothing to save once disconnected
    lab.connect()
    again = t.agent
    assert again is not None and again is not agent and again.to_dict() == before
    assert t.episodes == 3 and [r.number for r in t.history] == [1, 2, 3]
    assert t.total_steps == t.history[0].result.steps + t.history[1].result.steps + 1
    assert t.summary() == "Resting after 3 goes."
    # Later goes carry on the numbering.
    t.train(1)
    assert t.last is not None and t.last.number == 4 and t.episodes == 4


def test_the_log_says_whether_the_saved_brain_was_picked_up(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    """T-933: the user's log had five fresh brains and nothing saying why."""
    caplog.set_level(logging.INFO, logger="retro_ai_lab.training.service")
    lab.connect()
    t = lab.training
    store = t.store
    game_id = t.game_id
    assert store is not None and game_id is not None
    t.train(2)
    steps = t.total_steps
    lab.connection.disconnect()
    lab.connect()
    assert t.agent is not None
    exploring = round(t.agent.epsilon * 100)
    t.forget()
    lab.connection.disconnect()
    store.path_for(game_id).write_text("{", "utf-8")
    lab.connect()
    messages = [
        (r.levelname, r.getMessage())
        for r in caplog.records
        if not r.getMessage().startswith(
            ("Go ", "The brain after", "Found itself", "Not sure yet which thing")
        )
    ]
    assert messages == [
        ("INFO", "No saved brain for this game yet: the AI starts with a new one."),
        (
            "INFO",
            f"Picked up the saved brain for this game: 2 goes and {steps} steps learned so far; "
            f"exploring {exploring}% of the time.",
        ),
        ("INFO", "The AI forgot this game at your request: its brain and the saved copy are gone."),
        ("WARNING", t.fresh_start),
    ]
    assert "starting afresh" in t.fresh_start


def test_saved_brain_for_different_controls_or_damaged_file_starts_afresh(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    store = t.store
    game_id = t.game_id
    assert store is not None and game_id is not None
    t.train(1)
    lab.connection.disconnect()
    data = store.load(game_id)
    data["agent"]["space"]["actions"] = data["agent"]["space"]["actions"][:2]
    data["agent"]["table"] = {"actions": 2, "rows": {}}
    store.save(data)
    lab.connect()
    assert t.enabled and t.episodes == 0 and t.history == ()
    assert "different controls" in t.fresh_start
    lab.connection.disconnect()  # saves the fresh brain over the odd one
    assert Brain.from_dict(store.load(game_id)).episodes == 0
    store.path_for(game_id).write_text("{", "utf-8")
    lab.connect()
    assert t.enabled and "starting afresh" in t.fresh_start
    t.train(1)
    assert t.fresh_start != "" and t.episodes == 1
    lab.connection.disconnect()
    lab.connect()
    assert t.fresh_start == "" and t.episodes == 1


def test_forget_starts_from_scratch_and_deletes_the_file(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    store = t.store
    game_id = t.game_id
    assert store is not None and game_id is not None
    t.train(1)
    t.start()
    with pytest.raises(ConfigurationError, match="Stop training"):
        t.forget()
    t.stop()
    assert store.exists(game_id)
    old = t.agent
    t.forget()
    assert not store.exists(game_id) and t.agent is not old and t.agent is not None
    assert t.agent.updates == 0 and t.episodes == 0 and t.history == ()
    assert lab.changes[-1].reason == "forget" and t.summary() == "Ready to train."
    t.train(1)
    assert t.episodes == 1 and store.exists(game_id)
    lab.connection.disconnect()
    with pytest.raises(ConfigurationError, match="no brain"):
        t.forget()


def test_curve_reads_the_history(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    assert t.curve().attempts == 0 and t.curve().describe() == "No goes yet."
    t.train(3)
    c = t.curve(window=2)
    assert c.attempts == 3 and [p.number for p in c.points] == [1, 2, 3]
    assert c.rewards == tuple(r.result.reward for r in t.history)
    assert c.describe().startswith("3 goes so far")
    with pytest.raises(ConfigurationError):
        t.curve(window=0)


def test_behaviour_changes_and_experiments_follow_the_goes(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    assert t.behaviour_changes() == ((), 0, 0)
    assert t.experiments() == ()  # nothing to go on before any go
    t.train(1)
    changes, goes_ago, known = t.behaviour_changes()
    assert goes_ago == 0 and known > 0 and changes == ()  # one snapshot: nothing to compare
    t.train(3)
    changes, goes_ago, known = t.behaviour_changes(window=2)
    assert goes_ago == 2 and known >= 1
    assert all(c.state_key in t.agent.table.known_states() for c in changes)  # type: ignore[union-attr]
    changes, goes_ago, known = t.behaviour_changes(window=50)  # further back than we keep
    assert goes_ago == 3
    titles = [e.title for e in t.experiments(goal=lab.goals.goal, profile=lab.rewards.profile)]
    assert "Practise fast" in titles  # four goes: too early to see a trend
    # T-895: a confirmed guess whose byte behaves unlike its kind is something to check.
    k = lab.discovery.knowledge
    assert k is not None
    sworn = k.at("ram", LAYOUT.position) or k.propose("ram", LAYOUT.position, name="Position")
    k.confirm(k.update(sworn.id, kind=VariableKind.LIVES, name="Sworn lives").id)
    lab.discovery.review()  # the position changes nearly every step: no life count does
    assert [q.name for q in lab.discovery.questions] == ["Sworn lives"]
    assert t.experiments()[0].title == 'Check "Sworn lives"'
    assert "changes rarely" in t.experiments()[0].why
    t.forget()
    assert t.behaviour_changes() == ((), 0, 0)


def test_service_without_a_store_never_saves(tmp_path: Path, lab: Lab) -> None:
    t = TrainingService(lab.connection, lab.observation, lab.discovery, lab.bus, max_steps=5)
    lab.connect()
    assert t.store is None and t.game_id is not None
    t.train(1)
    assert t.saves == 0 and t.save() is None
    lab.connection.disconnect()
    assert t.saves == 0


# ------------------------------------------------------------ T-706: how the AI learns


def test_auto_starts_with_a_table_and_promotes_when_the_game_grows(
    lab: Lab, tmp_path: Path
) -> None:
    from retro_ai_lab.learning import AUTO, DQN, Q_TABLE, DQNLearner, QLearningAgent

    assert lab.training.algorithm == AUTO
    t = TrainingService(
        lab.connection,
        lab.observation,
        lab.discovery,
        lab.bus,
        frames_per_step=4,
        max_steps=50,
        settings=Settings(epsilon=0.5),
        seed=1,
        store=BrainStore(tmp_path / "auto"),
        promote_at=3,
    )
    changes: list[TrainingStateChanged] = []
    lab.bus.subscribe(TrainingStateChanged, changes.append)
    with pytest.raises(ConfigurationError):
        TrainingService(lab.connection, lab.observation, lab.discovery, lab.bus, promote_at=0)
    lab.connect()
    table = t.agent
    assert isinstance(table, QLearningAgent) and t.last_change == ""
    t.train(1)
    promoted = t.agent
    assert isinstance(promoted, DQNLearner) and promoted.algorithm == DQN
    assert promoted.episodes == 1 and promoted.updates == table.updates and promoted.known >= 3
    assert t.episodes == 1 and len(t.history) == 1
    assert [c.reason for c in changes if c.reason == "promoted"] == ["promoted"]
    assert "neural network" in t.last_change and str(promoted.known) in t.last_change
    assert t.algorithm == AUTO  # the setting is still "choose for me"
    store = t.store
    assert store is not None and t.game_id is not None
    assert Brain.from_dict(store.load(t.game_id)).algorithm == DQN
    t.train(1)
    assert t.agent is promoted and t.episodes == 2
    assert [c.reason for c in changes if c.reason == "promoted"] == ["promoted"]
    lab.connection.disconnect()
    lab.connect()
    again = t.agent
    assert isinstance(again, DQNLearner) and again.episodes == 2 and t.episodes == 2
    assert [c.reason for c in changes if c.reason == "promoted"] == ["promoted"]
    # A fixed choice never promotes.
    fixed = TrainingService(
        lab.connection, lab.observation, lab.discovery, lab.bus, max_steps=50, promote_at=3
    )
    fixed.set_algorithm(Q_TABLE)
    lab.connection.disconnect()
    lab.connect()
    fixed.train(1)
    assert isinstance(fixed.agent, QLearningAgent) and fixed.agent.known >= 3


def test_set_algorithm_converts_the_brain_either_way(lab: Lab) -> None:
    from retro_ai_lab.learning import AUTO, DQN, Q_TABLE, DQNLearner, QLearningAgent

    t = lab.training
    with pytest.raises(ConfigurationError, match="magic"):
        t.set_algorithm("magic")
    assert t.set_algorithm(DQN) == "" and t.algorithm == DQN  # no brain yet: nothing to convert
    lab.connect()
    assert isinstance(t.agent, DQNLearner)  # new brains follow the choice
    assert t.set_algorithm(AUTO) == "" and isinstance(t.agent, DQNLearner)
    t.train(1)
    dqn = t.agent
    assert isinstance(dqn, DQNLearner) and dqn.known > 0
    note = t.set_algorithm(Q_TABLE)
    table = t.agent
    assert isinstance(table, QLearningAgent) and "simple table" in note
    assert table.policy() == dqn.policy() and table.episodes == 1 and t.episodes == 1
    assert lab.changes[-1].reason == "algorithm change"
    assert t.store is not None and t.game_id is not None
    assert Brain.from_dict(t.store.load(t.game_id)).algorithm == Q_TABLE
    t.start()
    t.step()
    with pytest.raises(ConfigurationError, match="Stop training"):
        t.set_algorithm(DQN)
    t.stop()  # closes go 2
    note = t.set_algorithm(DQN)
    assert isinstance(t.agent, DQNLearner) and "neural network" in note
    assert t.agent.episodes == 2 and t.episodes == 2
    assert t.set_algorithm(DQN) == ""  # already there
    t.train(1)
    assert isinstance(t.agent, DQNLearner) and t.episodes == 3


# ------------------------------------------------------------ T-707: demonstrations


def test_manual_mode_keeps_the_childs_moves_as_demonstrations(lab: Lab) -> None:
    lab.connect()
    t = lab.training
    assert t.demonstrations.count == 0
    t.train(1)  # the AI's own play is not a demonstration
    assert t.demonstrations.count == 0
    t.start(Mode.MANUAL)
    for _ in range(5):
        lab.play(["up"], 4)
    assert t.demonstrations.count == 5 and t.demonstrations.shown == 5
    assert all(d.source == "child" for d in t.demonstrations)
    assert "shown the AI 5 moves" in t.summary()
    with pytest.raises(ConfigurationError, match="Stop training"):
        t.rehearse()
    t.stop()
    agent = t.agent
    assert agent is not None
    before = agent.updates
    assert t.rehearse() == 5 and agent.updates == before + 5
    assert t.rehearse(passes=2) == 10 and agent.updates == before + 15 and t.saves > 0
    with pytest.raises(ConfigurationError):
        t.rehearse(passes=-1)
    # They survive a reconnect through the saved brain, and a fresh brain has none.
    lab.connection.disconnect()
    lab.connect()
    assert t.demonstrations.count == 5
    t.forget()
    assert t.demonstrations.count == 0 and t.rehearse() == 0
    store = t.store
    assert store is not None and t.game_id is not None
    lab.connection.disconnect()
    data = store.load(t.game_id)
    del data["demonstrations"]  # a brain saved before T-707
    store.save(data)
    lab.connect()
    assert t.fresh_start == "" and t.demonstrations.count == 0


# ------------------------------------------------------------ T-708: skills per goal step


def test_the_goal_step_picks_the_skill(lab: Lab, tmp_path: Path) -> None:
    from retro_ai_lab.learning import SKILLS, SkillsLearner
    from retro_ai_lab.learning.skills import DEFAULT_SKILL

    t = TrainingService(
        lab.connection,
        lab.observation,
        lab.discovery,
        lab.bus,
        frames_per_step=4,
        max_steps=20,
        settings=Settings(epsilon=0.5),
        seed=1,
        store=BrainStore(tmp_path / "skills"),
        algorithm=SKILLS,
        goals=lab.goals,
        lookahead=1,  # updates are counted step by step below (T-963)
    )
    lab.connect()
    agent = t.agent
    assert isinstance(agent, SkillsLearner) and agent.skill == DEFAULT_SKILL
    goals = lab.goals
    goal = goals.goal
    assert goal is not None
    for o in list(goal):
        goals.remove_objective(o.id)
    first = goals.add_objective("Get going")
    second = goals.add_objective("Keep going", depends_on=[first.id])
    t.train(1)
    assert agent.skill == first.id and agent.skills == (DEFAULT_SKILL, first.id)
    assert agent.skill_learner(first.id).updates == 20
    goals.mark_complete(first.id)
    t.train(1)
    assert agent.skill == second.id and agent.skill_learner(second.id).updates == 20
    goals.mark_complete(second.id)
    t.start(Mode.MANUAL)
    lab.play(["up"], 4)
    # Both steps done: the AI works on whatever the goal suggests next (or the general skill).
    assert agent.skill == (t.active_skill() or DEFAULT_SKILL)
    assert agent.skill not in (first.id, second.id)
    assert agent.skill_learner(agent.skill).updates == 1
    t.stop()
    goals.mark_not_complete(second.id)
    t.start(Mode.MANUAL)
    lab.play(["up"], 4)
    assert agent.skill == second.id and agent.skill_learner(second.id).updates == 21
    t.stop()
    lab.connection.disconnect()
    lab.connect()
    again = t.agent
    assert isinstance(again, SkillsLearner) and again is not agent
    assert again.skills == agent.skills and again.updates == agent.updates
    # Changing to skills by hand wraps the brain that was there.
    plain = lab.training
    plain.train(1)
    assert plain.agent is not None
    note = plain.set_algorithm(SKILLS)
    assert isinstance(plain.agent, SkillsLearner) and "skills per goal step" in note
    assert plain.agent.episodes == 1
    plain.train(1)  # no goal service here: the general skill plays
    assert plain.agent.skill == DEFAULT_SKILL and plain.agent.episodes == 2
    assert plain.active_skill() is None


# ------------------------------------------------------------ T-844


def _drive_to_game_over(lab: Lab) -> None:
    game = lab.adapter().game
    while not game.state.game_over:
        game.step(["up"])
    lab.observation.observe()


def test_the_ai_works_out_which_button_starts_the_game(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-844: on a screen that ignores the pad the AI tries every button and keeps the one."""
    import retro_ai_lab.training.service as module

    # The go-start probe (T-848) would find it out first; here the goes have to show it.
    monkeypatch.setattr(
        module, "probe_inputs", lambda *args, **kwargs: InputProbe(True, frozenset())
    )
    k = lab.connect()
    assert k.wake_button is None
    _drive_to_game_over(lab)
    t = lab.training
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)  # the checkpoint is the game-over screen
    assert t.wake_button is None
    while t.episodes < 1:
        assert t.step()
    assert t.last is not None and t.last.result.ended_by == "stuck"
    saved: list[int] = []
    monkeypatch.setattr(lab.discovery, "save", lambda: saved.append(1))
    assert t.step()  # the next go starts by working out how to wake the game
    assert t.wake_button == "start" and k.wake_button == "start"
    assert saved == [1]  # remembered with the game's knowledge
    assert "trying each button to get it going" in caplog.text
    assert "Pressing Start gets the game going." in caplog.text
    assert "Start gets the game going." in lab.discovery.summary()
    episode = t.episode
    assert episode is not None and episode.unstick_buttons == ("start",)
    assert not lab.adapter().game.state.game_over  # go 2 is being played in the game
    while t.episodes < 2:
        assert t.step()
    assert t.last is not None and t.last.result.ended_by != "stuck"
    # Goes after that start from the new checkpoint, inside the game.
    assert t.step() and not lab.adapter().game.state.game_over
    t.stop()


def test_identical_goes_trigger_the_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T-844: two goes with the same states although the buttons differed = a demo."""
    import retro_ai_lab.training.service as module
    from retro_ai_lab.training.episode import EpisodeRunner

    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
    training = TrainingService(  # unstick_after=None: the screen never counts as stuck
        connection,
        observation,
        discovery,
        bus,
        frames_per_step=4,
        max_steps=12,
        seed=1,
        unstick_after=None,
    )
    lab = Lab(bus, connection, observation, discovery, None, None, training, [], [])  # type: ignore[arg-type]
    lab.connect()
    _drive_to_game_over(lab)
    monkeypatch.setattr(
        module, "probe_inputs", lambda *args, **kwargs: InputProbe(True, frozenset())
    )  # no T-848 probe
    monkeypatch.setattr(EpisodeRunner, "fingerprint", property(lambda self: ("same",) * 8))
    monkeypatch.setattr(EpisodeRunner, "actions_varied", property(lambda self: True))
    t = lab.training
    t.start(Mode.INDEPENDENT)
    while t.episodes < 2:
        assert t.step()
    assert t.wake_button is None and lab.adapter().game.state.game_over
    assert t.step()
    assert t.wake_button == "start" and not lab.adapter().game.state.game_over
    t.stop()


def test_goes_look_identical() -> None:
    from retro_ai_lab.training.service import IDENTICAL_GO_STEPS, goes_look_identical

    same = tuple(f"s{i}" for i in range(IDENTICAL_GO_STEPS))
    assert goes_look_identical((same, True), (same, True))
    assert not goes_look_identical((same, False), (same, True))  # the AI never varied
    assert not goes_look_identical((same, True), (same, False))
    assert not goes_look_identical((same[:-1], True), (same[:-1], True))  # too short
    assert not goes_look_identical((same, True), ((*same[:-1], "other"), True))


# ------------------------------------------------------------ T-845


def _end_go_by_reset(lab: Lab, ids: tuple[str, ...], steps: int = 1) -> None:
    t = lab.training
    before = t.episodes
    for _ in range(steps):
        assert t.step()  # a go is running after this
    reset = GameEvent(EventKind.RESET, 5, None, "went home", variables=ids)
    lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))
    t.step()
    assert t.episodes == before + 1


def test_three_short_resets_from_the_same_values_teach_the_ai_they_cycle(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    k = lab.connect()
    t = lab.training
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)
    from retro_ai_lab.training.service import RESET_SUSPECT_GOES, RESET_SUSPECT_STEPS

    assert RESET_SUSPECT_STEPS == 120 and RESET_SUSPECT_GOES == 3  # T-867: goes up to 120 steps
    ids = ("v3", "v4")
    _end_go_by_reset(lab, ids)
    _end_go_by_reset(lab, ids, steps=40)  # T-867: a go this long (the lab caps at 50) counts
    assert not any(e.kind == CYCLES for e in k.get("v3").evidence)
    _end_go_by_reset(lab, ids, steps=20)
    assert all(any(e.kind == CYCLES for e in k.get(v).evidence) for v in ids)
    assert "cycle back to their start; no longer reading that as a restart" in caplog.text
    # The next go is not cut short by those values: the detector ignores their return.
    assert t.step() and t.episode is not None and t.episode.steps == 1
    t.stop()


def test_short_resets_that_differ_are_left_alone(lab: Lab) -> None:
    k = lab.connect()
    t = lab.training
    t.start(Mode.INDEPENDENT)
    _end_go_by_reset(lab, ("v3", "v4"))
    _end_go_by_reset(lab, ("v3", "v4"))
    _end_go_by_reset(lab, ("v3",))  # a different group: the count starts again
    _end_go_by_reset(lab, ("v3",))
    assert not any(e.kind == CYCLES for v in ("v3", "v4") for e in k.get(v).evidence)
    # A long go in between also starts the count again.
    for _ in range(2):
        _end_go_by_reset(lab, ("v3",))
    assert any(e.kind == CYCLES for e in k.get("v3").evidence)
    t.stop()


def test_the_first_go_checks_the_game_listens(lab: Lab, caplog: pytest.LogCaptureFixture) -> None:
    """T-848: no waiting for goes to prove it; the first go asks the game directly."""
    k = lab.connect()
    _drive_to_game_over(lab)
    t = lab.training
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)  # the checkpoint is the game-over screen
    assert t.step()  # the very first step: the search runs, then go 1 begins in the game
    assert t.wake_button == "start" and k.wake_button == "start"
    assert t.episodes == 0 and not lab.adapter().game.state.game_over
    assert "trying each button to get it going" in caplog.text
    t.stop()


def test_training_stops_instead_of_playing_a_game_that_ignores_the_ai(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-852: no points from a demo; after three failed searches training stops and says so."""
    import retro_ai_lab.training.service as module

    searches: list[int] = []
    monkeypatch.setattr(module, "find_wake_button", lambda *a, **k: searches.append(1))
    lab.connect()
    _drive_to_game_over(lab)
    t = lab.training
    caplog.set_level(logging.INFO)
    frame = lab.adapter().game.state.frame
    t.start(Mode.INDEPENDENT)
    assert not t.step()  # no go begins
    assert t.status is Status.IDLE and t.episodes == 0 and t.episode is None
    assert len(searches) == module.DEAF_TRIES
    assert lab.changes[-1].reason == "not listening" and lab.changes[-1].status is Status.IDLE
    assert module.NOT_LISTENING in caplog.text
    # The game ran on between the searches (an intro may have been playing).
    assert lab.adapter().game.state.frame >= frame + (module.DEAF_TRIES - 1) * module.DEAF_WAIT
    assert "Resting" not in t.summary() and t.summary() == "Ready to train."
    t.start(Mode.INDEPENDENT)  # Start works again afterwards
    assert t.status is Status.RUNNING
    t.stop()


def test_a_second_search_after_a_wait_can_still_find_the_button(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-852: a failed search is followed by a wait and another try, not a demo go."""
    import retro_ai_lab.training.service as module

    real = module.find_wake_button
    searches: list[int] = []

    def flaky(*args: object, **kwargs: object) -> str | None:
        searches.append(1)
        if len(searches) == 1:
            return None
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "find_wake_button", flaky)
    k = lab.connect()
    _drive_to_game_over(lab)
    t = lab.training
    t.start(Mode.INDEPENDENT)
    assert t.step()  # the second search found Start; go 1 runs in the game
    assert t.status is Status.RUNNING and len(searches) == 2
    assert t.wake_button == "start" and k.wake_button == "start"
    assert not lab.adapter().game.state.game_over
    t.stop()


def test_two_starts_drop_a_guess_that_moves_by_itself(lab: Lab) -> None:
    """T-855: the probe at Start doubts a junk score guess, and the next Start drops it."""
    k = lab.connect()
    junk = k.propose(RAM, LAYOUT.timer, kind=VariableKind.SCORE, name="Timer as score")
    real = k.get(next(v.id for v in k if v.name == "Score"))
    t = lab.training
    for starts in (1, 2):
        t.start(Mode.INDEPENDENT)
        assert t.step()
        t.stop()
        assert k.get(junk.id).rejected == (starts == 2)
    assert not k.get(real.id).rejected
    # No save states: no probe, nothing doubted, no error.
    lab.connection.disconnect()
    k2 = lab.connect(capabilities=Capabilities(memory_read=True, frame_advance=True, input=True))
    junk2 = k2.propose(RAM, LAYOUT.timer, kind=VariableKind.SCORE, name="Timer again")
    t2 = lab.training
    t2.start(Mode.INDEPENDENT)
    assert t2.step() and not k2.get(junk2.id).evidence
    t2.stop()


def test_a_deaf_game_doubts_nothing(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    """T-868: while a demo plays, everything moves by itself; that says nothing about play."""
    import retro_ai_lab.training.service as module
    from retro_ai_lab.services.discovery import MOVES_ALONE

    k = lab.connect()
    junk = k.propose(RAM, LAYOUT.frame, kind=VariableKind.PROGRESS, name="Frame as progress")
    moving = frozenset({(RAM, LAYOUT.frame)})
    monkeypatch.setattr(module, "probe_inputs", lambda *a, **kw: InputProbe(False, moving))
    t = lab.training
    t.start(Mode.INDEPENDENT)
    t.step()  # the probe says the game is deaf: the search runs, nothing is doubted
    t.stop()
    assert not any(e.kind == MOVES_ALONE for e in k.get(junk.id).evidence)
    monkeypatch.setattr(module, "probe_inputs", lambda *a, **kw: InputProbe(True, moving))
    t.start(Mode.INDEPENDENT)
    assert t.step()
    t.stop()
    assert any(e.kind == MOVES_ALONE for e in k.get(junk.id).evidence)


def test_a_go_credits_the_values_that_moved(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    """T-870: every go the game listened to begins a credit window and settles it at its end."""
    lab.connect()
    calls: list[str] = []
    real_begin, real_credit = lab.discovery.begin_go, lab.discovery.credit_played

    def begin() -> None:
        calls.append("begin")
        real_begin()

    def credit(*, frame: int | None = None) -> tuple[Variable, ...]:
        calls.append(f"credit@{frame}")
        return real_credit(frame=frame)

    monkeypatch.setattr(lab.discovery, "begin_go", begin)
    monkeypatch.setattr(lab.discovery, "credit_played", credit)
    t = lab.training
    t.train(2)
    assert [c.split("@")[0] for c in calls] == ["begin", "credit", "begin", "credit"]
    assert all(int(c.split("@")[1]) > 0 for c in calls if c.startswith("credit"))


def test_go_settings_are_a_persons_choice_per_game(lab: Lab) -> None:
    """T-873: how long a go lasts and when it gives up can be set, idle only, and are kept."""
    k = lab.connect()
    t = lab.training
    assert DEFAULT_MAX_STEPS == 600 and t.max_steps == 50  # the lab's own default
    assert t.unstick_after == DEFAULT_UNSTICK_AFTER
    for bad in (0, -1, 2.5):
        with pytest.raises(ConfigurationError):
            t.set_go_length(bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        t.set_give_up_after(0)
    t.set_go_length(20)
    t.set_give_up_after(None)
    assert t.max_steps == 20 and t.unstick_after is None
    assert lab.changes[-1].reason == "go settings"
    brain = t.brain()
    assert brain is not None and brain.go_length == 20 and brain.give_up_after is None
    assert t.saves >= 1  # written to disk at once
    t.start(Mode.INDEPENDENT)
    with pytest.raises(ConfigurationError):
        t.set_go_length(30)  # not while training
    t.step()
    t.stop()
    t.train(1)
    assert t.last is not None and t.last.result.steps == 20
    t.set_give_up_after(7)
    assert t.unstick_after == 7
    # Reconnecting brings the same game's choices back; another game gets the defaults.
    lab.connection.disconnect()
    assert t.max_steps == 20  # untouched until a game is connected
    lab.connect()
    assert t.max_steps == 20 and t.unstick_after == 7 and k.game_id == t.game_id
    lab.connection.disconnect()
    lab.connection.connect("fake", seed=9)
    assert t.max_steps == 50 and t.unstick_after == DEFAULT_UNSTICK_AFTER


def test_every_step_announces_the_buttons_held(lab: Lab) -> None:
    """T-874: the runner says what each step held, before the step's reward is collected."""
    from retro_ai_lab.rewards import AGAIN_ID, Category
    from retro_ai_lab.services import ButtonsHeld

    held: list[ButtonsHeld] = []
    lab.bus.subscribe(ButtonsHeld, held.append)
    lab.connect()
    t = lab.training
    t.train(1)
    assert t.last is not None and len(held) == t.last.result.steps
    assert all(h.source == "ai" for h in held)
    # The points for a repeated button land in the go that pressed it.
    profile = lab.rewards.profile
    assert profile is not None
    lab.rewards.set_value(AGAIN_ID, 1.0)
    before = lab.rewards.tally.by_category(profile)[Category.CONTROLS]
    held.clear()
    t.train(1)
    repeats = sum(len(a.buttons & b.buttons) for a, b in pairwise(held))
    assert lab.rewards.tally.by_category(profile)[Category.CONTROLS] - before == float(repeats)
    assert t.last is not None and t.last.result.reward >= float(repeats) - 1e-9


def test_the_search_runs_flat_out_and_puts_the_speed_back(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-853: the probe and the search run at FAST_SPEED, then the speed goes back."""
    import retro_ai_lab.training.service as module

    speeds: list[float] = []
    real_probe, real_search = module.probe_inputs, module.find_wake_button

    def probe(*args: object, **kwargs: object) -> InputProbe:
        speeds.append(lab.adapter().speed)
        return real_probe(*args, **kwargs)  # type: ignore[arg-type]

    def search(*args: object, **kwargs: object) -> str | None:
        speeds.append(lab.adapter().speed)
        return real_search(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "probe_inputs", probe)
    monkeypatch.setattr(module, "find_wake_button", search)
    lab.connect()
    _drive_to_game_over(lab)
    t = lab.training
    t.start(Mode.INDEPENDENT)
    assert t.step() and t.wake_button == "start"
    assert speeds and set(speeds) == {module.FAST_SPEED}  # the probe, the search
    assert lab.adapter().speed == 1.0  # LIVE again for the go
    t.stop()
    # At "Practise fast" the emulator is already fast and stays so.
    t.set_speed(Speed.FAST)
    speeds.clear()
    _drive_to_game_over(lab)
    t.start(Mode.INDEPENDENT)
    assert t.step() and set(speeds) == {module.FAST_SPEED}
    assert lab.adapter().speed == module.FAST_SPEED
    t.stop()
    assert lab.adapter().speed == 1.0
    # No speed control: left alone, no error.
    lab.connection.disconnect()
    lab.connect(
        capabilities=Capabilities(
            memory_read=True, frame_advance=True, input=True, save_states=True
        )
    )
    _drive_to_game_over(lab)
    t2 = lab.training
    speeds.clear()
    t2.start(Mode.INDEPENDENT)
    assert t2.step() and speeds and set(speeds) == {1.0}
    t2.stop()


def test_a_listening_game_is_probed_only_after_a_go_that_went_wrong(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-848: one probe at Start, none after goes that ran their course, one after a reset."""
    import retro_ai_lab.training.service as module

    probes: list[int] = []
    real = module.probe_inputs

    def counting(*args: object, **kwargs: object) -> InputProbe:
        probes.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "probe_inputs", counting)
    lab.connect()
    t = lab.training
    t.start(Mode.INDEPENDENT)  # in the game: it listens
    assert t.step() and probes == [1] and t.wake_button is None
    while t.episodes < 1:
        assert t.step()
    assert t.last is not None and t.last.result.ended_by == "max_steps"
    assert t.step() and probes == [1]  # go 2 starts without asking again
    _end_go_by_reset(lab, ("v1",))
    assert t.step() and probes == [1, 1]  # a reset: ask again
    t.stop()
    # No save states: no probe, no error.
    t2 = lab.training
    lab.connection.disconnect()
    lab.connect(capabilities=Capabilities(memory_read=True, frame_advance=True, input=True))
    probes.clear()
    t2.start(Mode.INDEPENDENT)
    assert t2.step() and probes == []
    t2.stop()


def test_behaviour_changes_do_not_ask_the_brain_again(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-910: the panel asks at every go's end; a fresh policy pass cost the window a second."""
    from retro_ai_lab.training import service as service_module

    calls = 0
    original = service_module.policy_of

    def counted(learner: object) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return original(learner)  # type: ignore[arg-type]

    monkeypatch.setattr(service_module, "policy_of", counted)
    lab.connect()
    t = lab.training
    t.train(2)
    assert calls == 2  # one snapshot per go, taken as the go ends
    _, goes_ago, known = t.behaviour_changes(window=1)
    assert goes_ago == 1 and known > 0 and calls == 2  # read from the snapshot, not the brain
    for _ in range(5):
        t.behaviour_changes()
    assert calls == 2


def test_each_go_logs_the_brain_and_what_it_sees(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    """T-913: the line after a go's says which brain, how much it guesses and what it sees."""
    lab.connect()
    t = lab.training
    caplog.set_level(logging.INFO, logger="retro_ai_lab.training.service")
    t.start(Mode.INDEPENDENT)
    while t.episodes < 1:
        assert t.step()
    messages = [r.getMessage() for r in caplog.records]
    go = messages.index(next(m for m in messages if m.startswith("Go 1 ended by")))
    brain = messages[go + 1]
    assert brain.startswith("The brain after go 1: Simple table. Knows ")
    assert "tries something new" in brain and brain.endswith(f"It sees {t.encoder.describe()}.")
    assert "position" in brain or "value" in brain
    t.stop()


def test_training_starts_in_hold_mode(lab: Lab, caplog: pytest.LogCaptureFixture) -> None:
    """T-942: a link the play clock let run free is held again before the AI's first step."""
    from retro_ai_lab.emulator.bizhawk.simulator import BizHawkSimulator

    with BizHawkSimulator() as sim:
        lab.connection.connect("bizhawk", port=sim.port, call_timeout_s=2.0, timeout_s=2.0)
        lab.connection.call(lambda a: a.set_free_run(True))
        assert sim.free_run and not sim.paused
        with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.service"):
            lab.training.start(Mode.INDEPENDENT)
        assert not sim.free_run and sim.paused
        assert not lab.connection.call(lambda a: a.free_run)
        assert "holds it between steps" in caplog.text
        lab.training.stop()
        lab.connection.disconnect()


def test_practising_fast_names_fast_training_in_an_ai_mode(lab: Lab) -> None:
    """T-944: the window rests on this, as the emulator's speed-up already did."""
    lab.connect()
    t = lab.training
    assert not t.practising_fast
    t.set_speed(Speed.FAST)
    assert not t.practising_fast  # not started
    t.start(Mode.INDEPENDENT)
    assert t.practising_fast
    t.pause()
    assert not t.practising_fast
    t.resume()
    assert t.practising_fast
    t.set_speed(Speed.LIVE)
    assert not t.practising_fast
    t.set_speed(Speed.FAST)
    assert t.practising_fast
    t.stop()
    assert not t.practising_fast


def _goes_lose_their_first_life_at(
    monkeypatch: pytest.MonkeyPatch, steps: list[int | None]
) -> None:
    """The next goes report these first-loss steps, whatever happened in them (T-961)."""
    from retro_ai_lab.training.episode import EpisodeResult, EpisodeRunner

    real = EpisodeRunner.finish
    queue = list(steps)

    def finish(self: EpisodeRunner, *, ended_by: str | None = None) -> EpisodeResult:
        result = real(self, ended_by=ended_by)
        if not queue:
            return result
        step = queue.pop(0)
        return replace(result, first_setback_step=step, setbacks=0 if step is None else 1)

    monkeypatch.setattr(EpisodeRunner, "finish", finish)


def _run_goes(lab: Lab, count: int) -> None:
    t = lab.training
    while t.episodes < count:
        assert t.step()


def _the_game_takes_a_life_every(lab: Lab, steps: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """From the frame the game is at, a life goes every ``steps`` steps whatever is held: a
    dying moment, or a timed hazard (T-965). Lives are topped up so the game never ends, and
    the life count is confirmed so the losses do not make the AI doubt it."""
    game = lab.adapter().game
    ram = game._ram
    ram[LAYOUT.lives] = 200
    k = lab.discovery.knowledge
    assert k is not None
    lives = k.at(RAM, LAYOUT.lives)
    assert lives is not None
    k.confirm(lives.id)
    every = steps * lab.training.frames_per_step
    first = game._u16(LAYOUT.frame)
    real = game._simulate

    def simulate(held: frozenset[str]) -> None:
        real(held)
        if (game._u16(LAYOUT.frame) + 1 - first) % every == 0 and ram[LAYOUT.lives] > 0:
            ram[LAYOUT.lives] -= 1

    monkeypatch.setattr(game, "_simulate", simulate)


def test_a_start_that_loses_a_life_by_itself_moves_the_goes_past_the_loss(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-961: the user's checkpoint was a dying moment; every go opened with a -10."""
    from retro_ai_lab.training.service import DOOMED_GOES, DOOMED_START_STEPS, DOOMED_STEP_SLACK

    assert (DOOMED_GOES, DOOMED_START_STEPS, DOOMED_STEP_SLACK) == (3, 75, 2)
    lab.connect()
    _the_game_takes_a_life_every(lab, 6, monkeypatch)  # the car stands still; it dies anyway
    game = lab.adapter().game
    lives, position = game.state.lives, game.state.position
    t = lab.training
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)  # the checkpoint is that doomed moment
    _goes_lose_their_first_life_at(monkeypatch, [6, 7, 6])  # within the slack of one another
    _run_goes(lab, 2)
    assert t.step() and game.state.lives == lives  # two goes: no suspicion yet
    _run_goes(lab, 3)
    assert "loses a life by itself" not in caplog.text  # checked when the next go starts
    assert t.step() and t.episode is not None and t.episode.steps == 1
    assert (
        "The spot the goes start from loses a life by itself, 6 steps in, whatever is held "
        "(the last 3 goes all lost one there): the goes now start 6 steps later, where the "
        "game listens to the AI again." in caplog.text
    )
    # Go 4 started after the loss: a life down, the car where it stood, stopped.
    assert game.state.lives == lives - 1 and game.state.position == position
    assert t.last is not None and t.last.result.first_setback_step == 6
    # The goes after it start there too.
    _run_goes(lab, 4)
    assert t.step() and game.state.lives == lives - 1
    t.stop()


def test_a_loss_a_held_button_avoids_leaves_the_start_where_it_is(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-965: the user's start lost a life every 60 steps to a player who held nothing, and
    the check moved the goes past it 14 times; a loss a button avoids is the AI's."""
    lab.connect()
    lab.play(["up"], 3)  # speed 3 with an obstacle ahead: holding nothing crashes at step 6
    game = lab.adapter().game
    lives, position = game.state.lives, game.state.position
    t = lab.training
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)
    _goes_lose_their_first_life_at(monkeypatch, [6, 6, 6, 6, 6, 6])
    _run_goes(lab, 3)
    assert t.step() and t.episode is not None and t.episode.steps == 1
    assert (
        "The spot the goes start from loses a life 6 steps in holding nothing, but keeps it "
        "holding Down: the life the last 3 goes lost at step 6 is the AI's to avoid." in caplog.text
    )
    assert "loses a life by itself" not in caplog.text
    # Go 4 started where the others did: no life down, one step of speed 3 driven.
    assert game.state.lives == lives
    assert game.state.position == position + 3 * t.frames_per_step
    _run_goes(lab, 6)  # three more goes losing it at step 6: cleared, not checked again
    assert t.step() and game.state.lives == lives
    assert caplog.text.count("the AI's to avoid") == 1
    t.stop()


def test_the_start_is_moved_three_times_at_most(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-965: a game that takes a life every few steps whatever is held is not escaped by
    moving the start on and on; after three moves the start stays."""
    from retro_ai_lab.training.service import DOOMED_MOVES_MOST

    assert DOOMED_MOVES_MOST == 3
    lab.connect()
    _the_game_takes_a_life_every(lab, 6, monkeypatch)
    game = lab.adapter().game
    lives = game.state.lives
    t = lab.training
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)
    _goes_lose_their_first_life_at(monkeypatch, [6] * 40)
    for moves in range(1, 4):  # goes 4, 7 and 10 start one life further on
        _run_goes(lab, 3 * moves)
        assert t.step() and game.state.lives == lives - moves
    _run_goes(lab, 12)
    assert t.step() and game.state.lives == lives - 3  # the fourth loss moves nothing
    assert (
        "The goes' start has been moved 3 times since this checkpoint was taken and the "
        "last 3 goes still lost a life 6 steps in: the game takes lives here whatever is "
        "held, so the start stays and the loss is the AI's to learn around." in caplog.text
    )
    _run_goes(lab, 15)
    assert t.step() and game.state.lives == lives - 3
    assert caplog.text.count("whatever is held (the last") == 3
    assert caplog.text.count("the start stays") == 1
    t.stop()
    t.start(Mode.INDEPENDENT)  # a new checkpoint: its start may be moved again
    _run_goes(lab, 19)  # the stop ended go 16; goes 17 to 19 start from the new checkpoint
    assert t.step()
    assert caplog.text.count("whatever is held (the last") == 4
    t.stop()


def test_a_loss_the_ai_causes_leaves_the_start_where_it_is(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-961: a car standing still loses nothing holding nothing; the losses were the AI's."""
    lab.connect()
    game = lab.adapter().game
    lives, position = game.state.lives, game.state.position
    t = lab.training
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)
    _goes_lose_their_first_life_at(monkeypatch, [6, 6, 6])
    _run_goes(lab, 3)
    assert t.step() and t.episode is not None and t.episode.steps == 1
    assert (
        "The spot the goes start from keeps its lives for 8 steps holding nothing: the "
        "life the last 3 goes lost at step 6 is the AI's doing." in caplog.text
    )
    assert "loses a life by itself" not in caplog.text
    assert game.state.lives == lives and game.state.position == position
    t.stop()


def test_losses_at_different_steps_or_late_raise_no_suspicion(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-961: only the same early step three goes running says anything about the start."""
    lab.connect()
    lab.play(["up"], 3)  # the doomed start of the first test
    game = lab.adapter().game
    lives = game.state.lives
    t = lab.training
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)
    # 6 and 9 are too far apart; a go with no loss starts the count again; 80 is too late.
    _goes_lose_their_first_life_at(monkeypatch, [6, 9, 6, None, 6, 6, 80, 80, 80])
    for goes in range(1, 10):
        _run_goes(lab, goes)
        assert t.step() and game.state.lives == lives
    assert "The spot the goes start from" not in caplog.text
    t.stop()


def test_goes_without_a_checkpoint_do_not_count_as_a_doomed_start(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-961: without save states the goes carry on from wherever the game is."""
    lab.connect()
    lab.play(["up"], 3)
    t = lab.training
    t._harness = None  # an emulator without save states
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)
    _goes_lose_their_first_life_at(monkeypatch, [6, 6, 6])
    _run_goes(lab, 3)
    assert t.step()
    assert "The spot the goes start from" not in caplog.text
    t.stop()


def test_the_service_hands_the_runner_its_lookahead(lab: Lab, tmp_path: Path) -> None:
    """T-963: the app's goes learn each step from the return over the next eight."""
    from retro_ai_lab.training import LOOKAHEAD_STEPS

    assert LOOKAHEAD_STEPS == 8
    with pytest.raises(ConfigurationError):
        TrainingService(lab.connection, lab.observation, lab.discovery, lab.bus, lookahead=0)
    t = TrainingService(
        lab.connection,
        lab.observation,
        lab.discovery,
        lab.bus,
        frames_per_step=4,
        max_steps=50,
        settings=Settings(epsilon=0.5),
        seed=1,
        store=BrainStore(tmp_path / "b2"),
        lookahead=3,
    )
    lab.connect()
    t.start(Mode.INDEPENDENT)
    agent = t.agent
    assert agent is not None
    assert t.step() and t.step() and agent.updates == 0  # two steps pending
    assert t.step() and agent.updates == 1  # the first step learned from the three
    t.stop()  # the go ends: the pending steps are learned too
    assert agent.updates == 3 and t.total_steps == 3


def test_deaths_are_counted_on_the_value_your_reward_penalises(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    """T-964: the user's "Lifes goes down" fired twice a go while the count said five."""
    k = lab.connect()
    lives = k.at(RAM, LAYOUT.lives)
    assert lives is not None
    stray = k.propose(RAM, LAYOUT.timer, kind=VariableKind.LIVES, evidence=SURE)  # the AI's guess
    own = lab.rewards.add_own(stray.id, goes_up=False, value=-5.0, name="Fuel")
    speed = k.at(RAM, LAYOUT.speed)
    assert speed is not None
    back = lab.rewards.add_own(speed.id, goes_up=False, value=-2.0)  # slowing down: no death
    t = lab.training
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)
    assert t.step() and t.episode is not None
    assert t.episode.signals.watched == frozenset({stray.id})
    assert (  # the guess is unnamed, so its kind and place name it
        f"Lives lost are counted on Lives at {RAM}[{LAYOUT.timer}], which your reward penalises."
        in caplog.text
    )
    # Only that value's drops count as deaths this go.
    lab.bus.publish(
        DiscoveryUpdated((), (), (GameEvent(EventKind.SETBACK, 20, lives.id, "", 3, 2),), ())
    )
    assert t.episode.signals.take_setbacks() == 0
    lab.bus.publish(
        DiscoveryUpdated((), (), (GameEvent(EventKind.SETBACK, 20, stray.id, "", 9, 8),), ())
    )
    assert t.episode.signals.take_setbacks() == 1
    t.stop()
    # The reward gone, the next go counts the AI's best guesses again, and says so once.
    lab.rewards.remove_own(own.id)
    lab.rewards.remove_own(back.id)
    caplog.clear()
    t.start(Mode.INDEPENDENT)
    assert t.step() and t.step() and t.episode is not None
    best = k.best(VariableKind.LIVES)
    assert best is not None and t.episode.signals.watched == frozenset({best.id})
    assert caplog.text.count("Lives lost are counted on") == 1
    assert "the AI's best life count and health guesses" in caplog.text
    t.stop()


# ------------------------------------------------------------ T-974: it finds itself on the screen


def test_the_first_go_finds_the_ai_on_the_screen(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    k = lab.connect()
    for _ in range(8):  # off the start line, which is drawn through the car
        lab.adapter().game.step(["up"])
    lab.observation.observe()
    t = lab.training
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)
    assert t.step()
    assert k.player is not None
    assert {"left", "right"} <= k.player.buttons
    assert "Found itself on the screen" in caplog.text
    assert "It found itself on the screen" in lab.discovery.summary()
    t.stop()


def test_a_sure_answer_is_not_looked_for_again(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    import retro_ai_lab.training.service as module
    from retro_ai_lab.understanding.knowledge import Confidence
    from retro_ai_lab.understanding.player import PlayerFinding

    k = lab.connect()
    k.player = PlayerFinding(None, (30, 45), frozenset({"left", "right"}), 2, Confidence.MEDIUM)
    calls: list[int] = []
    monkeypatch.setattr(module, "find_player_in_game", lambda *a, **kw: calls.append(1))
    t = lab.training
    t.start(Mode.INDEPENDENT)
    assert t.step()
    assert calls == []
    t.stop()


def test_a_weak_answer_is_looked_for_once_per_start(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    import retro_ai_lab.training.service as module

    lab.connect()
    calls: list[int] = []
    monkeypatch.setattr(module, "find_player_in_game", lambda *a, **kw: calls.append(1))
    t = lab.training
    t.start(Mode.INDEPENDENT)
    for _ in range(30):
        t.step()
    t.stop()
    assert calls == [1]  # nothing found: not asked again until the next Start
    t.start(Mode.INDEPENDENT)
    t.step()
    t.stop()
    assert calls == [1, 1]
