"""T-505: the training service (modes, start / pause / stop, history)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import Settings
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    GoalService,
    ObservationService,
    RewardService,
)
from retro_ai_lab.storage import GoalStore, RewardStore
from retro_ai_lab.training import (
    EpisodeFinished,
    Mode,
    Speed,
    Status,
    TrainingService,
    TrainingStateChanged,
)
from retro_ai_lab.understanding import Encoding, Evidence, Knowledge, VariableKind

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
    from retro_ai_lab.core import Capabilities

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
    t.start(Mode.INDEPENDENT)
    assert t.status is Status.RUNNING and t.mode is Mode.INDEPENDENT and t.ai_in_control
    assert not t.mode.allows_guidance and Mode.GUIDED.allows_guidance
    assert t.step() and t.episode is not None and t.episode.steps == 1
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
    lab.play(["up"], 1)
    assert agent.updates == 1 and t.total_steps == 1 and t.episode.steps == 1
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
    lab.connect(drop_after_ops=60)
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
    assert t.agent is agent  # kept until the next game, for saving (T-506)
    lab.connect()
    assert t.enabled and t.agent is not agent and t.history == () and t.episodes == 0
    assert t.summary() == "Ready to train."
    # A disconnect while idle is quiet.
    lab.connection.disconnect()
    assert [c.reason for c in lab.changes] == ["start", "disconnected"]


def test_app_context_wires_the_training_service(tmp_path: Path) -> None:
    from retro_ai_lab.ui.context import AppContext

    context = AppContext(data_dir=tmp_path)
    assert isinstance(context.training, TrainingService)
    context.connection.connect("fake")
    assert context.training.enabled
    context.connection.disconnect()
