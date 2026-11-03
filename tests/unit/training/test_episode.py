"""T-504: one episode of play-and-learn on the toy racer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings, StateEncoder
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    ExperimentHarness,
    GoalService,
    ObservationService,
    RewardService,
)
from retro_ai_lab.storage import GoalStore, RewardStore
from retro_ai_lab.training import EpisodeRunner, EpisodeSignals
from retro_ai_lab.understanding import (
    Encoding,
    EventKind,
    Evidence,
    GameEvent,
    Knowledge,
    VariableKind,
)

RAM = "ram"
WENT_HOME = "Lives, Lap all went back to their starting values."
SURE = [Evidence("test", "", 6.0)]


@dataclass
class Lab:
    bus: EventBus
    connection: ConnectionService
    observation: ObservationService
    discovery: DiscoveryService
    rewards: RewardService
    goals: GoalService

    def adapter(self) -> FakeEmulatorAdapter:
        a = self.connection.adapter
        assert isinstance(a, FakeEmulatorAdapter)
        return a


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
    rewards = RewardService(connection, discovery, bus, store=RewardStore(tmp_path / "r"))
    goals = GoalService(connection, discovery, bus, store=GoalStore(tmp_path / "g"))
    return Lab(bus, connection, observation, discovery, rewards, goals)


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


def runner(lab: Lab, **kw: object) -> EpisodeRunner:
    k = lab.discovery.knowledge
    assert k is not None
    space = ActionSpace.from_buttons(BUTTONS)
    agent = QLearningAgent(space, Settings(epsilon=0.5), seed=1)
    return EpisodeRunner(
        lab.connection,
        lab.observation,
        lab.bus,
        agent,
        StateEncoder(k),
        **kw,  # type: ignore[arg-type]
    )


def test_signals_collect_only_while_listening(lab: Lab) -> None:
    signals = EpisodeSignals(lab.bus)
    lab.connection.connect("fake", seed=3)
    k = teach(lab)
    score = k.at(RAM, LAYOUT.score)
    assert score
    gain = GameEvent(EventKind.SCORE_GAIN, 5, score.id, "")
    lab.bus.publish(DiscoveryUpdated((), (), (gain,), ()))
    assert signals.take() == (0.0, 0)  # not listening yet
    signals.start()
    lab.bus.publish(DiscoveryUpdated((), (), (gain, GameEvent(EventKind.RESET, 6, None, "")), ()))
    reward, resets = signals.take()
    assert reward == 5.0 + lab.rewards.profile.get("reset:*").value and resets == 1  # type: ignore[union-attr]
    assert signals.take() == (0.0, 0)
    signals.stop()
    lab.bus.publish(DiscoveryUpdated((), (), (gain,), ()))
    assert signals.take() == (0.0, 0)


def test_signals_remember_the_reset_that_ended_the_go(lab: Lab) -> None:
    """T-840: the detail of the reset goes to the log."""
    lab.connection.connect("fake", seed=3)
    teach(lab)
    r = runner(lab, frames_per_step=4, max_steps=30)
    r.begin()
    assert r.last_reset == ""
    reset = GameEvent(EventKind.RESET, 42, None, WENT_HOME)
    lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))
    assert r.last_reset == f"{WENT_HOME} (frame 42)"
    assert not r.step() and r.finish().ended_by == "reset"
    r.begin()
    assert r.last_reset == ""  # a new go starts clean
    r.finish()


def test_episode_runs_the_requested_steps(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3)
    teach(lab)
    r = runner(lab, frames_per_step=3, max_steps=40)
    frames_before = lab.observation.frame
    result = r.run_episode()
    assert result.steps == 40 and result.frames == 120 and result.ended_by == "max_steps"
    assert lab.observation.frame == frames_before + 120
    assert r._agent.updates == 40 and r._agent.episodes == 1
    assert 1 <= result.states_seen <= 41
    ai_inputs = lab.observation.inputs.by_source("ai")
    assert ai_inputs and all(e.source == "ai" for e in ai_inputs)
    space = r._agent.space
    assert all(space.matches(e.buttons).buttons == e.buttons for e in ai_inputs)
    assert lab.observation.inputs.held_at(lab.observation.frame) == frozenset()  # released
    assert not r.running


def test_game_over_is_unstuck_with_start_and_the_restart_ends_the_episode(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3)
    teach(lab)
    lab.observation.observe()  # the engine sees the fresh game first
    game = lab.adapter().game
    while not game.state.game_over:  # drive into game over by hand
        game.step(["up"])
    r = runner(lab, frames_per_step=4, max_steps=500, unstick_after=10)
    result = r.run_episode()
    assert r.unsticks == 1 and result.ended_by == "reset"
    assert 10 < result.steps < 30
    assert any(e.buttons == {"start"} for e in lab.observation.inputs.by_source("ai"))
    # The learner never learns from the Start press itself.
    assert r._agent.updates == result.steps - 1
    # With unsticking off, the AI stays stuck on the game-over screen until max_steps.
    while not game.state.game_over:
        game.step(["up"])
    stuck = runner(lab, frames_per_step=4, max_steps=40, unstick_after=None)
    assert stuck.run_episode().ended_by == "max_steps" and stuck.unsticks == 0
    with pytest.raises(ConfigurationError):
        runner(lab, unstick_after=0)


def test_dropped_link_ends_the_episode_quietly(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3, drop_after_ops=40)
    teach(lab)
    r = runner(lab, max_steps=100)
    result = r.run_episode()
    assert result.ended_by == "disconnected" and not lab.connection.is_connected
    assert result.steps < 100


def test_checkpoint_restarts_from_the_same_place(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3)
    teach(lab)
    harness = ExperimentHarness(lab.connection)
    harness.checkpoint()
    start = lab.adapter().game.ram
    r = runner(lab, max_steps=30, harness=harness)
    r.run_episode()
    assert lab.adapter().game.ram != start
    r.begin()
    assert lab.adapter().game.ram == start
    r.finish()
    # Without a checkpoint the game carries on from where it is.
    bare = runner(lab, max_steps=5)
    bare.run_episode()
    here = lab.adapter().game.ram
    bare.begin()
    assert lab.adapter().game.ram == here
    assert bare.finish().ended_by == "stopped"


def test_a_reset_seen_just_after_a_checkpoint_restore_is_the_restore(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3)
    teach(lab)
    harness = ExperimentHarness(lab.connection)
    harness.checkpoint()
    r = runner(lab, frames_per_step=4, max_steps=30, harness=harness, settle_frames=5)
    r.run_episode()
    r.begin()  # restores the checkpoint
    reset = GameEvent(EventKind.RESET, lab.observation.frame + 1, None, "")
    lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))
    assert r.step()  # 4 frames in: the engine is still reacting to the restore
    assert r.running
    lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))
    assert not r.step()  # 8 frames in: a real restart
    assert r.finish().ended_by == "reset"
    # Without a restore nothing is ignored.
    bare = runner(lab, frames_per_step=4, max_steps=30, settle_frames=5)
    bare.begin()
    lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))
    assert not bare.step() and bare.finish().ended_by == "reset"
    with pytest.raises(ConfigurationError):
        runner(lab, settle_frames=-1)


def test_settings_and_misuse(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3)
    teach(lab)
    with pytest.raises(ConfigurationError):
        runner(lab, frames_per_step=0)
    with pytest.raises(ConfigurationError):
        runner(lab, max_steps=0)
    r = runner(lab, max_steps=3)
    with pytest.raises(ConfigurationError):
        r.step()
    with pytest.raises(ConfigurationError):
        r.finish()
    r.begin()
    with pytest.raises(ConfigurationError):
        r.begin()
    assert r.running and r.steps == 0 and r.state is not None
    assert r.step() and r.step() and not r.step()
    assert r.finish().ended_by == "max_steps"


def test_short_training_run_learns_some_situations(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3)
    teach(lab)
    harness = ExperimentHarness(lab.connection)
    harness.checkpoint()
    r = runner(lab, frames_per_step=4, max_steps=60, harness=harness)
    results = [r.run_episode() for _ in range(8)]
    agent = r._agent
    assert agent.episodes == 8 and agent.updates == sum(x.steps for x in results) - r.unsticks
    assert len(agent.table) >= 3
    assert all(x.ended_by in ("max_steps", "reset") for x in results)
    assert agent.epsilon < 0.5
    # Reaching a checkpoint completes the suggested objective, and the go says so.
    assert sum(x.objectives for x in results) >= 1 and any(x.succeeded for x in results)


def test_runner_reports_every_learned_step_with_its_source(lab: Lab) -> None:
    from retro_ai_lab.learning import Transition

    lab.connection.connect("fake", seed=3)
    teach(lab)
    learned: list[Transition] = []
    r = runner(lab, max_steps=5, on_learned=learned.append)
    r.begin()
    while r.step():
        pass
    r.finish()
    assert len(learned) == 5 and all(t.source == "ai" for t in learned)
    # Observed steps (the child's) are reported as the child's.
    learned.clear()
    first = r.begin(restore=False)
    assert r.observe_step(["up"], 4)
    assert len(learned) == 1 and learned[0].source == "child"
    assert learned[0].action == r.last_action and learned[0].state == first
    assert learned[0].next_state == r.state
    r.finish()


def test_a_training_step_is_one_request_on_a_socket_backend(lab: Lab) -> None:
    """T-824: buttons, frames and the memory snapshot travel in one step request."""
    from retro_ai_lab.emulator.bizhawk.simulator import BizHawkSimulator

    with BizHawkSimulator() as sim:
        lab.connection.connect("bizhawk", port=sim.port, call_timeout_s=2.0, timeout_s=2.0)
        teach(lab)
        r = runner(lab, frames_per_step=3, max_steps=5)
        before = len(sim.requests)
        result = r.run_episode()
        assert result.steps == 5 and result.frames == 15
        steps = [q for q in sim.requests[before:] if q["cmd"] == "step"]
        others = [q["cmd"] for q in sim.requests[before:] if q["cmd"] != "step"]
        assert len(steps) == 5 and all(q["frames"] == 3 for q in steps)
        assert all("buttons" in q and [d["domain"] for d in q["reads"]] == ["RAM"] for q in steps)
        # Besides the steps: the opening snapshot of the episode and the release at the end.
        assert others == ["read", "input"]
        assert sim.requests[before]["cmd"] == "read"
        assert lab.observation.latest is not None and lab.observation.frame >= 15
        lab.connection.disconnect()
