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
from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings, State, StateEncoder
from retro_ai_lab.rewards import Trigger
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    ExperimentHarness,
    GoalService,
    ObservationService,
    RewardService,
)
from retro_ai_lab.services.rewards import RewardEarned
from retro_ai_lab.storage import GoalStore, RewardStore
from retro_ai_lab.training import (
    Earning,
    EpisodeResult,
    EpisodeRunner,
    EpisodeSignals,
    describe_earned,
)
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
    assert r._agent.recent_rewards == (result.reward,)  # exploring follows results (T-932)
    assert 1 <= result.states_seen <= 41
    ai_inputs = lab.observation.inputs.by_source("ai")
    assert ai_inputs and all(e.source == "ai" for e in ai_inputs)
    space = r._agent.space
    assert all(space.matches(e.buttons).buttons == e.buttons for e in ai_inputs)
    assert lab.observation.inputs.held_at(lab.observation.frame) == frozenset()  # released
    assert not r.running


def test_a_go_records_its_lives_lost(lab: Lab) -> None:
    """T-934: how many lives went and at which step the first did, for the log."""
    lab.connection.connect("fake", seed=3)
    k = teach(lab)
    lives = k.at(RAM, LAYOUT.lives)
    assert lives is not None
    r = runner(lab, frames_per_step=4, max_steps=10)
    r.begin()
    crash = GameEvent(EventKind.SETBACK, 20, lives.id, "", 3, 2)
    lab.bus.publish(DiscoveryUpdated((), (), (crash,), ()))
    assert r.step()  # step 1 collects the loss
    for _ in range(3):
        assert r.step()
    lab.bus.publish(DiscoveryUpdated((), (), (crash, crash), ()))
    while r.step():
        pass
    result = r.finish()
    # The toy racer may lose lives of its own in those steps, so at least the three sent.
    assert result.setbacks >= 3 and result.first_setback_step == 1
    assert result.describe_losses().startswith(", lost a life ")
    assert result.describe_losses().endswith(" times, the first at step 1")
    assert EpisodeResult.from_dict(result.to_dict()) == result
    clean = EpisodeResult(5, 20, 0.0, "max_steps", 2)
    assert clean.setbacks == 0 and clean.first_setback_step is None
    assert clean.describe_losses() == ""
    old = {"steps": 1, "frames": 4, "reward": 0.0, "ended_by": "stopped", "states_seen": 1}
    assert EpisodeResult.from_dict(old).setbacks == 0  # a go saved before T-934
    assert EpisodeResult.from_dict(old).first_setback_step is None
    once = EpisodeResult(5, 20, 0.0, "max_steps", 2, setbacks=1, first_setback_step=4)
    assert once.describe_losses() == ", lost a life once, the first at step 4"


def test_game_over_is_unstuck_with_start_and_the_restart_ends_the_episode(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3)
    teach(lab)
    lab.observation.observe()  # the engine sees the fresh game first
    game = lab.adapter().game
    while not game.state.game_over:  # drive into game over by hand
        game.step(["up"])
    r = runner(lab, frames_per_step=4, max_steps=500, unstick_after=10, unstick_buttons=("start",))
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


def test_only_the_best_life_count_s_losses_are_counted(lab: Lab) -> None:
    """T-940: the user's log added up every life-count guess's drops: 45 "lives" a go."""
    lab.connection.connect("fake", seed=3)
    k = teach(lab)
    lives = k.at(RAM, LAYOUT.lives)
    assert lives is not None
    timer = k.propose(RAM, LAYOUT.timer, kind=VariableKind.LIVES, evidence=SURE)  # a stray
    k.confirm(lives.id)  # your word outranks any evidence
    r = runner(lab, frames_per_step=4, max_steps=6)
    assert r.signals.watched is None  # on their own the signals count every value
    r.begin()
    assert r.signals.watched == frozenset({lives.id})
    r.signals.take_setbacks()  # whatever the restore brought
    stray = GameEvent(EventKind.SETBACK, 20, timer.id, "", 9, 8)
    real = GameEvent(EventKind.SETBACK, 20, lives.id, "", 3, 2)
    lab.bus.publish(DiscoveryUpdated((), (), (stray, stray), ()))
    assert r.signals.take_setbacks() == 0  # the stray guess does not count
    lab.bus.publish(DiscoveryUpdated((), (), (real, stray), ()))
    assert r.signals.take_setbacks() == 1
    r.signals.watch_setbacks(None)
    lab.bus.publish(DiscoveryUpdated((), (), (stray,), ()))
    assert r.signals.take_setbacks() == 1
    r.finish()


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
    # No wake button is known here, so a go on the game-over screen ends "stuck" (T-844).
    assert all(x.ended_by in ("max_steps", "reset", "stuck") for x in results)
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


# ------------------------------------------------------------ T-844


def test_a_stuck_go_ends_when_no_wake_button_is_known(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3)
    teach(lab)
    lab.observation.observe()
    game = lab.adapter().game
    while not game.state.game_over:
        game.step(["up"])
    r = runner(lab, frames_per_step=4, max_steps=500, unstick_after=10)
    assert r.unstick_buttons == ()
    result = r.run_episode()
    assert result.ended_by == "stuck" and r.unsticks == 0 and r.stuck_from_start
    assert 10 <= result.steps <= 12
    assert not any(e.buttons == {"start"} for e in lab.observation.inputs.by_source("ai"))
    # Once the service has found the button, the runner presses it instead.
    r.set_unstick_buttons(("start",))
    assert r.unstick_buttons == ("start",)
    while not game.state.game_over:
        game.step(["up"])
    result = r.run_episode()
    assert result.ended_by == "reset" and r.unsticks == 1


def test_goes_carry_a_fingerprint(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3)
    teach(lab)
    lab.observation.observe()
    r = runner(lab, frames_per_step=4, max_steps=20)
    assert r.fingerprint == () and not r.actions_varied
    result = r.run_episode()
    assert len(r.fingerprint) == result.steps == 20
    assert r.actions_varied  # epsilon 0.5: more than one action in 20 steps
    r.begin()
    assert r.fingerprint == () and not r.actions_varied


def test_a_go_stuck_part_way_is_not_stuck_from_the_start(lab: Lab) -> None:
    lab.connection.connect("fake", seed=3)
    teach(lab)
    lab.observation.observe()
    r = runner(lab, frames_per_step=4, max_steps=2000, unstick_after=10)
    result = r.run_episode()  # drives until the lives run out, then sits on game over
    assert result.ended_by == "stuck" and not r.stuck_from_start
    assert result.steps > 20


def test_signals_keep_the_ids_a_reset_blamed(lab: Lab) -> None:
    signals = EpisodeSignals(lab.bus)
    signals.start()
    reset = GameEvent(EventKind.RESET, 7, None, WENT_HOME, variables=("v3", "v4"))
    lab.bus.publish(DiscoveryUpdated((), (), (reset,), ()))
    assert signals.last_reset_ids == ("v3", "v4")
    signals.start()
    assert signals.last_reset_ids == ()


def test_signals_say_what_earned_the_points(lab: Lab) -> None:
    """T-857: the points are split by the behaviour (or goal step) that paid them."""
    signals = EpisodeSignals(lab.bus)
    lab.connection.connect("fake", seed=3)
    k = teach(lab)
    score = k.at(RAM, LAYOUT.score)
    assert score
    profile = lab.rewards.profile
    assert profile is not None
    jump = profile.find(f"score_gain:{score.id}")
    assert jump is not None
    gain = GameEvent(EventKind.SCORE_GAIN, 5, score.id, "")
    lab.bus.publish(DiscoveryUpdated((), (), (gain,), ()))
    assert not signals.earned  # not listening yet
    signals.start()
    lab.bus.publish(DiscoveryUpdated((), (), (gain,), ()))
    lab.bus.publish(DiscoveryUpdated((), (), (gain,), ()))
    assert [(e.id, e.name, e.count, e.amount) for e in signals.earned] == [
        (jump.id, jump.name, 2, 2 * jump.value)
    ]
    # A completed step is credited by its description; a setback by the same key.
    step = lab.goals.add_objective("Score some points", Trigger(EventKind.SCORE_GAIN, score.id))
    lab.bus.publish(DiscoveryUpdated((), (), (gain,), ()))
    earned = signals.earned
    assert earned[0].id == f"step:{step.id}" and earned[0].name == "Score some points"
    assert earned[0].count == 1 and earned[0].amount == step.reward
    assert earned[1].id == jump.id and earned[1].count == 3
    signals.stop()
    lab.bus.publish(DiscoveryUpdated((), (), (gain,), ()))
    assert signals.earned[1].count == 3  # ignored while not listening
    signals.start()
    assert not signals.earned  # a new go starts clean


def test_shares_are_credited_not_split_evenly(lab: Lab) -> None:
    """T-857: a reward event naming shares credits each behaviour what it gave."""
    signals = EpisodeSignals(lab.bus)
    signals.start()
    event = GameEvent(EventKind.PROGRESS, 9, "v1", "")
    lab.bus.publish(RewardEarned(9, 12.0, ("a", "b"), event, ("Alpha", "Beta"), (10.0, 2.0)))
    lab.bus.publish(RewardEarned(10, 4.0, ("c", "d"), event))  # no shares: split evenly
    assert [(e.name, e.count, e.amount) for e in signals.earned] == [
        ("Alpha", 1, 10.0),
        ("Beta", 1, 2.0),
        ("c", 1, 2.0),
        ("d", 1, 2.0),
    ]


def test_earnings_are_worded_and_saved() -> None:
    """T-857: at most three earners, biggest first, the rest counted; a result round-trips."""
    earned = (
        Earning("s", "Score goes up", 120, 2500.0),
        Earning("p", "Checkpoint", 1, 40.0),
        Earning("l", "Losing Lives", 3, -30.0),
        Earning("f", "Flag on", 2, 4.0),
        Earning("g", "Gaining Lives", 1, 2.0),
    )
    assert describe_earned(earned) == (
        "Score goes up +2,500.0 (120 times), Checkpoint +40.0 (once), "
        "Losing Lives -30.0 (3 times), and 2 more"
    )
    assert describe_earned(earned, most=5).endswith("Gaining Lives +2.0 (once)")
    assert describe_earned(earned, most=None) == describe_earned(earned, most=5)  # T-877
    assert "more" not in describe_earned(earned, most=None)
    with pytest.raises(ConfigurationError):
        describe_earned(earned, most=0)
    assert describe_earned(()) == ""
    with pytest.raises(ConfigurationError):
        describe_earned(earned, most=0)
    result = EpisodeResult(10, 40, 2516.0, "max_steps", 3, 1, earned)
    assert EpisodeResult.from_dict(result.to_dict()) == result
    old = {"steps": 1, "frames": 4, "reward": 0.0, "ended_by": "reset", "states_seen": 1}
    assert EpisodeResult.from_dict(old).earned == ()
    with pytest.raises(ConfigurationError):
        EpisodeResult.from_dict({**old, "earned": [{"id": "x"}]})


def _recording(agent: QLearningAgent, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    real = agent.learn

    def learn(
        state: State, action: int, reward: float, next_state: State, done: bool = False, **kw: int
    ) -> float:
        calls.append(
            dict(state=state, action=action, reward=reward, next=next_state, done=done, **kw)
        )
        return real(state, action, reward, next_state, done, **kw)

    monkeypatch.setattr(agent, "learn", learn)
    return calls


def test_the_runner_learns_each_step_from_the_return_over_the_lookahead(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-963: n-step returns, flushed with shorter spans when the go ends."""
    from retro_ai_lab.learning import Transition
    from retro_ai_lab.training.episode import LOOKAHEAD_STEPS

    assert LOOKAHEAD_STEPS == 8
    lab.connection.connect("fake", seed=3)
    teach(lab)
    with pytest.raises(ConfigurationError):
        runner(lab, lookahead=0)
    steps: list[Transition] = []
    r = runner(lab, frames_per_step=4, max_steps=5, lookahead=3, on_learned=steps.append)
    agent = r._agent
    assert isinstance(agent, QLearningAgent)
    g = agent.settings.gamma
    calls = _recording(agent, monkeypatch)
    r.begin()
    assert r.step() and r.step() and calls == [] and len(steps) == 2  # pending
    assert r.step() and len(calls) == 1  # the lookahead is full: the first step is learned
    while r.step():
        pass
    assert len(calls) == 3 and agent.updates == 3
    r.finish()
    assert len(calls) == 5 and agent.updates == 5  # the last two, over 2 and 1 steps
    spans = (3, 3, 3, 2, 1)
    for i, (call, span) in enumerate(zip(calls, spans, strict=True)):
        covered = steps[i : i + span]
        assert call["state"] == steps[i].state and call["action"] == steps[i].action
        assert call["steps"] == span and call["done"] is False
        assert call["next"] == covered[-1].next_state
        assert call["reward"] == pytest.approx(sum(g**k * t.reward for k, t in enumerate(covered)))
    # A lookahead of 1 learns every step at once, as before.
    r1 = runner(lab, frames_per_step=4, max_steps=3, lookahead=1)
    calls1 = _recording(r1._agent, monkeypatch)  # type: ignore[arg-type]
    r1.begin()
    assert r1.step() and len(calls1) == 1 and calls1[0]["steps"] == 1


def test_a_reset_flushes_the_pending_steps_as_final(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-963: the game starting over ends every pending step's return, with no future."""
    lab.connection.connect("fake", seed=3)
    teach(lab)
    r = runner(lab, frames_per_step=4, max_steps=30, lookahead=5)
    calls = _recording(r._agent, monkeypatch)  # type: ignore[arg-type]
    r.begin()
    assert r.step() and r.step() and calls == []
    lab.bus.publish(
        DiscoveryUpdated((), (), (GameEvent(EventKind.RESET, 42, None, WENT_HOME),), ())
    )
    assert not r.step()
    assert [c["steps"] for c in calls] == [3, 2, 1] and all(c["done"] for c in calls)
    assert all(c["next"] == r.state for c in calls)
    assert r.finish().ended_by == "reset" and len(calls) == 3


def test_a_drop_of_the_value_you_penalise_counts_as_a_death(lab: Lab) -> None:
    """T-967: the user's survival line said 0.0 lives lost a go while "Lifes goes down"
    fired 1.3 times a go: a person's value drops as a DOWN event, not as a SETBACK, which
    only the AI's life-count and health guesses raise."""
    lab.connection.connect("fake", seed=3)
    k = teach(lab)
    lives = k.at(RAM, LAYOUT.lives)
    assert lives is not None
    fuel = k.propose(RAM, LAYOUT.timer, kind=VariableKind.COUNTER, evidence=SURE)  # the person's
    r = runner(lab, frames_per_step=4, max_steps=6)
    r.begin()
    r.signals.watch_setbacks(frozenset({fuel.id}))
    r.signals.take_setbacks()
    fell = GameEvent(EventKind.DOWN, 20, fuel.id, "", 9, 8)
    rose = GameEvent(EventKind.UP, 20, fuel.id, "", 8, 9)
    other = GameEvent(EventKind.DOWN, 20, lives.id, "", 3, 2)
    setback = GameEvent(EventKind.SETBACK, 20, fuel.id, "", 9, 8)
    lab.bus.publish(DiscoveryUpdated((), (), (fell, rose, other), ()))
    assert r.signals.take_setbacks() == 1  # the drop counts; a rise and another value do not
    # A value the AI also takes for a life count drops as both events: one death.
    lab.bus.publish(DiscoveryUpdated((), (), (fell, setback), ()))
    assert r.signals.take_setbacks() == 1
    # Two updates, two deaths.
    lab.bus.publish(DiscoveryUpdated((), (), (fell,), ()))
    lab.bus.publish(DiscoveryUpdated((), (), (fell,), ()))
    assert r.signals.take_setbacks() == 2
    # Left to its own guesses, the count follows SETBACK events only, as before.
    r.signals.watch_setbacks(None)
    lab.bus.publish(DiscoveryUpdated((), (), (fell, other), ()))
    assert r.signals.take_setbacks() == 0
    lab.bus.publish(DiscoveryUpdated((), (), (setback,), ()))
    assert r.signals.take_setbacks() == 1
