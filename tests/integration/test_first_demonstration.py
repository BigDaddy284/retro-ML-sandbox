"""T-606: the first working demonstration (DESIGN.md section 17) on the Fake Emulator.

One long test walks the ten steps the design asks for, through the same services
the user interface drives, with nothing taught by hand: the sandbox connects to a
game it has never seen, works out what matters, lets the rewards be configured,
trains, explains itself, and then shows that different rewards make it play
differently.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, LAYOUT
from retro_ai_lab.learning import Settings
from retro_ai_lab.rewards import Trigger
from retro_ai_lab.training import Mode, TrainingService, TrainingStepped
from retro_ai_lab.training.insights import describe_changes, describe_experiments
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.understanding import EventKind, VariableKind

BY_NAME = {b.name: b for b in BUTTONS}
EXPLORE_FRAMES = 2400
GOES = 30
MAX_STEPS = 150
WINDOW = 10


def lab(tmp_path: Path, name: str) -> AppContext:
    ctx = AppContext(data_dir=tmp_path / name)
    ctx.training = TrainingService(
        ctx.connection,
        ctx.observation,
        ctx.discovery,
        ctx.bus,
        max_steps=MAX_STEPS,
        seed=7,
        settings=Settings(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9),
        store=ctx.brain_store,
    )
    return ctx


def explore(ctx: AppContext, frames: int) -> None:
    """A child having a first go: mostly accelerating, steering now and then."""
    adapter = ctx.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    for i in range(frames):
        held = [BY_NAME["start"]] if adapter.game.state.game_over else [BY_NAME["up"]]
        if i % 50 in range(10, 13):
            held.append(BY_NAME["left"])
        if i % 90 in range(40, 43):
            held.append(BY_NAME["right"])
        if i % 200 in range(150, 160):
            held = [BY_NAME["down"]]
        ctx.connection.call(lambda a, h=held: a.set_input(h))
        ctx.observation.record_input([b.name for b in held])
        ctx.connection.call(lambda a: a.advance_frames(1))
        ctx.observation.advance(1)


@dataclass
class Played:
    """How the AI played, one entry per go."""

    rewards: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)  # most points held in the go
    speeds: list[float] = field(default_factory=list)  # mean speed over the go

    def recent(self, last: int = WINDOW) -> tuple[float, float, float, float]:
        """Means over the last ``last`` goes: reward, steps, score, speed."""
        return tuple(  # type: ignore[return-value]
            statistics.fmean(v[-last:])
            for v in (self.rewards, self.steps, self.scores, self.speeds)
        )


def train(ctx: AppContext, goes: int) -> Played:
    """Train ``goes`` goes from the checkpoint and report how each one went."""
    adapter = ctx.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    t = ctx.training
    played = Played()
    t.start(Mode.INDEPENDENT)
    while t.episodes < goes:
        n = t.episodes
        speeds: list[int] = []
        best = 0
        while t.episodes == n:
            t.step()
            speeds.append(adapter.game.state.speed)
            best = max(best, adapter.game.state.score)
        result = t.history[-1].result
        played.rewards.append(result.reward)
        played.steps.append(result.steps)
        played.scores.append(best)
        played.speeds.append(statistics.fmean(speeds))
    t.stop()
    return played


@pytest.mark.slow
def test_first_working_demonstration(tmp_path: Path) -> None:
    ctx = lab(tmp_path, "demo")

    # 1. Connect to an emulator.
    session = ctx.connection.connect("fake", seed=5)
    assert ctx.connection.is_connected and session.game_info.title == "Toy Racer"

    # 2. Observe the game, 3. identify the controller inputs, 4. examine memory.
    assert session.capabilities.memory_read and session.capabilities.input
    assert {b.name for b in session.buttons} >= {"up", "down", "left", "right", "start"}
    explore(ctx, EXPLORE_FRAMES)
    adapter = ctx.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    starting_line = adapter.game.snapshot()  # both AIs below start from here
    assert ctx.observation.frame >= EXPLORE_FRAMES and ctx.observation.tracker.observed > 100
    latest = ctx.observation.latest
    assert latest is not None and len(latest.regions["ram"]) == 32
    assert ctx.observation.tracker.classify()["state"]  # some bytes carry game state

    # 5. Identify useful game state: lives, score and speed found without any teaching.
    k = ctx.discovery.knowledge
    assert k is not None
    found = {v.offset: v.kind for v in k if not v.rejected}
    assert found[LAYOUT.lives] is VariableKind.LIVES
    assert found[LAYOUT.score] is VariableKind.SCORE
    assert found[LAYOUT.speed] is VariableKind.SPEED
    assert found[LAYOUT.frame] is VariableKind.COUNTER  # and the frame counter is not "progress"
    assert ctx.training.encoder.refresh() and all(
        v.kind is not VariableKind.COUNTER for v in ctx.training.encoder.variables
    )

    # 6. The child configures rewards: the suggestions are there to be changed.
    profile = ctx.rewards.profile
    assert profile is not None
    score_gain = next(b for b in profile if b.id.startswith("score_gain"))
    setback = next(b for b in profile if b.id.startswith("setback"))
    assert score_gain.value > 0 > setback.value
    ctx.rewards.set_value(score_gain.id, 20)
    ctx.rewards.set_value(setback.id, 0)
    assert profile.get(score_gain.id).value == 20 and profile.get(setback.id).value == 0

    # 7. Train towards a simple objective the child wrote: score some points.
    score_var = k.at("ram", LAYOUT.score)
    assert score_var is not None
    goal = ctx.goals.goal
    assert goal is not None
    objective = ctx.goals.add_objective(
        "Score some points", Trigger(EventKind.SCORE_GAIN, score_var.id)
    )
    for o in goal:
        if o.suggested:  # the AI's own guesses must not get in the way of the child's step
            ctx.goals.update_objective(o.id, enabled=False)
    decisions: list[TrainingStepped] = []
    ctx.bus.subscribe(TrainingStepped, decisions.append)
    played = train(ctx, GOES)
    assert ctx.training.episodes == GOES and len(played.rewards) == GOES
    assert goal.get(objective.id).completed
    assert sum(r.result.objectives for r in ctx.training.history) >= 1

    # 8. Its decisions and learning progress are on show.
    assert len(decisions) == sum(played.steps)
    assert any(d.decision is not None and d.decision.explored for d in decisions)
    assert any(d.decision is not None and not d.decision.explored for d in decisions)
    curve = ctx.training.curve(WINDOW)
    assert curve.attempts == GOES and curve.best is not None
    assert "go" in curve.describe()
    changes, goes_ago, known = ctx.training.behaviour_changes(WINDOW)
    assert goes_ago == WINDOW and known > 0
    assert describe_changes(changes, known, goes_ago, k, ctx.training.space)
    assert ctx.training.summary() and ctx.training.agent is not None
    assert describe_experiments(ctx.training.experiments(None, goal, profile))
    score_lover = played.recent()
    assert score_lover[2] > 0  # it picks up points at all

    # 9. Reward values can be changed: now crashing hurts and points mean nothing,
    #    not even the child's "score some points" step.
    ctx.training.forget()
    ctx.rewards.set_enabled(score_gain.id, False)
    ctx.rewards.set_value(setback.id, -50)
    ctx.goals.update_objective(objective.id, enabled=False)
    # By now the AI also understands checkpoints and laps (T-709) and has suggested
    # rewards and goal steps for them; the child switches every carrot off too.
    for behaviour in profile:
        if behaviour.enabled and behaviour.value > 0:
            ctx.rewards.set_enabled(behaviour.id, False)
    for o in goal:
        if o.enabled and o.id != objective.id:
            ctx.goals.update_objective(o.id, enabled=False)
    assert not profile.get(score_gain.id).enabled and profile.get(setback.id).value == -50
    assert all(b.value <= 0 for b in profile if b.enabled)
    adapter.game.restore(starting_line)
    careful = train(ctx, GOES).recent()

    # 10. Changing the rewards changed how it plays: the point-lover drives fast,
    #     scores more per step and runs out of lives sooner; the careful one crawls
    #     and lasts (long enough that its total can creep past the lover's, which is
    #     why the score is compared per step).
    lover_reward, lover_steps, lover_score, lover_speed = score_lover
    careful_reward, careful_steps, careful_score, careful_speed = careful
    assert lover_speed >= 2 * careful_speed, (score_lover, careful)
    assert lover_score / lover_steps > careful_score / careful_steps, (score_lover, careful)
    assert lover_steps < careful_steps, (score_lover, careful)
    assert lover_reward > 0 > careful_reward, (score_lover, careful)

    # Everything the child did was saved and comes back after a reconnect.
    ctx.connection.disconnect()
    ctx.connection.connect("fake", seed=5)
    assert ctx.training.episodes == GOES  # the forgotten brain stays forgotten
    profile = ctx.rewards.profile
    assert profile is not None and profile.get(setback.id).value == -50
    goal = ctx.goals.goal
    assert goal is not None and objective.id in goal
