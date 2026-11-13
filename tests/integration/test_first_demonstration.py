"""T-606: the first working demonstration (DESIGN.md section 17) on the Fake Emulator.

One long test walks the ten steps the design asks for, through the same services
the user interface drives, with nothing taught by hand: the sandbox connects to a
game it has never seen, works out what matters, lets the rewards be configured,
trains, explains itself, and then shows that different rewards make it play
differently.

Steps 1 to 9 are walked once for each of a few learner seeds, and step 10's claims
are about all of them together (T-992). They used to be about one seeded run: thirty
goes of a learner that explores at random are mostly luck, and the "twice as fast"
claim needed its seed moved twice, and held on between none and four of six seeds
depending on what the AI's eyes told its brain that week. Summed over three seeds the
claims below held for every three of the six seeds measured (1 to 5 and 7): the
point-lover scores 1.7 to 2.7 times as much per step and drives faster, and the
careful one's crashes per step fall to between 0.47 and 0.69 of what they were.

Since T-1010 the AI goes back at a life lost and looks for a way past. On this game
that seldom finds one (the eyes count hardly any new ground here) and it gives up
after eight searches for nothing, but the one or two ways a training does find come
in its first goes and send the rest of it another way. Measured with that on, the
claims held over the six seeds together (the point-lover 1.5 times the score a step,
crashes down to 0.49) and for 17 of the 20 threes of them; all three that failed had
seed 3 in them, where two early ways taught the point-lover to brake. Three seeds no
longer tell the claims from luck, so all six are walked. The claims are unchanged.
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
EXPLORE_FRAMES = 2500  # and a brake: the AI's checkpoint must not be a dying moment (T-961)
GOES = 30
SEEDS = (1, 2, 3, 4, 5, 7)  # the learner's; the game is the same one each time
MAX_STEPS = 150
WINDOW = 10


def lab(tmp_path: Path, name: str, seed: int) -> AppContext:
    ctx = AppContext(data_dir=tmp_path / name)
    ctx.training = TrainingService(
        ctx.connection,
        ctx.observation,
        ctx.discovery,
        ctx.bus,
        max_steps=MAX_STEPS,
        seed=seed,
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
    # The child brakes before handing over: a car still moving when the AI's checkpoint
    # is taken crashes on its own at step 1, and the AI would start past that (T-961).
    ctx.connection.call(lambda a: a.set_input([BY_NAME["b"]]))
    ctx.observation.record_input(["b"])
    ctx.connection.call(lambda a: a.advance_frames(1))
    ctx.observation.advance(1)


@dataclass
class Played:
    """How the AI played, one entry per go."""

    rewards: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)  # most points held in the go
    speeds: list[float] = field(default_factory=list)  # mean speed over the go
    crashes: list[int] = field(default_factory=list)  # lives lost in the go

    def crashes_per_step(self, goes: slice) -> float:
        return sum(self.crashes[goes]) / sum(self.steps[goes])

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
        played.crashes.append(sum(e.count for e in result.earned if e.id.startswith("setback:")))
    t.stop()
    return played


@dataclass
class Shown:
    """What one walk through steps 1 to 9 showed, for step 10."""

    lover: tuple[float, float, float, float]  # reward, steps, score, speed (last goes)
    careful: tuple[float, float, float, float]
    careful_played: Played


def walk(tmp_path: Path, seed: int) -> Shown:
    ctx = lab(tmp_path, "demo", seed)

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
    careful_played = train(ctx, GOES)
    careful = careful_played.recent()

    # Everything the child did was saved and comes back after a reconnect.
    ctx.connection.disconnect()
    ctx.connection.connect("fake", seed=5)
    assert ctx.training.episodes == GOES  # the forgotten brain stays forgotten
    profile = ctx.rewards.profile
    assert profile is not None and profile.get(setback.id).value == -50
    goal = ctx.goals.goal
    assert goal is not None and objective.id in goal
    ctx.connection.disconnect()
    return Shown(score_lover, careful, careful_played)


@pytest.mark.slow
def test_first_working_demonstration(tmp_path: Path) -> None:
    shown = [walk(tmp_path / f"seed{seed}", seed) for seed in SEEDS]

    # 10. Changing the rewards changed how it plays. The point-lover's rewards add up
    #     above zero and the careful one's below, on every seed.
    for one in shown:
        assert one.lover[0] > 0 > one.careful[0], (one.lover, one.careful)
    #     The point-lover scores more per step (the careful one's total can creep past
    #     the lover's, which is why the score is compared per step; go length is not
    #     compared: a crawler's goes end as "stuck", so steps measure the give-up rule,
    #     not caution; T-875), and drives faster.
    lover_score = sum(one.lover[2] for one in shown) / sum(one.lover[1] for one in shown)
    careful_score = sum(one.careful[2] for one in shown) / sum(one.careful[1] for one in shown)
    assert lover_score > careful_score, [(one.lover, one.careful) for one in shown]
    assert sum(one.lover[3] for one in shown) > sum(one.careful[3] for one in shown)
    #     And the careful one, with crashing at -50, crashes less and less as it learns:
    #     at least a quarter fewer per step in its last goes than in its first.
    early, late = slice(0, WINDOW), slice(-WINDOW, None)

    def crashes_per_step(goes: slice) -> float:
        played = [one.careful_played for one in shown]
        return sum(sum(p.crashes[goes]) for p in played) / sum(sum(p.steps[goes]) for p in played)

    assert crashes_per_step(late) < 0.75 * crashes_per_step(early), [
        (one.careful_played.crashes, one.careful_played.steps) for one in shown
    ]
