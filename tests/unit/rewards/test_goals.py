"""T-407: goals and objectives."""

from __future__ import annotations

import math
import time

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.rewards import (
    GOAL_IDEAS,
    Goal,
    Objective,
    Trigger,
    merge_suggestions,
    suggest_goal,
)
from retro_ai_lab.understanding import (
    Confidence,
    Encoding,
    EventDetector,
    EventKind,
    Evidence,
    GameEvent,
    Knowledge,
    Relationship,
    VariableKind,
)

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]
MAYBE = [Evidence("test", "", 1.0)]


def ev(kind: EventKind, var: str | None = "v1", frame: int = 1) -> GameEvent:
    return GameEvent(kind, frame, var, "")


def obj(id: str, var: str | None = "v1", **kw: object) -> Objective:
    return Objective(id, id.capitalize(), Trigger(EventKind.PROGRESS, var), **kw)  # type: ignore[arg-type]


# ------------------------------------------------------------ objectives


def test_objective_validation_and_round_trip() -> None:
    o = Objective(
        "a",
        "Get a badge",
        Trigger(EventKind.PROGRESS, "v1"),
        reward=30,
        penalty=-5,
        setback_trigger=Trigger(EventKind.SETBACK),
        confidence=Confidence.MEDIUM,
        depends_on=("b", "b", "c"),
        notes="n",
        suggested=True,
    )
    assert o.depends_on == ("b", "c") and o.reward == 30.0 and o.penalty == -5.0
    assert Objective.from_dict(o.to_dict()) == o
    plain = Objective("p", "By hand only")
    assert plain.trigger is None and Objective.from_dict(plain.to_dict()) == plain
    for bad in (
        dict(id="", description="x"),
        dict(id="a", description="  "),
        dict(id="a", description="x", reward=math.nan),
        dict(id="a", description="x", reward=1000),
        dict(id="a", description="x", penalty=5),
        dict(id="a", description="x", depends_on=("a",)),
        dict(id="a", description="x", trigger="progress"),
        dict(id="a", description="x", confidence="high"),
    ):
        with pytest.raises(ConfigurationError):
            Objective(**bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        Objective.from_dict({"id": "a"})
    with pytest.raises(ConfigurationError):
        Objective.from_dict({"id": "a", "description": "x", "trigger": {"kind": "boom"}})


# ------------------------------------------------------------ goal container


def test_goal_add_get_remove_update_move() -> None:
    g = Goal("g", title="Win")
    assert len(g) == 0 and g.new_id() == "obj1" and g.next() is None
    a = g.add(obj("a"))
    b = g.add(obj("b", depends_on=("a",)))
    assert g.ids() == ("a", "b") and g.get("a") == a and "b" in g and g.find("zz") is None
    with pytest.raises(ConfigurationError, match="already"):
        g.add(obj("a"))
    with pytest.raises(ConfigurationError):
        g.get("zz")
    updated = g.update("b", description="Second", reward=40)
    assert updated.description == "Second" and updated.reward == 40.0 and not updated.suggested
    with pytest.raises(ConfigurationError):
        g.update("b", completed=True)
    g.add(obj("c"))
    g.move("c", 0)
    assert g.ids() == ("c", "a", "b")
    with pytest.raises(ConfigurationError):
        g.move("c", 5)
    g.remove("a")
    assert g.get("b").depends_on == ()  # the dependency on a is dropped
    with pytest.raises(ConfigurationError):
        g.remove("a")
    with pytest.raises(ConfigurationError):
        Goal("")
    assert b.depends_on == ("a",)


def test_dependencies_reject_unknown_ids_and_cycles() -> None:
    g = Goal("g")
    g.add(obj("a"))
    g.add(obj("b"))
    g.add(obj("c"))
    with pytest.raises(ConfigurationError, match="Unknown"):
        g.set_dependencies("a", ["zz"])
    with pytest.raises(ConfigurationError, match="itself"):
        g.set_dependencies("a", ["a"])
    with pytest.raises(ConfigurationError, match="itself"):
        g.add(obj("d", depends_on=("d",)))
    g.set_dependencies("b", ["a"])
    g.set_dependencies("c", ["b"])
    with pytest.raises(ConfigurationError, match="depend on each other"):
        g.set_dependencies("a", ["c"])
    with pytest.raises(ConfigurationError, match="depend on each other"):
        g.add(obj("d", depends_on=("c",))) and g.set_dependencies("a", ["d"])
    assert g.get("c").depends_on == ("b",)


# ------------------------------------------------------------ progress


def test_readiness_observe_and_reset() -> None:
    g = Goal("g")
    g.add(obj("a", "v1", reward=10))
    g.add(obj("b", "v2", reward=20, depends_on=("a",)))
    g.add(obj("off", "v1", reward=99, enabled=False))
    g.add(
        obj(
            "risky",
            "v3",
            reward=5,
            penalty=-7,
            setback_trigger=Trigger(EventKind.SETBACK, "lives"),
        )
    )
    assert [o.id for o in g.ready()] == ["a", "risky"]
    assert g.next() is not None and g.next().id == "a"  # type: ignore[union-attr]
    assert g.progress() == (0, 3) and not g.is_complete
    # b is blocked until a is done; the disabled one never counts.
    out = g.observe(ev(EventKind.PROGRESS, "v2"))
    assert out.is_empty and out.reward == 0.0
    out = g.observe(ev(EventKind.PROGRESS, "v1", frame=7))
    assert out.reward == 10.0 and [o.id for o in out.completed] == ["a"]
    assert g.get("a").completed and g.get("a").completed_frame == 7
    assert g.observe(ev(EventKind.PROGRESS, "v1")).is_empty  # done once only
    assert g.is_ready("b") and g.next().id == "b"  # type: ignore[union-attr]
    out = g.observe(ev(EventKind.SETBACK, "lives"))
    assert out.reward == -7.0 and [o.id for o in out.penalised] == ["risky"]
    out = g.observe(ev(EventKind.PROGRESS, "v3"))
    assert out.reward == 5.0 and g.get("risky").completed
    assert g.observe(ev(EventKind.SETBACK, "lives")).is_empty  # no penalty once done
    g.observe(ev(EventKind.PROGRESS, "v2"))
    assert g.progress() == (3, 3) and g.is_complete and g.next() is None
    g.reset_progress()
    assert g.progress() == (0, 3) and g.get("a").completed_frame is None
    g.uncomplete("a")
    assert g.complete("a").completed


def test_goal_round_trip_with_forward_dependencies() -> None:
    g = Goal("g", title="Win", notes="n")
    g.add(obj("a"))
    g.add(obj("b", depends_on=("a",)))
    g.move("b", 0)  # b now comes first but depends on a
    g.complete("a", 3)
    data = g.to_dict()
    again = Goal.from_dict(data)
    assert again.ids() == ("b", "a") and again.title == "Win" and again.notes == "n"
    assert again.get("b").depends_on == ("a",) and again.get("a").completed_frame == 3
    with pytest.raises(ConfigurationError, match="newer"):
        Goal.from_dict({**data, "schema": 99})
    with pytest.raises(ConfigurationError):
        Goal.from_dict({"schema": 1})
    with pytest.raises(ConfigurationError):
        Goal.from_dict({**data, "objectives": [{"id": "x"}]})


# ------------------------------------------------------------ suggestions


def test_suggestions_follow_progress_variables_and_relationships() -> None:
    k = Knowledge("g", title="Game")
    lap = k.propose(RAM, 0, kind=VariableKind.PROGRESS, name="Lap", evidence=SURE)
    cp = k.propose(RAM, 1, kind=VariableKind.PROGRESS, name="Checkpoint", evidence=MAYBE)
    k.propose(RAM, 2, kind=VariableKind.SCORE, name="Score", evidence=SURE)
    k.propose(RAM, 3, kind=VariableKind.PROGRESS, name="Nope")  # no evidence
    bad = k.propose(RAM, 4, kind=VariableKind.PROGRESS, name="Rejected", evidence=SURE)
    k.reject(bad.id)
    # Checkpoints lead to laps, so checkpoints come first even though Lap was found first.
    k.set_relationship(Relationship(f"progress:{cp.id}", f"progress:{lap.id}", 4, 4, 4, 5))
    g = suggest_goal(k)
    assert g.title == GOAL_IDEAS[-1] and g.game_id == "g"
    assert [o.description for o in g] == ["Get further in Checkpoint", "Get further in Lap"]
    first, second = tuple(g)
    assert first.confidence is Confidence.LOW and second.confidence is Confidence.HIGH
    assert second.depends_on == (first.id,) and first.suggested and second.suggested
    assert g.next() is first
    assert merge_suggestions(g, k) == ()
    # A person's edit survives a later merge, and a new belief is added after the rest.
    g.update(first.id, description="Pass a flag")
    k.propose(RAM, 5, kind=VariableKind.PROGRESS, name="Stage", evidence=SURE)
    added = merge_suggestions(g, k)
    assert len(added) == 1 and g.get(added[0]).depends_on == (second.id,)
    assert g.get(first.id).description == "Pass a flag" and not g.get(first.id).suggested
    assert suggest_goal(k, title="Win it all", at_least=Confidence.HIGH).title == "Win it all"
    assert len(suggest_goal(k, at_least=Confidence.HIGH)) == 2


def test_steps_follow_the_offered_guesses() -> None:
    """T-858: steps are suggested for the best three progress guesses, in plain words."""
    k = Knowledge("g", title="Game")
    first = k.propose(RAM, 0, kind=VariableKind.PROGRESS, evidence=SURE)
    second = k.propose(RAM, 1, kind=VariableKind.PROGRESS, evidence=SURE)
    k.propose(RAM, 2, kind=VariableKind.PROGRESS, name="Lap", evidence=MAYBE)
    k.propose(RAM, 3, kind=VariableKind.PROGRESS, evidence=MAYBE)  # the fourth: not offered
    g = suggest_goal(k)
    assert [o.description for o in g] == [
        "Get further in Progress",
        "Get further in Progress (2nd guess)",
        "Get further in Lap",
    ]
    # A name a person gives moves the guess to the front and rewords its step.
    k.correct(second.id, name="World")
    assert merge_suggestions(g, k) == ()
    assert g.get(f"goal:progress:{second.id}").description == "Get further in World"
    assert g.get(f"goal:progress:{first.id}").description == "Get further in Progress"


def test_merge_retires_and_rewords_stale_suggestions() -> None:
    k = Knowledge("g", title="Game")
    cp = k.propose(RAM, 1, kind=VariableKind.PROGRESS, name="Progress at ram[1]", evidence=SURE)
    lap = k.propose(RAM, 0, kind=VariableKind.PROGRESS, name="Lap", evidence=SURE)
    g = suggest_goal(k)
    cp_obj, lap_obj = tuple(g)
    g.complete(cp_obj.id, 9)
    mine = Objective("mine", "Beat my record", trigger=Trigger(EventKind.PROGRESS, cp.id))
    g.add(mine)
    # The AI changes its mind: ram[1] was a position, and Lap gets a proper name.
    k.update(cp.id, kind=VariableKind.POSITION)
    k.update(lap.id, name="Lap number")
    assert merge_suggestions(g, k) == ()
    assert cp_obj.id not in g  # retired although it was "completed"
    assert g.get("mine").description == "Beat my record"  # the child's step is untouched
    assert g.get(lap_obj.id).description == "Get further in Lap number"
    assert g.get(lap_obj.id).suggested and g.get(lap_obj.id).depends_on == ()
    # A suggested step for a guess the knowledge no longer has at all is retired too
    # (T-938); a step the child wrote on such a guess is left alone.
    g.add(
        Objective(
            "goal:progress:ghost",
            "Get further in Ghost",
            trigger=Trigger(EventKind.PROGRESS, "ghost"),
            suggested=True,
        )
    )
    g.add(Objective("my-ghost", "Beat Ghost", trigger=Trigger(EventKind.PROGRESS, "ghost")))
    assert merge_suggestions(g, k) == ()
    assert "goal:progress:ghost" not in g and "my-ghost" in g
    assert g.get(lap_obj.id).description == "Get further in Lap number"
    # While the knowledge is empty the AI has not looked yet: nothing is retired.
    fresh = Knowledge("g", title="Game")
    assert merge_suggestions(g, fresh) == ()
    assert lap_obj.id in g and "my-ghost" in g


def test_toy_racer_goal_completes_a_checkpoint() -> None:
    k = Knowledge("toyracer")
    k.propose(RAM, LAYOUT.lives, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
    k.propose(RAM, LAYOUT.checkpoint, kind=VariableKind.PROGRESS, name="Checkpoint", evidence=SURE)
    k.propose(RAM, LAYOUT.lap, kind=VariableKind.PROGRESS, name="Lap", evidence=SURE)
    k.propose(
        RAM,
        LAYOUT.score,
        encoding=Encoding.U16LE,
        kind=VariableKind.SCORE,
        name="Score",
        evidence=SURE,
    )
    g = suggest_goal(k, title="Win the championship")
    assert {o.description for o in g} == {"Get further in Checkpoint", "Get further in Lap"}
    game = ToyRacer(seed=2)
    d = EventDetector(k)
    d.observe(MemorySnapshot(0, {RAM: game.ram}))
    earned = 0.0
    for i in range(1, 900):
        game.step(["start"] if game.state.game_over else ["up"])
        for e in d.observe(MemorySnapshot(i, {RAM: game.ram})):
            earned += g.observe(e).reward
    checkpoint = next(o for o in g if "Checkpoint" in o.description)
    assert checkpoint.completed and earned >= checkpoint.reward


def test_merge_handles_many_objectives_quickly() -> None:
    """T-842: merging and observing stay linear in the number of objectives."""
    k = Knowledge("big")
    for i in range(400):
        k.propose(RAM, i, kind=VariableKind.PROGRESS, name=f"P{i}", evidence=SURE)
    goal = Goal("big")
    started = time.perf_counter()
    added = merge_suggestions(goal, k)
    # The AI offers steps for its best three progress values (T-858), no more.
    assert len(added) == 3 and merge_suggestions(goal, k) == ()
    outcome = goal.observe(GameEvent(EventKind.PROGRESS, 1, "v1", "up", 0, 1))
    assert [o.id for o in outcome.completed] == ["goal:progress:v1"]
    # The second objective depends on the first, now done: it completes on the same pass.
    second = goal.observe(GameEvent(EventKind.PROGRESS, 2, "v2", "up", 0, 1))
    assert [o.id for o in second.completed] == ["goal:progress:v2"]
    assert goal.ready()[0].id == "goal:progress:v3"
    assert time.perf_counter() - started < 2.0
