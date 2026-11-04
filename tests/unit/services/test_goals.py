"""T-408: the goal service keeps a live, saved goal per game."""

from __future__ import annotations

from pathlib import Path

import pytest

from retro_ai_lab.core import Capabilities, ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.rewards import Trigger, merge_suggestions
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    GoalChanged,
    GoalService,
    ObjectiveCompleted,
    ObjectiveSetback,
    ObservationService,
)
from retro_ai_lab.storage import GoalStore
from retro_ai_lab.understanding import (
    Encoding,
    EventKind,
    Evidence,
    GameEvent,
    Knowledge,
    Relationship,
    VariableKind,
)

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]


@pytest.fixture
def bus() -> EventBus:
    return EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))


@pytest.fixture
def connection(bus: EventBus) -> ConnectionService:
    return ConnectionService(default_registry(), bus)


@pytest.fixture
def discovery(connection: ConnectionService, bus: EventBus) -> DiscoveryService:
    observation = ObservationService(connection, bus)
    return DiscoveryService(connection, observation, bus, experiment_every=None)


@pytest.fixture
def store(tmp_path: Path) -> GoalStore:
    return GoalStore(tmp_path / "goals")


@pytest.fixture
def goals(
    connection: ConnectionService, discovery: DiscoveryService, bus: EventBus, store: GoalStore
) -> GoalService:
    return GoalService(connection, discovery, bus, store=store)


@pytest.fixture
def changes(bus: EventBus) -> list[GoalChanged]:
    seen: list[GoalChanged] = []
    bus.subscribe(GoalChanged, seen.append)
    return seen


def teach(discovery: DiscoveryService, bus: EventBus) -> Knowledge:
    k = discovery.knowledge
    assert k is not None
    added = (
        k.propose(RAM, LAYOUT.lives, kind=VariableKind.LIVES, name="Lives", evidence=SURE),
        k.propose(
            RAM, LAYOUT.checkpoint, kind=VariableKind.PROGRESS, name="Checkpoint", evidence=SURE
        ),
        k.propose(RAM, LAYOUT.lap, kind=VariableKind.PROGRESS, name="Lap", evidence=SURE),
        k.propose(
            RAM,
            LAYOUT.score,
            encoding=Encoding.U16LE,
            kind=VariableKind.SCORE,
            name="Score",
            evidence=SURE,
        ),
    )
    bus.publish(DiscoveryUpdated(added, (), (), ()))
    return k


def test_no_goal_until_a_memory_capable_game_connects(
    goals: GoalService, connection: ConnectionService, changes: list[GoalChanged]
) -> None:
    assert not goals.enabled and goals.goal is None and goals.summary() == "No goal yet."
    with pytest.raises(UnsupportedCapabilityError):
        goals.set_title("Win")
    assert goals.trigger_options()[-1].label == "Starting over"
    connection.connect("fake", capabilities=Capabilities(screen_capture=True, frame_advance=True))
    assert not goals.enabled and goals.goal is None
    connection.disconnect()
    connection.connect("fake", seed=3)
    assert goals.enabled and goals.goal is not None
    assert goals.goal.game_id == "toyracer-3"
    assert goals.goal.title == "Get as far as you can in Toy Racer"
    assert goals.summary() == "Get as far as you can in Toy Racer: no steps yet."
    assert [c.reason for c in changes] == ["loaded"]


def test_new_beliefs_add_suggested_objectives(
    goals: GoalService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    changes: list[GoalChanged],
) -> None:
    connection.connect("fake", seed=3)
    teach(discovery, bus)
    goal = goals.goal
    assert goal is not None and changes[-1].reason == "suggestions"
    assert [o.description for o in goal] == ["Get further in Checkpoint", "Get further in Lap"]
    assert goals.summary().endswith("0 of 2 steps done. Next: Get further in Checkpoint.")
    assert goals.refresh_suggestions() == ()
    assert goals.saves >= 1 and goals.store is not None and goals.store.exists("toyracer-3")
    labels = [o.label for o in goals.trigger_options()]
    assert "Checkpoint goes up" in labels and "Losing Lives" in labels
    assert labels[-1] == "Starting over" and len(labels) == 6


def test_trigger_options_are_plain_words(
    goals: GoalService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
) -> None:
    """T-858: an unnamed guess is offered by its role, and only the best of each role."""
    connection.connect("fake", seed=3)
    k = discovery.knowledge
    assert k is not None
    k.propose(RAM, 40, kind=VariableKind.SCORE, evidence=[Evidence("t", "", 1.0)])
    k.propose(RAM, 41, kind=VariableKind.SCORE, evidence=SURE)
    k.propose(RAM, 42, kind=VariableKind.LIVES, evidence=SURE)
    labels = [o.label for o in goals.trigger_options()]
    assert labels == ["Your score jumps up", "Losing a life", "Gaining a life", "Starting over"]


def test_events_complete_and_penalise_objectives(
    goals: GoalService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    changes: list[GoalChanged],
) -> None:
    done: list[ObjectiveCompleted] = []
    hit: list[ObjectiveSetback] = []
    bus.subscribe(ObjectiveCompleted, done.append)
    bus.subscribe(ObjectiveSetback, hit.append)
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    lives, checkpoint, lap = (
        k.at(RAM, LAYOUT.lives),
        k.at(RAM, LAYOUT.checkpoint),
        k.at(RAM, LAYOUT.lap),
    )
    assert lives and checkpoint and lap
    goal = goals.goal
    assert goal is not None
    cp_id = next(o.id for o in goal if "Checkpoint" in o.description)
    goals.update_objective(cp_id, penalty=-4, setback_trigger=Trigger(EventKind.SETBACK, lives.id))
    before = len(changes)
    # Lap is blocked until the checkpoint is done.
    bus.publish(DiscoveryUpdated((), (), (GameEvent(EventKind.PROGRESS, 10, lap.id, ""),), ()))
    assert done == [] and len(changes) == before
    bus.publish(DiscoveryUpdated((), (), (GameEvent(EventKind.SETBACK, 20, lives.id, ""),), ()))
    assert len(hit) == 1 and hit[0].penalty == -4.0 and hit[0].frame == 20
    assert changes[-1].reason == "progress"
    bus.publish(
        DiscoveryUpdated((), (), (GameEvent(EventKind.PROGRESS, 30, checkpoint.id, ""),), ())
    )
    assert len(done) == 1 and done[0].objective.id == cp_id and done[0].reward == 25.0
    bus.publish(DiscoveryUpdated((), (), (GameEvent(EventKind.SETBACK, 40, lives.id, ""),), ()))
    assert len(hit) == 1  # no penalty once done
    bus.publish(DiscoveryUpdated((), (), (GameEvent(EventKind.PROGRESS, 50, lap.id, ""),), ()))
    assert len(done) == 2 and goal.is_complete
    assert goals.summary().endswith("2 of 2 steps done. All done!")


def test_edits_save_announce_and_are_remembered(
    goals: GoalService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    changes: list[GoalChanged],
    store: GoalStore,
) -> None:
    connection.connect("fake", seed=3)
    teach(discovery, bus)
    goals.set_title("  Win the cup  ")
    assert goals.goal is not None and goals.goal.title == "Win the cup"
    mine = goals.add_objective("Beat my best lap", Trigger(EventKind.PROGRESS), reward=40)
    assert mine.id == "obj1" and mine.reward == 40.0 and not mine.suggested
    ids = goals.goal.ids()
    goals.move_objective(mine.id, 0)
    assert goals.goal.ids()[0] == mine.id
    goals.set_dependencies(ids[0], [mine.id])
    assert goals.goal.get(ids[0]).depends_on == (mine.id,)
    goals.update_objective(mine.id, description="Beat my best", enabled=False)
    goals.mark_complete(ids[1])
    assert goals.goal.get(ids[1]).completed
    goals.mark_not_complete(ids[1])
    assert not goals.goal.get(ids[1]).completed
    goals.mark_complete(mine.id)
    goals.reset_progress()
    assert goals.goal.progress() == (0, 2)
    assert [c.reason for c in changes[-9:]] == ["edit"] * 5 + ["progress"] * 4
    saves = goals.saves
    with pytest.raises(ConfigurationError):
        goals.set_title("   ")
    with pytest.raises(ConfigurationError):
        goals.set_dependencies(mine.id, [ids[0]])  # a cycle
    with pytest.raises(ConfigurationError):
        goals.update_objective(mine.id, completed=True)
    assert goals.saves == saves  # refused edits save nothing
    goals.remove_objective(ids[1])
    assert ids[1] not in goals.goal
    connection.disconnect()
    assert goals.saves == saves + 2 and not goals.enabled and goals.goal is not None
    connection.connect("fake", seed=3)
    goal = goals.goal
    assert goal is not None and goal.title == "Win the cup"
    assert goal.ids()[:2] == (mine.id, ids[0]) and not goal.get(mine.id).enabled
    assert goal.get(mine.id).description == "Beat my best"
    assert goal.get(ids[0]).depends_on == (mine.id,)
    # Once the AI knows the game again, the removed suggestion comes back after the rest.
    assert len(goal) == 2
    teach(discovery, bus)
    assert len(goal) == 3 and goal.ids()[-1] == ids[1]
    connection.disconnect()
    connection.connect("fake", seed=4)
    assert goals.goal is not None and goals.goal.game_id == "toyracer-4" and len(goals.goal) == 0
    connection.disconnect()  # saved on disconnect even when nothing was edited
    assert set(store.games()) == {"toyracer-3", "toyracer-4"}


def test_damaged_file_starts_fresh_and_no_store_saves_nothing(
    connection: ConnectionService, discovery: DiscoveryService, bus: EventBus, store: GoalStore
) -> None:
    store.path_for("toyracer-3").write_text("{nope", "utf-8")
    goals = GoalService(connection, discovery, bus, store=store)
    connection.connect("fake", seed=3)
    assert goals.goal is not None and len(goals.goal) == 0
    connection.disconnect()
    assert store.load("toyracer-3").title.startswith("Get as far")
    bare = GoalService(connection, discovery, bus)
    connection.connect("fake", seed=3)
    bare.set_title("x")
    assert bare.save() is None and bare.saves == 0 and bare.store is None


def test_relationship_updates_refresh_suggestions_sparingly(
    connection: ConnectionService,
    discovery: DiscoveryService,
    goals: GoalService,
    bus: EventBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-842: see the reward service's test of the same name."""
    connection.connect("fake")
    k = teach(discovery, bus)
    calls: list[int] = []
    real = merge_suggestions

    def counted(goal: object, knowledge: object, **kw: object) -> tuple[str, ...]:
        calls.append(1)
        return real(goal, knowledge, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr("retro_ai_lab.services.goals.merge_suggestions", counted)
    weak = Relationship("progress:v2", "progress:v3", 1, 3, 3, 5)
    k.set_relationship(weak)
    for _ in range(10):
        bus.publish(DiscoveryUpdated((), (), (), (weak,)))
    assert calls == [1]
    sure = Relationship("progress:v3", "progress:v2", 6, 6, 6, 5)  # lap before checkpoint
    k.set_relationship(sure)
    bus.publish(DiscoveryUpdated((), (), (), (sure,)))
    bus.publish(DiscoveryUpdated((), (), (), (sure,)))
    assert len(calls) == 2
    bus.publish(DiscoveryUpdated((k.get("v2"),), (), (), ()))
    assert len(calls) == 3
