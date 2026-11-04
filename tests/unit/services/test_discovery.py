"""T-306: the discovery runner learns while the game is played."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core import Capabilities, ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryProgress,
    DiscoveryService,
    DiscoveryUpdated,
    ObservationService,
)
from retro_ai_lab.services.discovery import MOVES_ALONE, game_id_for
from retro_ai_lab.understanding import EventKind, VariableKind
from retro_ai_lab.understanding.events import CYCLES

BY_NAME = {b.name: b for b in BUTTONS}


@pytest.fixture
def bus() -> EventBus:
    return EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))


@pytest.fixture
def connection(bus: EventBus) -> ConnectionService:
    return ConnectionService(default_registry(), bus)


@pytest.fixture
def observation(connection: ConnectionService, bus: EventBus) -> ObservationService:
    return ObservationService(connection, bus)


@pytest.fixture
def discovery(
    connection: ConnectionService, observation: ObservationService, bus: EventBus
) -> DiscoveryService:
    return DiscoveryService(connection, observation, bus, review_every=100, experiment_every=250)


@pytest.fixture
def progress(bus: EventBus) -> list[DiscoveryProgress]:
    seen: list[DiscoveryProgress] = []
    bus.subscribe(DiscoveryProgress, seen.append)
    return seen


@pytest.fixture
def updates(bus: EventBus) -> list[DiscoveryUpdated]:
    seen: list[DiscoveryUpdated] = []
    bus.subscribe(DiscoveryUpdated, seen.append)
    return seen


def play(connection: ConnectionService, observation: ObservationService, frames: int) -> None:
    """Drive like a child would: mostly accelerate, weave a bit, restart after crashes."""
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    for i in range(frames):
        if adapter.game.state.game_over:
            held = [BY_NAME["start"]]
        else:
            held = [BY_NAME["up"]]
            if i % 90 in range(20, 24):
                held.append(BY_NAME["left"])
            if i % 130 in range(40, 44):
                held.append(BY_NAME["right"])
        connection.call(lambda a, held=held: a.set_input(held))
        connection.call(lambda a: a.advance_frames(1))
        observation.advance(1)


def test_idle_until_a_memory_capable_game_connects(
    discovery: DiscoveryService, connection: ConnectionService, progress: list[DiscoveryProgress]
) -> None:
    assert not discovery.enabled and discovery.knowledge is None
    assert discovery.summary() == "Not watching a game."
    with pytest.raises(UnsupportedCapabilityError):
        discovery.review()
    connection.connect("fake", capabilities=Capabilities(screen_capture=True, frame_advance=True))
    assert not discovery.enabled and discovery.stage == "idle"
    assert "memory" in progress[-1].detail
    connection.disconnect()
    connection.connect("fake", seed=3)
    assert discovery.enabled and discovery.stage == "watching"
    assert discovery.knowledge is not None
    assert discovery.knowledge.game_id == "toyracer-3" and discovery.knowledge.title == "Toy Racer"
    assert discovery.summary().startswith("Watching 0 values")


def test_game_id_prefers_the_rom_hash() -> None:
    assert game_id_for("NES", "Game", "abc") == "abc"
    assert game_id_for("NES", "Game", None) == "NES:Game"


def test_reviews_propose_variables_on_schedule(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    updates: list[DiscoveryUpdated],
    progress: list[DiscoveryProgress],
) -> None:
    connection.connect("fake", seed=2)
    play(connection, observation, 99)
    assert discovery.reviews == 0
    play(connection, observation, 1)
    assert discovery.reviews == 1
    k = discovery.knowledge
    assert k is not None and len(k) >= 5
    assert any(u.added for u in updates)
    assert [p.stage for p in progress[-2:]] == ["thinking", "watching"]
    play(connection, observation, 100)
    assert discovery.reviews == 2
    assert "Watching" in discovery.summary()


def test_events_and_relationships_flow_while_playing(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    updates: list[DiscoveryUpdated],
) -> None:
    connection.connect("fake", seed=2)
    play(connection, observation, 1200)
    assert discovery.events.total > 0
    kinds = {e.kind for e in discovery.events}
    assert EventKind.PROGRESS in kinds or EventKind.SCORE_GAIN in kinds
    k = discovery.knowledge
    assert k is not None and k.relationships()
    assert any(u.events for u in updates) and any(u.relationships for u in updates)


def test_experiments_run_on_schedule_and_on_demand(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    progress: list[DiscoveryProgress],
) -> None:
    connection.connect("fake", seed=5)
    assert discovery.can_experiment()
    play(connection, observation, 249)
    assert discovery.experiments == 0
    play(connection, observation, 1)
    assert discovery.experiments == 1
    assert any(p.stage == "experimenting" for p in progress)
    k = discovery.knowledge
    assert k is not None
    mask = k.at("ram", LAYOUT.input)
    assert mask is not None and mask.kind is VariableKind.INPUT
    lane = k.at("ram", LAYOUT.lane)
    assert lane is not None and lane.kind is VariableKind.POSITION
    update = discovery.experiment()
    assert discovery.experiments == 2
    assert update.is_empty or update.changed or update.added  # nothing new is fine
    assert discovery.stage == "watching"


def test_scheduled_experiments_can_be_held_back(
    discovery: DiscoveryService, connection: ConnectionService, observation: ObservationService
) -> None:
    """T-839: training holds the scheduled experiments back; on demand still works."""
    connection.connect("fake", seed=5)
    assert discovery.experiments_allowed
    discovery.set_experiments_allowed(False)
    play(connection, observation, 300)
    assert discovery.experiments == 0
    discovery.experiment()  # the panel's "Try the buttons" is not held back
    assert discovery.experiments == 1
    play(connection, observation, 250)
    assert discovery.experiments == 1
    assert discovery.experiment_due
    assert discovery.run_due_experiment() and discovery.experiments == 2  # between goes
    assert not discovery.experiment_due and not discovery.run_due_experiment()
    play(connection, observation, 250)
    assert discovery.experiments == 2  # still held back
    discovery.set_experiments_allowed(True)
    play(connection, observation, 1)
    assert discovery.experiments == 3
    discovery.set_experiments_allowed(False)
    connection.disconnect()
    connection.connect("fake", seed=5)
    assert discovery.experiments_allowed  # a new link starts afresh


def test_experiments_unavailable_without_save_states(
    discovery: DiscoveryService, connection: ConnectionService, observation: ObservationService
) -> None:
    connection.connect(
        "fake",
        capabilities=Capabilities(
            memory_read=True, input=True, frame_advance=True, screen_capture=True
        ),
    )
    assert not discovery.can_experiment()
    with pytest.raises(UnsupportedCapabilityError):
        discovery.experiment()
    play(connection, observation, 300)  # schedule passes quietly
    assert discovery.experiments == 0 and discovery.reviews == 3


def test_disconnect_goes_idle_and_reconnect_starts_fresh(
    discovery: DiscoveryService, connection: ConnectionService, observation: ObservationService
) -> None:
    connection.connect("fake", seed=2)
    play(connection, observation, 150)
    first = discovery.knowledge
    assert first is not None and len(first) > 0
    connection.disconnect()
    assert not discovery.enabled and discovery.stage == "idle"
    assert discovery.knowledge is first  # kept for review until the next game
    connection.connect("fake", seed=2)
    assert discovery.knowledge is not first and discovery.reviews == 0
    assert discovery.events.total == 0


def test_link_loss_during_an_experiment_is_survived(
    discovery: DiscoveryService, connection: ConnectionService, observation: ObservationService
) -> None:
    # 250 frames of play cost ~750 adapter operations; the experiment that
    # follows needs ~200 more, so the link drops in the middle of it.
    connection.connect("fake", seed=2, drop_after_ops=850)
    play(connection, observation, 250)
    assert not connection.is_connected
    assert not discovery.enabled


def test_bad_schedules() -> None:
    bus = EventBus()
    c = ConnectionService(default_registry(), bus)
    o = ObservationService(c, bus)
    with pytest.raises(ValueError):
        DiscoveryService(c, o, bus, review_every=0)
    with pytest.raises(ValueError):
        DiscoveryService(c, o, bus, experiment_every=0)
    DiscoveryService(c, o, bus, experiment_every=None)  # experiments off is allowed


# ------------------------------------------------------- persistence (T-308)


def test_knowledge_is_remembered_between_connections(
    connection: ConnectionService, observation: ObservationService, bus: EventBus, tmp_path: object
) -> None:
    from pathlib import Path

    from retro_ai_lab.storage import KnowledgeStore
    from retro_ai_lab.understanding import Confidence

    store = KnowledgeStore(Path(str(tmp_path)) / "knowledge")
    d = DiscoveryService(
        connection, observation, bus, review_every=100, experiment_every=None, store=store
    )
    assert d.store is store and d.save() is None  # nothing to save yet
    connection.connect("fake", seed=2)
    play(connection, observation, 100)
    k = d.knowledge
    assert k is not None and len(k) > 0 and d.saves >= 1
    first = next(iter(k))
    k.confirm(first.id)
    d.save()
    connection.disconnect()
    saved = store.load("toyracer-2")
    assert saved.get(first.id).confidence is Confidence.CONFIRMED
    connection.connect("fake", seed=2)
    again = d.knowledge
    assert again is not None and again is not k
    assert again.get(first.id).confidence is Confidence.CONFIRMED
    assert len(again) == len(k)
    connection.disconnect()
    connection.connect("fake", seed=9)  # a different game starts empty
    assert d.knowledge is not None and len(d.knowledge) == 0


def test_damaged_file_starts_over(
    connection: ConnectionService, observation: ObservationService, bus: EventBus, tmp_path: object
) -> None:
    from pathlib import Path

    from retro_ai_lab.storage import KnowledgeStore

    store = KnowledgeStore(Path(str(tmp_path)) / "knowledge")
    store.path_for("toyracer-2").write_text("{broken", "utf-8")
    d = DiscoveryService(connection, observation, bus, store=store)
    connection.connect("fake", seed=2)
    assert d.knowledge is not None and len(d.knowledge) == 0


def test_export_and_import(
    connection: ConnectionService, observation: ObservationService, bus: EventBus, tmp_path: object
) -> None:
    from pathlib import Path

    from retro_ai_lab.core import ConfigurationError
    from retro_ai_lab.storage.knowledge import load
    from retro_ai_lab.understanding import Knowledge, VariableKind

    base = Path(str(tmp_path))
    d = DiscoveryService(connection, observation, bus, review_every=100, experiment_every=None)
    with pytest.raises(ConfigurationError):
        d.export_to(base / "x.json")
    with pytest.raises(ConfigurationError):
        d.import_from(base / "x.json")
    connection.connect("fake", seed=2)
    play(connection, observation, 100)
    out = d.export_to(base / "shared.json")
    assert load(out).game_id == "toyracer-2"
    shared = Knowledge("toyracer-2", title="Toy Racer")
    shared.propose("ram", LAYOUT.score, kind=VariableKind.SCORE, name="Points")
    shared.confirm("v1")
    from retro_ai_lab.storage.knowledge import export

    export(shared, base / "friend.json")
    imported = d.import_from(base / "friend.json")
    assert d.knowledge is imported and d.knowledge.get("v1").name == "Points"
    play(connection, observation, 100)  # discovery carries on with the imported beliefs
    assert len(d.knowledge) > 1
    other = Knowledge("someone-else")
    export(other, base / "other.json")
    with pytest.raises(ConfigurationError, match="not"):
        d.import_from(base / "other.json")


def test_the_long_look_explains_checkpoints_and_laps(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    updates: list[DiscoveryUpdated],
) -> None:
    from retro_ai_lab.understanding.longlook import COUNTS_ROUNDS, SAME_PLACES

    connection.connect("fake", seed=2)
    play(connection, observation, 3000)
    k = discovery.knowledge
    assert k is not None
    checkpoint, lap = k.at("ram", LAYOUT.checkpoint), k.at("ram", LAYOUT.lap)
    assert checkpoint is not None and lap is not None
    assert checkpoint.kind is VariableKind.PROGRESS and lap.kind is VariableKind.PROGRESS
    assert any(e.kind == SAME_PLACES for e in checkpoint.evidence)
    assert any(e.kind == COUNTS_ROUNDS for e in lap.evidence)
    assert any(checkpoint in u.changed or lap in u.changed for u in updates)


def test_doubt_reset_adds_the_evidence_once_and_announces(
    discovery: DiscoveryService,
    connection: ConnectionService,
    bus: EventBus,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-845: values that cycle back to their start stop counting as a restart."""
    connection.connect("fake", seed=5)
    k = discovery.knowledge
    assert k is not None
    a = k.propose("ram", LAYOUT.score, kind=VariableKind.SCORE, name="Score")
    b = k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, name="Lives")
    seen: list[DiscoveryUpdated] = []
    bus.subscribe(DiscoveryUpdated, seen.append)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    changed = discovery.doubt_reset((a.id, b.id, "v999"))
    assert [v.id for v in changed] == [a.id, b.id]
    assert all(any(e.kind == CYCLES for e in k.get(v.id).evidence) for v in changed)
    assert len(seen) == 1 and seen[0].changed == changed
    assert "Score, Lives cycle back to their start" in caplog.text
    assert discovery.doubt_reset((a.id,)) == () and len(seen) == 1  # only once
    with pytest.raises(UnsupportedCapabilityError):
        connection.disconnect()
        discovery.doubt_reset((a.id,))


def test_self_moving_guesses_are_doubted_then_rejected(
    discovery: DiscoveryService,
    connection: ConnectionService,
    bus: EventBus,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-855: a score guess that changes while nobody presses anything is dropped."""
    connection.connect("fake", seed=5)
    k = discovery.knowledge
    assert k is not None
    real = k.propose("ram", LAYOUT.score, kind=VariableKind.SCORE, name="Score")
    junk = k.propose("ram", LAYOUT.timer, kind=VariableKind.SCORE, name="Timer as score")
    wide = k.propose(
        "ram", LAYOUT.frame, kind=VariableKind.PROGRESS, name="Frame as progress"
    )  # one byte; its second byte is what moves below
    kept = k.propose("ram", LAYOUT.noise, kind=VariableKind.SCORE, name="Noise as score")
    k.confirm(kept.id)
    speed = k.propose("ram", LAYOUT.speed, kind=VariableKind.SPEED, name="Speed")
    lives = k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, name="Lives")
    seen: list[DiscoveryUpdated] = []
    bus.subscribe(DiscoveryUpdated, seen.append)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    moved = {
        ("ram", LAYOUT.timer),
        ("ram", LAYOUT.frame),
        ("ram", LAYOUT.noise),
        ("ram", LAYOUT.speed),
        ("ram", LAYOUT.lives),  # lost by standing still: never doubted
    }
    changed = discovery.doubt_self_moving(moved, frame=7)
    assert {v.id for v in changed} == {junk.id, wide.id}
    assert all(not k.get(v.id).rejected for v in changed)  # a doubt, not yet a no
    assert [e.kind for e in k.get(junk.id).evidence][-1] == MOVES_ALONE
    assert len(seen) == 1 and set(seen[0].changed) == set(changed)
    assert "2 doubted (Timer as score, Frame as progress), 0 dropped (none)." in caplog.text
    changed = discovery.doubt_self_moving(moved)
    assert {v.id for v in changed} == {junk.id, wide.id}
    assert all(k.get(v.id).rejected for v in changed)
    assert "Rejected by the AI." in k.get(junk.id).evidence[-1].detail
    assert not k.get(real.id).rejected and k.get(kept.id).confirmed
    for other in (speed, lives):  # not doubted kinds
        assert not any(e.kind == MOVES_ALONE for e in k.get(other.id).evidence)
    assert "0 doubted (none), 2 dropped (Timer as score, Frame as progress)." in caplog.text
    assert discovery.doubt_self_moving(moved) == () and discovery.doubt_self_moving(()) == ()
    assert len(seen) == 2


def test_a_person_names_a_value_by_address(
    discovery: DiscoveryService,
    connection: ConnectionService,
    updates: list[DiscoveryUpdated],
) -> None:
    """T-862: naming an address confirms the AI's guess there, or makes a named value."""
    with pytest.raises(UnsupportedCapabilityError):
        discovery.name_value("ram", 0, "Lives")
    connection.connect("fake", seed=3)
    k = discovery.knowledge
    assert k is not None
    guess = k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES)
    updates.clear()
    named = discovery.name_value("ram", LAYOUT.lives, "  Hearts ")
    assert named.id == guess.id and named.name == "Hearts" and named.confirmed
    assert updates[-1].changed == (named,) and updates[-1].added == ()
    fresh = discovery.name_value("ram", LAYOUT.lane, "Lane")
    assert fresh.id != guess.id and fresh.name == "Lane" and fresh.confirmed
    assert fresh.kind is VariableKind.UNKNOWN and k.at("ram", LAYOUT.lane) == fresh
    assert updates[-1].added == (fresh,) and updates[-1].changed == ()
    with pytest.raises(ConfigurationError):
        discovery.name_value("ram", LAYOUT.lane, "   ")


def test_memory_from_several_threads_is_observed_one_at_a_time(
    connection: ConnectionService,
    observation: ObservationService,
    bus: EventBus,
) -> None:
    """T-847: snapshots come from the play clock's worker and the GUI thread at once."""
    import sys
    import threading

    discovery = DiscoveryService(connection, observation, bus, review_every=50)
    connection.connect("fake", seed=5)
    k = discovery.knowledge
    assert k is not None
    k.propose("ram", LAYOUT.score, kind=VariableKind.SCORE, name="Score")
    k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, name="Lives")
    play(connection, observation, 60)
    events = [e for e in (observation.observe() for _ in range(8)) if e is not None]
    assert len(events) == 8
    failures: list[BaseException] = []

    def publish(times: int) -> None:
        try:
            for i in range(times):
                bus.publish(events[i % len(events)])
        except BaseException as exc:
            failures.append(exc)

    before = discovery.reviews
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    threads = [threading.Thread(target=publish, args=(100,), daemon=True) for _ in range(4)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        sys.setswitchinterval(interval)
    assert not any(t.is_alive() for t in threads) and failures == []
    assert discovery.reviews == before + 400 // 50  # every snapshot counted, none lost
