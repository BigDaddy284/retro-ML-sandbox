"""T-306: the discovery runner learns while the game is played."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import Capabilities, UnsupportedCapabilityError
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
from retro_ai_lab.services.discovery import game_id_for
from retro_ai_lab.understanding import EventKind, VariableKind

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
