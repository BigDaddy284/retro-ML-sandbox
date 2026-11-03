"""T-404: the reward service keeps a live, saved profile per game."""

from __future__ import annotations

from pathlib import Path

import pytest

from retro_ai_lab.core import Capabilities, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.rewards import Category
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    ObservationService,
    RewardEarned,
    RewardProfileChanged,
    RewardService,
)
from retro_ai_lab.storage import RewardStore
from retro_ai_lab.understanding import (
    Encoding,
    EventKind,
    Evidence,
    GameEvent,
    Knowledge,
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
def store(tmp_path: Path) -> RewardStore:
    return RewardStore(tmp_path / "rewards")


@pytest.fixture
def rewards(
    connection: ConnectionService, discovery: DiscoveryService, bus: EventBus, store: RewardStore
) -> RewardService:
    return RewardService(connection, discovery, bus, store=store)


@pytest.fixture
def earned(bus: EventBus) -> list[RewardEarned]:
    seen: list[RewardEarned] = []
    bus.subscribe(RewardEarned, seen.append)
    return seen


@pytest.fixture
def changes(bus: EventBus) -> list[RewardProfileChanged]:
    seen: list[RewardProfileChanged] = []
    bus.subscribe(RewardProfileChanged, seen.append)
    return seen


def teach(discovery: DiscoveryService, bus: EventBus) -> Knowledge:
    """Give the knowledge the toy racer's variables, the way a review would."""
    k = discovery.knowledge
    assert k is not None
    added = (
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
    )
    bus.publish(DiscoveryUpdated(added, (), (), ()))
    return k


def test_no_profile_until_a_memory_capable_game_connects(
    rewards: RewardService, connection: ConnectionService
) -> None:
    assert not rewards.enabled and rewards.profile is None
    assert rewards.summary() == "No game to reward."
    with pytest.raises(UnsupportedCapabilityError):
        rewards.increase("reset:*")
    connection.connect("fake", capabilities=Capabilities(screen_capture=True, frame_advance=True))
    assert not rewards.enabled and rewards.profile is None
    connection.disconnect()
    connection.connect("fake", seed=3)
    assert rewards.enabled
    assert rewards.profile is not None and rewards.profile.game_id == "toyracer-3"
    assert rewards.profile.ids() == ("reset:*",)  # nothing known yet but "Starting over"
    assert rewards.summary() == "1 reward switched on; +0.0 earned."


def test_new_beliefs_add_behaviours_and_announce(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    changes: list[RewardProfileChanged],
) -> None:
    connection.connect("fake", seed=3)
    assert [c.reason for c in changes] == ["loaded", "suggestions"]
    teach(discovery, bus)
    assert changes[-1].reason == "suggestions" and len(changes[-1].added) == 4
    profile = rewards.profile
    assert profile is not None
    names = {b.name for b in profile}
    assert {"Losing Lives", "Gaining Lives", "Checkpoint goes up", "Score jumps up"} <= names
    assert rewards.refresh_suggestions().is_empty
    assert rewards.saves >= 2 and rewards.store is not None and rewards.store.exists("toyracer-3")


def test_events_earn_rewards_into_the_tally(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    earned: list[RewardEarned],
) -> None:
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    lives = k.at(RAM, LAYOUT.lives)
    checkpoint = k.at(RAM, LAYOUT.checkpoint)
    assert lives and checkpoint
    crash = GameEvent(EventKind.SETBACK, 50, lives.id, "", 3, 2)
    step = GameEvent(EventKind.PROGRESS, 60, checkpoint.id, "", 0, 1)
    nothing = GameEvent(EventKind.FLAG_ON, 61, "unknown", "")
    bus.publish(DiscoveryUpdated((), (), (crash, step, nothing), ()))
    assert [e.amount for e in earned] == [-10.0, 10.0]
    assert earned[0].behaviour_ids == (f"setback:{lives.id}",) and earned[0].event is crash
    assert rewards.tally.total == 0.0 and rewards.tally.count == 2
    assert rewards.tally.by_category(rewards.profile or k)[Category.SURVIVAL] == -10.0  # type: ignore[arg-type]
    # Switched off: earns nothing and is not announced.
    rewards.set_enabled(f"progress:{checkpoint.id}", False)
    bus.publish(DiscoveryUpdated((), (), (step,), ()))
    assert len(earned) == 2 and rewards.tally.count == 2
    assert "+" in rewards.summary() or "-" in rewards.summary()


def test_controls_save_and_are_remembered(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    changes: list[RewardProfileChanged],
    store: RewardStore,
) -> None:
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    lives = k.at(RAM, LAYOUT.lives)
    assert lives
    hit = f"setback:{lives.id}"
    assert rewards.increase(hit).value == -9.0
    assert rewards.decrease(hit).value == -10.0
    assert rewards.set_value(hit, -20.0).value == -20.0
    assert not rewards.set_enabled("reset:*", False).enabled
    assert rewards.choose_personality("survivor").id == "survivor"
    assert [c.reason for c in changes[-5:]] == ["control"] * 4 + ["personality"]
    rewards.reset_to_suggested()
    assert changes[-1].reason == "control" and rewards.profile is not None
    assert rewards.profile.get(hit).value == -20.0  # survivor: -10 x 2
    rewards.set_value(hit, -7.0)
    before = rewards.saves
    connection.disconnect()
    assert rewards.saves == before + 1 and not rewards.enabled
    assert rewards.profile is not None  # readable after disconnect
    connection.connect("fake", seed=3)
    profile = rewards.profile
    assert profile is not None and profile.personality == "survivor"
    assert profile.get(hit).value == -7.0 and profile.get(hit).custom
    assert not profile.get("reset:*").enabled
    assert rewards.tally.count == 0
    # Another game starts fresh.
    connection.disconnect()
    connection.connect("fake", seed=4)
    assert rewards.profile is not None and rewards.profile.game_id == "toyracer-4"
    assert rewards.profile.personality == "custom" and rewards.profile.ids() == ("reset:*",)
    assert set(store.games()) == {"toyracer-3", "toyracer-4"}


def test_damaged_file_starts_fresh(
    rewards: RewardService, connection: ConnectionService, store: RewardStore
) -> None:
    store.path_for("toyracer-3").write_text("{nope", "utf-8")
    connection.connect("fake", seed=3)
    assert rewards.profile is not None and rewards.profile.ids() == ("reset:*",)
    connection.disconnect()
    assert store.load("toyracer-3").ids() == ("reset:*",)


def test_without_a_store_nothing_is_saved(
    connection: ConnectionService, discovery: DiscoveryService, bus: EventBus
) -> None:
    rewards = RewardService(connection, discovery, bus)
    connection.connect("fake", seed=3)
    rewards.increase("reset:*")
    assert rewards.save() is None and rewards.saves == 0 and rewards.store is None
