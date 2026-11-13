"""T-1009: going back for a way past a life lost, switched on in training."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from retro_ai_lab.core import Capabilities, ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.adapter import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT, ToyHopper
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import Settings
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    ObservationService,
    RewardService,
)
from retro_ai_lab.storage import RewardStore
from retro_ai_lab.training import TrainingService
from retro_ai_lab.training import service as module
from retro_ai_lab.training.episode import NOW_SLOT
from retro_ai_lab.training.records import Mode, Speed
from retro_ai_lab.training.service import BACK_SLOTS, WAY_SECONDS, WAYS_PER_GO
from retro_ai_lab.understanding import Evidence, VariableKind

SURE = [Evidence("test", "", 6.0)]
GAP_ENDS = 240  # the far edge of the first gap on this level
NO_SAVES = Capabilities(
    memory_read=True, screen_capture=True, input=True, frame_advance=True, speed_control=True
)


@dataclass
class Lab:
    bus: EventBus
    connection: ConnectionService
    observation: ObservationService
    discovery: DiscoveryService
    rewards: RewardService
    saved: list[str] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)

    def training(self, **kw: object) -> TrainingService:
        kw.setdefault("max_steps", 150)
        return TrainingService(
            self.connection,
            self.observation,
            self.discovery,
            self.bus,
            frames_per_step=4,
            settings=Settings(epsilon=0.5),
            seed=1,
            rewards=self.rewards,
            **kw,  # type: ignore[arg-type]
        )

    def connect(self, **kw: object) -> ToyHopper:
        """Connect the practice game, with its place and its lives known."""
        self.connection.connect("fake", game="hopper", seed=0, **kw)  # type: ignore[arg-type]
        adapter = self.connection.adapter
        assert isinstance(adapter, FakeEmulatorAdapter)
        game = adapter.game
        assert isinstance(game, ToyHopper)
        k = self.discovery.knowledge
        assert k is not None
        added = (
            k.propose("ram", LAYOUT.x_low, kind=VariableKind.POSITION, name="X", evidence=SURE),
            k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, name="Lives", evidence=SURE),
        )
        self.bus.publish(DiscoveryUpdated(added, (), (), ()))
        save, load, speed = adapter.save_state, adapter.load_state, adapter.set_speed
        adapter.save_state = lambda s: (self.saved.append(s), save(s))[1]  # type: ignore[method-assign]
        adapter.load_state = lambda s: (self.loaded.append(s), load(s))[1]  # type: ignore[method-assign]
        adapter.set_speed = lambda x: (self.speeds.append(x), speed(x))[1]  # type: ignore[method-assign]
        return game


@pytest.fixture
def lab(tmp_path: Path) -> Iterator[Lab]:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
    rewards = RewardService(connection, discovery, bus, store=RewardStore(tmp_path / "r"))
    made = Lab(bus, connection, observation, discovery, rewards)
    yield made
    connection.disconnect()


def walking_right(training: TrainingService, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start training with a brain that only ever walks right."""
    training.start(Mode.INDEPENDENT)
    agent = training.agent
    assert agent is not None
    walk = agent.space.index_of(["right"])
    monkeypatch.setattr(agent, "choose", lambda state: walk)


def until_it_looks(training: TrainingService, most: int = 80) -> int:
    """Step until the go has gone back to look for a way once. Returns the steps taken."""
    for taken in range(1, most + 1):
        assert training.step()
        go = training.episode
        if go is not None and go.ways_sought:
            return taken
    raise AssertionError(f"No search in {most} steps.")


def test_training_goes_back_and_gets_across(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    assert (WAYS_PER_GO, WAY_SECONDS, len(BACK_SLOTS)) == (3, 15.0, 6)
    assert NOW_SLOT not in BACK_SLOTS and module.CHECKPOINT_SLOT not in BACK_SLOTS
    t = lab.training()  # it listens for the game being connected
    game = lab.connect()
    walking_right(t, monkeypatch)
    until_it_looks(t)
    go = t.episode
    assert go is not None and (go.ways_sought, go.ways_found, go.times_back) == (1, 1, 1)
    assert go.moments is not None and go.moments.slots == BACK_SLOTS
    assert game.state.lives == toy_hopper.START_LIVES
    assert NOW_SLOT in lab.saved and set(BACK_SLOTS) & set(lab.saved)
    # The way is played, and it is beyond the gap with every life it started with.
    while game.state.x <= GAP_ENDS:
        assert t.step() and t.episode is go
        assert game.state.lives == toy_hopper.START_LIVES
    # The go's record says so.
    while t.episodes == 0:
        assert t.step()
    result = t.history[-1].result
    assert result.setbacks >= 1 and 1 <= result.ways_found <= result.ways_sought <= WAYS_PER_GO
    t.stop()


def test_a_search_runs_at_top_speed(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    t = lab.training()
    lab.connect()
    during: list[float] = []
    real = module.EpisodeRunner._seek

    def seeking(runner: object) -> bool:
        adapter = lab.connection.adapter
        assert isinstance(adapter, FakeEmulatorAdapter)
        before = len(lab.speeds)
        out = real(runner)  # type: ignore[arg-type]
        during.extend(lab.speeds[before:])
        return out

    monkeypatch.setattr(module.EpisodeRunner, "_seek", seeking)
    walking_right(t, monkeypatch)
    until_it_looks(t)
    adapter = lab.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    assert during == [module.FAST_SPEED, 1.0] and adapter.speed == 1.0
    t.stop()
    # Training that is already running fast stays fast: nothing to raise or put back.
    t.set_speed(Speed.FAST)
    during.clear()
    walking_right(t, monkeypatch)
    until_it_looks(t)
    assert during == [] and adapter.speed == module.FAST_SPEED
    t.stop()


def test_a_console_without_save_states_trains_as_before(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    t = lab.training()
    game = lab.connect(capabilities=NO_SAVES)
    with caplog.at_level(logging.INFO):
        walking_right(t, monkeypatch)
        while t.episodes == 0:
            assert t.step()
    go = t.history[-1].result
    assert go.setbacks >= 1 and (go.ways_sought, go.ways_found) == (0, 0)
    assert lab.saved == [] and lab.loaded == []
    assert game.state.lives < toy_hopper.START_LIVES
    said = [rec.getMessage() for rec in caplog.records]
    assert not any("goes back" in line or "went back" in line for line in said)
    t.stop()


def test_it_can_be_switched_off_and_nonsense_is_refused(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = lab.training(ways_per_go=0)
    lab.connect()
    walking_right(t, monkeypatch)
    while t.episodes == 0:
        assert t.step()
        go = t.episode
        assert go is None or go.moments is None
    result = t.history[-1].result
    assert result.setbacks >= 1 and result.ways_sought == 0
    assert not set(lab.saved) & {*BACK_SLOTS, NOW_SLOT}
    t.stop()
    for bad in (
        {"ways_per_go": -1},
        {"ways_per_go": 2.5},
        {"ways_per_go": True},
        {"way_seconds": 0},
        {"way_seconds": -3},
        {"way_seconds": "long"},
        {"way_seconds": False},
    ):
        with pytest.raises(ConfigurationError):
            lab.training(**bad)
    # No limit at all is allowed: the practice bench asks for that, to be repeatable.
    assert lab.training(way_seconds=None) is not None


def test_nothing_is_saved_while_a_person_plays(lab: Lab) -> None:
    t = lab.training()
    lab.connect()
    t.start(Mode.MANUAL)
    adapter = lab.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    right = [b for b in adapter.buttons() if b.name == "right"]
    lab.observation.record_input(["right"])
    adapter.set_input(right)
    for _ in range(200):  # into the gap and out the other side of losing a life
        adapter.advance_frames(1)
        lab.observation.advance(1)
    assert adapter.game.state.lives < toy_hopper.START_LIVES  # type: ignore[union-attr]
    assert t.episode is not None and t.episode.steps > 10  # it was learning from the play
    assert not set(lab.saved) & {*BACK_SLOTS, NOW_SLOT} and lab.loaded == []
    t.stop()


def test_the_log_says_it_will_and_what_came_of_it(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    t = lab.training()
    lab.connect()
    with caplog.at_level(logging.INFO):
        walking_right(t, monkeypatch)
        while t.episodes < 2:
            assert t.step()
    said = [rec.getMessage() for rec in caplog.records]
    will = [line for line in said if "goes back a little" in line]
    assert will == [
        "When the AI loses a life it goes back a little and tries small changes to find a "
        "way past, up to 3 times a go."
    ]  # once, not once a go
    first = t.history[0].result
    assert first.ways_sought >= 1
    ended = next(line for line in said if line.startswith("Go 1 ended"))
    assert first.describe_losses() in ended and first.describe_ways() in ended
    assert ended.index(first.describe_losses()) < ended.index(first.describe_ways())
    t.stop()
