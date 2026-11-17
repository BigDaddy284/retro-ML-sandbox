"""T-1055: the app's trainer learns with the screen network, saves it and changes to it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake.adapter import FakeEmulatorAdapter
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import (
    AUTO,
    DQN,
    PICTURE_SIZE,
    Q_TABLE,
    SCREEN,
    SKILLS,
    DQNLearner,
    QLearningAgent,
    ScreenLearner,
    Settings,
    shrink,
)
from retro_ai_lab.services import ConnectionService, DiscoveryService, ObservationService
from retro_ai_lab.storage import BrainStore
from retro_ai_lab.training import TrainingService

AFRESH = "and starts learning afresh: what it had learned could not be taken along."


def service(tmp_path: Path, algorithm: str, **kw: object) -> TrainingService:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
    training = TrainingService(
        connection,
        observation,
        discovery,
        bus,
        frames_per_step=4,
        max_steps=40,
        settings=Settings(epsilon=0.5),
        seed=1,
        store=BrainStore(tmp_path / "b"),
        algorithm=algorithm,
        **kw,  # type: ignore[arg-type]
    )
    training.connection = connection  # type: ignore[attr-defined]
    return training


@pytest.mark.parametrize("game", ["racer", "hopper"])
def test_the_trainer_learns_with_the_screen_network_and_brings_it_back(
    tmp_path: Path, game: str
) -> None:
    training = service(tmp_path, SCREEN)
    connection: ConnectionService = training.connection  # type: ignore[attr-defined]
    connection.connect("fake", game=game, seed=3)
    agent = training.agent
    assert training.algorithm == SCREEN and isinstance(agent, ScreenLearner)
    training.train(3)
    assert agent.episodes == 3 and agent.updates > 60 and agent.unseen == 0
    assert agent.known == agent.updates  # every step is in its memory, with its pictures
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    # What it remembers last is the screen the game shows now.
    newest = agent._memory[-1]
    assert len(newest[3].now) == PICTURE_SIZE and newest[3].now == shrink(adapter.game.render())
    before = agent.to_dict()
    brain = training.brain()
    assert brain is not None and brain.algorithm == SCREEN
    connection.disconnect()
    # The saved brain is the network without its memory of steps.
    files = [p for p in (tmp_path / "b").rglob("*") if p.is_file()]
    assert files and sum(p.stat().st_size for p in files) < 4_000_000
    connection.connect("fake", game=game, seed=3)
    again = training.agent
    assert isinstance(again, ScreenLearner) and again is not agent
    assert again.to_dict() == before and again.known == 0
    assert json.dumps(again.to_dict()) == json.dumps(before)
    training.train(1)  # and it goes on learning
    assert again.episodes == 4 and again.updates > agent.updates
    connection.disconnect()


def test_changing_to_the_screen_network_and_back_starts_afresh_and_says_so(
    tmp_path: Path,
) -> None:
    training = service(tmp_path, Q_TABLE)
    connection: ConnectionService = training.connection  # type: ignore[attr-defined]
    for algorithm in (AUTO, Q_TABLE, DQN, SCREEN, SKILLS):
        assert training.starts_afresh(algorithm) is False  # no game, no brain to lose
    with pytest.raises(ConfigurationError, match="magic"):
        training.starts_afresh("magic")  # no way of learning, brain or no brain
    connection.connect("fake", seed=3)
    table = training.agent
    assert isinstance(table, QLearningAgent)
    assert training.starts_afresh(SCREEN) is False  # it has learned nothing yet
    training.train(2)
    assert table.updates > 0
    assert training.starts_afresh(SCREEN) is True
    for algorithm in (AUTO, Q_TABLE, DQN, SKILLS):
        assert training.starts_afresh(algorithm) is False
    with pytest.raises(ConfigurationError, match="magic"):
        training.starts_afresh("magic")
    # Between the table and the first network nothing is lost, as before.
    assert training.set_algorithm(DQN).endswith("keeping what it learned.")
    assert isinstance(training.agent, DQNLearner) and training.agent.updates == table.updates
    assert training.starts_afresh(SCREEN) is True
    said = training.set_algorithm(SCREEN)
    assert said == f"The AI moved from the neural network to the screen network, {AFRESH}"
    assert training.last_change == said and "keeping" not in said
    screen = training.agent
    assert isinstance(screen, ScreenLearner) and training.algorithm == SCREEN
    assert screen.updates == 0 and screen.known == 0 and screen.episodes == 2
    assert training.episodes == 2  # the goes played are still counted
    assert training.starts_afresh(Q_TABLE) is False  # the new one has nothing to lose yet
    assert training.set_algorithm(SCREEN) == ""  # already is
    training.train(1)
    assert screen.updates > 0 and screen.unseen == 0
    for algorithm in (Q_TABLE, DQN, SKILLS):
        assert training.starts_afresh(algorithm) is True
    assert training.starts_afresh(SCREEN) is False and training.starts_afresh(AUTO) is False
    assert training.set_algorithm(AUTO) == "" and training.agent is screen
    said = training.set_algorithm(Q_TABLE)
    assert said == f"The AI moved from the screen network to the simple table, {AFRESH}"
    fresh = training.agent
    assert isinstance(fresh, QLearningAgent) and fresh.updates == 0 and fresh.known == 0
    # The change is what is saved: reconnected, it is the new, empty table.
    connection.disconnect()
    connection.connect("fake", seed=3)
    assert isinstance(training.agent, QLearningAgent) and training.agent.updates == 0
    connection.disconnect()


def test_the_way_of_learning_cannot_be_changed_while_it_trains(tmp_path: Path) -> None:
    from retro_ai_lab.training import Mode

    training = service(tmp_path, Q_TABLE)
    connection: ConnectionService = training.connection  # type: ignore[attr-defined]
    connection.connect("fake", seed=3)
    training.start(Mode.INDEPENDENT)
    with pytest.raises(ConfigurationError, match="Stop training before changing"):
        training.set_algorithm(SCREEN)
    assert isinstance(training.agent, QLearningAgent)
    training.stop()
    connection.disconnect()
