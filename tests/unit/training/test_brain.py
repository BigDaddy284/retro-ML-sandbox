"""T-506: a brain round-trips with everything the AI has learned."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings, State
from retro_ai_lab.training import Brain, EpisodeRecord, EpisodeResult, Mode


def trained_agent() -> QLearningAgent:
    space = ActionSpace.from_buttons(BUTTONS)
    agent = QLearningAgent(space, Settings(epsilon=0.8), seed=4)
    a, b = State((("v1", 1), ("v2", 2))), State((("v1", 2), ("v2", 2)))
    for _ in range(5):
        agent.learn(a, agent.choose(a), 1.0, b, False)
        agent.learn(b, agent.choose(b), -1.0, a, True)
        agent.end_episode()
    return agent


def record(n: int) -> EpisodeRecord:
    return EpisodeRecord(n, Mode.GUIDED, EpisodeResult(10 * n, 40 * n, 2.5, "max_steps", 3), 40 * n)


def test_brain_round_trips_everything() -> None:
    agent = trained_agent()
    brain = Brain("toyracer-0", agent, (record(1), record(2)), 2, 20)
    data = brain.to_dict()
    assert data["schema"] == 1 and data["game_id"] == "toyracer-0" and len(data["goes"]) == 2
    again = Brain.from_dict(data, space=agent.space)
    assert again.game_id == "toyracer-0" and again.goes == brain.goes
    assert again.episodes == 2 and again.total_steps == 20
    assert again.agent.to_dict() == agent.to_dict()
    assert again.agent.epsilon == agent.epsilon and again.agent.updates == 10
    # The restored agent makes the very same next choice.
    s = State((("v1", 1), ("v2", 2)))
    assert again.agent.choose(s) == agent.choose(s)
    # Counts may exceed the goes kept (history is bounded); not the other way round.
    assert Brain("g", agent, (record(1),), 5, 50).episodes == 5
    with pytest.raises(ConfigurationError):
        Brain("g", agent, (record(1), record(2)), 1, 20)
    with pytest.raises(ConfigurationError):
        Brain("", agent)


def test_brain_refuses_bad_data() -> None:
    agent = trained_agent()
    data = Brain("g", agent).to_dict()
    with pytest.raises(ConfigurationError, match="newer version"):
        Brain.from_dict({**data, "schema": 99})
    with pytest.raises(ConfigurationError, match="Bad brain data"):
        Brain.from_dict({"schema": 1, "game_id": "g"})
    with pytest.raises(ConfigurationError, match="different controls"):
        Brain.from_dict(data, space=ActionSpace.from_buttons(BUTTONS[:3]))
    with pytest.raises(ConfigurationError, match="Bad go record"):
        Brain.from_dict({**data, "goes": [{"number": 1}]})
    with pytest.raises(ConfigurationError, match="Bad go result"):
        EpisodeResult.from_dict({"steps": "ten"})
    with pytest.raises(ConfigurationError, match="Bad go record"):
        EpisodeRecord.from_dict({**record(1).to_dict(), "mode": "psychic"})
    assert EpisodeRecord.from_dict(record(3).to_dict()) == record(3)
