"""T-806: saved AIs racing under one set of conditions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from retro_ai_lab.comparison import (
    Challenge,
    CompetitionFinished,
    Entry,
    SavedAiService,
    Scoreboard,
)
from retro_ai_lab.core import Capabilities, ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings
from retro_ai_lab.learning.greedy import GreedyPlayer
from retro_ai_lab.learning.state import State
from retro_ai_lab.storage import SavedAiStore
from retro_ai_lab.training import EpisodeResult, Status
from tests.unit.training.test_service import Lab


def _jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value))


def result(reward: float, steps: int = 10, objectives: int = 0) -> EpisodeResult:
    return EpisodeResult(steps, steps * 4, reward, "max_steps", 2, objectives)


def test_greedy_player_plays_the_best_move_and_learns_nothing() -> None:
    space = ActionSpace.from_buttons(BUTTONS)
    agent = QLearningAgent(space, Settings(epsilon=1.0, epsilon_min=0.0, alpha=1.0), seed=3)
    s1, s2 = State((("lane", 1),)), State((("lane", 2),))
    agent.learn(s1, 2, 9.0, s2)
    before = agent.to_dict()
    player = GreedyPlayer(agent)
    assert player.space == space and player.known == agent.known and player.epsilon == 0.0
    for _ in range(20):  # the agent itself would explore; the player never does
        assert player.choose(s1) == agent.greedy(s1) == 2
        assert player.last_decision is not None and not player.last_decision.explored
    assert player.learn(s1, 0, -50.0, s2) == 0.0
    assert player.end_episode() == 0.0 and player.episodes == 1
    assert agent.to_dict() == before and player.updates == 0
    assert player.policy() == agent.policy() and player.to_dict() == before
    assert "best known move" in player.stats()
    player.reset()
    assert player.last_decision is None and player.episodes == 0


def test_challenge_and_scoreboard_by_hand() -> None:
    c = Challenge(goes=2, max_steps=50, frames_per_step=4)
    assert c.describe() == (
        "2 goes each from the same starting point, up to 50 steps a go, "
        "with the rewards and goal set now."
    )
    assert "1 go each" in Challenge(goes=1).describe()
    for bad in ({"goes": 0}, {"goes": 51}, {"max_steps": 0}, {"frames_per_step": True}):
        with pytest.raises(ConfigurationError, match="whole number from 1 to"):
            Challenge(**bad)  # type: ignore[arg-type]
    a = Entry("AI A", (result(5.0, 10, 1), result(9.0, 20)))  # +7.0 a go
    b = Entry("AI B", (result(1.0, 5), result(11.0, 5, 2)))  # +6.0 a go, 2 goal steps
    assert a.goes == 2 and a.total_reward == 14.0 and a.average_reward == 7.0
    assert a.objectives == 1 and a.average_steps == 15.0 and a.successes == 1
    assert Entry("Empty", ()).average_reward == 0.0 and Entry("Empty", ()).average_steps == 0.0
    board = Scoreboard(c, [b, a])
    assert [e.name for e in board.ranked()] == ["AI A", "AI B"]  # more points a go
    assert board.winner() is a and board.challenge is c and board.entries == (b, a)
    assert board.describe() == (
        "AI A wins! 1. AI A: +7.0 points a go, 1 goal step, 15 steps a go. "
        "2. AI B: +6.0 points a go, 2 goal steps, 5 steps a go. "
        "(2 goes each from the same starting point, up to 50 steps a go, "
        "with the rewards and goal set now.)"
    )
    # same points: more goal steps win; same again: fewer steps win
    slow = Entry("AI C", (result(6.0, 30), result(6.0, 30)))
    quick = Entry("AI D", (result(6.0, 2, 2), result(6.0, 2)))
    assert [e.name for e in Scoreboard(c, [slow, b]).ranked()] == ["AI B", "AI C"]
    assert [e.name for e in Scoreboard(c, [b, quick]).ranked()] == ["AI D", "AI B"]
    tie = Entry("AI E", (result(5.0, 10, 1), result(9.0, 20)))
    even = Scoreboard(c, [a, tie])
    assert even.winner() is None and even.describe().startswith("It's a tie at the top!")
    assert Scoreboard(c, []).describe() == "Nobody raced." and Scoreboard(c, []).winner() is None
    with pytest.raises(ConfigurationError, match="own name"):
        Scoreboard(c, [a, a])


def test_saved_ais_race_from_the_same_start(lab: Lab, tmp_path: Path) -> None:
    ais = SavedAiService(
        lab.rewards, lab.goals, lab.training, lab.bus, store=SavedAiStore(tmp_path / "ais")
    )
    with pytest.raises(UnsupportedCapabilityError):
        ais.compete(["AI A", "AI B"])
    lab.connect()
    t = lab.training
    t.train(2)
    a = ais.save_as("AI A")
    t.forget()
    profile = lab.rewards.profile
    assert profile is not None
    for behaviour in profile.ids():
        lab.rewards.set_value(behaviour, -1.0)
    t.train(2)
    b = ais.save_as("AI B")
    finished: list[CompetitionFinished] = []
    lab.bus.subscribe(CompetitionFinished, finished.append)
    with pytest.raises(ConfigurationError, match="at least two saved AIs"):
        ais.compete(["AI A"])
    with pytest.raises(ConfigurationError, match="No saved AI called 'Nope'"):
        ais.compete(["AI A", "Nope"])
    episodes, brain = t.episodes, _jsonish(t.brain().to_dict())  # type: ignore[union-attr]
    start = lab.adapter().game.snapshot()
    challenge = Challenge(goes=2, max_steps=15, frames_per_step=3)
    board = ais.compete(["AI A", "AI B"], challenge)
    assert [e.name for e in board.entries] == ["AI A", "AI B"]
    assert all(e.goes == 2 for e in board.entries)
    assert all(r.steps <= 15 for e in board.entries for r in e.results)
    assert all(r.frames == r.steps * 3 for e in board.entries for r in e.results)
    assert finished == [CompetitionFinished(a.game_id, board)]
    assert board.describe().startswith(("AI A wins!", "AI B wins!", "It's a tie"))
    # the race changed nothing about the AI in training, the saved AIs, or the game's place
    assert t.status is Status.IDLE and t.episodes == episodes
    assert _jsonish(t.brain().to_dict()) == brain  # type: ignore[union-attr]
    assert ais.get("AI A").brain == _jsonish(a.brain) and ais.get("AI B").brain == _jsonish(b.brain)
    assert lab.adapter().game.snapshot() == start
    # training carries on afterwards as if nothing happened
    t.train(1)
    assert t.episodes == episodes + 1
    t.start()
    with pytest.raises(ConfigurationError, match="Stop training before a race"):
        ais.compete(["AI A", "AI B"])
    t.stop()
    # a console without save states can't race
    lab.connection.disconnect()
    lab.connect(
        capabilities=Capabilities(
            screen_capture=True, memory_read=True, input=True, frame_advance=True
        )
    )
    with pytest.raises(UnsupportedCapabilityError, match="no save states"):
        ais.compete(["AI A", "AI B"])
