"""T-803: saved AIs side by side."""

from __future__ import annotations

import json
from typing import Any

import pytest

from retro_ai_lab.comparison import (
    MEASURES,
    AiComparison,
    BehaviourDifference,
    SavedAi,
    SavedAiService,
    Scorecard,
    behaviour_differences,
    describe,
)
from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings
from retro_ai_lab.learning.state import State
from retro_ai_lab.rewards.goals import Goal
from retro_ai_lab.rewards.model import RewardProfile
from retro_ai_lab.storage import SavedAiStore
from retro_ai_lab.training import Brain, EpisodeRecord, EpisodeResult, Mode
from tests.unit.training.test_service import Lab

GAME = "toyracer-0"
SPACE = ActionSpace.from_buttons(BUTTONS)


def go(number: int, reward: float, steps: int = 10, objectives: int = 0) -> EpisodeRecord:
    result = EpisodeResult(steps, steps * 4, reward, "max_steps", 3, objectives)
    return EpisodeRecord(number, Mode.GUIDED, result, number * 100)


def saved(
    name: str,
    goes: tuple[EpisodeRecord, ...] = (),
    agent: QLearningAgent | None = None,
    personality: str = "custom",
) -> SavedAi:
    agent = agent or QLearningAgent(SPACE, seed=1)
    brain = Brain(
        GAME, agent, goes, episodes=len(goes), total_steps=sum(g.result.steps for g in goes)
    )
    data: dict[str, Any] = {
        "rewards": RewardProfile(GAME, personality=personality).to_dict(),
        "goal": Goal(GAME).to_dict(),
        "brain": brain.to_dict(),
    }
    return SavedAi(name, GAME, **json.loads(json.dumps(data)))


def test_scorecard_reads_the_goes() -> None:
    a = saved(
        "AI A", (go(1, 5.0, 10, 1), go(2, -1.0, 20), go(3, 8.0, 12, 2)), personality="explorer"
    )
    card = Scorecard.from_saved(a)
    assert card.name == "AI A" and card.personality == "explorer" and card.algorithm == "q_table"
    assert card.attempts == 3
    assert card.average_reward == pytest.approx(4.0) and card.best_reward == 8.0
    assert (
        card.objectives == 3 and card.successes == 2 and card.success_rate == pytest.approx(2 / 3)
    )
    assert card.average_steps == pytest.approx(14.0) and card.average_frames == pytest.approx(56.0)
    assert card.situations == 0
    empty = Scorecard.from_saved(saved("AI B"))
    assert empty.attempts == 0 and empty.average_reward is None and empty.best_reward is None
    assert empty.average_steps is None and empty.successes == 0 and empty.success_rate == 0.0
    assert card.value("average_reward") == 4.0 and empty.value("average_reward") is None
    with pytest.raises(ConfigurationError, match="no measure called 'luck'"):
        card.value("luck")
    with pytest.raises(ConfigurationError, match="damaged goes"):
        Scorecard.from_saved(SavedAi("C", GAME, {}, {}, {"goes": "x"}))


def test_leaders_follow_the_direction_of_each_measure() -> None:
    a = saved("AI A", (go(1, 5.0, 10, 1), go(2, 7.0, 10, 1)))
    b = saved("AI B", (go(1, 1.0, 5), go(2, 9.0, 5)))
    c = AiComparison.of([a, b])
    assert [card.name for card in c.cards] == ["AI A", "AI B"]
    assert {m.key for m in c.measures()} == {m.key for m in MEASURES}
    leaders = {k: (v.name if v else None) for k, v in c.leaders().items()}
    assert leaders == {
        "average_reward": "AI A",
        "best_reward": "AI B",
        "objectives": "AI A",
        "successes": "AI A",
        "average_steps": "AI B",  # fewer steps is better
        "attempts": None,  # a tie
    }
    assert AiComparison.of([a]).leader("average_reward") is None
    assert AiComparison.of([]).leader("average_reward") is None
    assert AiComparison.of([a, saved("AI C")]).leader("average_reward") is None  # one has no goes
    with pytest.raises(ConfigurationError, match="own name"):
        AiComparison.of([a, saved("AI A")])


def _trained(seed: int, prefer: int) -> QLearningAgent:
    agent = QLearningAgent(SPACE, Settings(epsilon=0.0, epsilon_min=0.0, alpha=1.0), seed=seed)
    s1, s2, s3 = State((("lane", 1),)), State((("lane", 2),)), State((("lane", 3),))
    agent.learn(s1, prefer, 5.0, s2)  # both know lane 1, differently
    agent.learn(s2, 0, 5.0, s3)  # both know lane 2 and agree
    agent.learn(s3 if seed == 1 else State((("lane", 9),)), 1, 5.0, s1)  # one each knows alone
    return agent


def test_behaviour_differences_cover_shared_situations_only() -> None:
    a = saved("AI A", agent=_trained(1, 1))
    b = saved("AI B", agent=_trained(2, 2))
    assert Scorecard.from_saved(a).situations == 3
    diffs = behaviour_differences(a, b)
    assert diffs == (BehaviourDifference("lane=1", 1, 2),)
    assert behaviour_differences(a, a) == ()
    other = QLearningAgent(ActionSpace.from_buttons(BUTTONS[:2]), seed=1)
    with pytest.raises(ConfigurationError, match="different controls"):
        behaviour_differences(a, saved("AI C", agent=other))
    with pytest.raises(ConfigurationError, match="no brain to compare"):
        behaviour_differences(a, SavedAi("AI D", GAME, {}, {}, {}))


def test_describe_in_a_childs_words() -> None:
    a = saved("AI A", (go(1, 5.0, 10, 1), go(2, 7.0, 10, 1)), agent=_trained(1, 1))
    b = saved("AI B", (go(1, 1.0, 5), go(2, 9.0, 5)), agent=_trained(2, 2))
    text = describe(AiComparison.of([a, b]), differences=behaviour_differences(a, b), space=SPACE)
    assert text == (
        "AI A scored more points a go (+6.0 against AI B's +5.0). "
        "AI B had the best single go (+9.0 against AI A's +7.0). "
        "AI A completed more goal steps (2 against AI B's 0). "
        "AI A reached a goal in more goes (2 of 2 against AI B's 0 of 2). "
        "AI B finished its goes in fewer steps (5 against AI A's 10). "
        "In 1 situation they would press different buttons: "
        f"where lane=1 AI A presses {SPACE[1].label} and AI B presses {SPACE[2].label}."
    )
    assert describe(AiComparison.of([])) == "Save two AIs for this game and I'll compare them."
    assert describe(AiComparison.of([a])) == (
        "Only AI A is saved so far. Save another AI to compare them."
    )
    assert describe(AiComparison.of([saved("AI A"), saved("AI B")])) == (
        "AI A and AI B haven't had any goes yet, so there is nothing to compare."
    )
    twin = saved("AI B", (go(1, 5.0, 10, 1), go(2, 7.0, 10, 1)), agent=_trained(1, 1))
    assert describe(AiComparison.of([a, twin]), differences=behaviour_differences(a, twin)) == (
        "AI A and AI B did about as well as each other. "
        "In every situation they both know, they would press the same buttons."
    )
    rested = describe(AiComparison.of([a, saved("AI B"), saved("AI C")]))
    assert rested.startswith("AI B and AI C have not had a go yet. ")
    assert "AI A did about as well as each other" not in rested
    # without a space the moves are still named somehow
    assert "presses move 1" in describe(
        AiComparison.of([a, b]), differences=[BehaviourDifference("lane=1", 1, 2)]
    )


def test_two_ais_saved_from_the_practice_console_compare(lab: Lab, tmp_path: Any) -> None:
    ais = SavedAiService(
        lab.rewards, lab.goals, lab.training, lab.bus, store=SavedAiStore(tmp_path / "ais")
    )
    lab.connect()
    t = lab.training
    t.train(3)
    a = ais.save_as("AI A")
    t.forget()
    profile = lab.rewards.profile
    assert profile is not None
    for behaviour in profile.ids():
        lab.rewards.set_value(behaviour, -1.0)
    t.train(3)
    b = ais.save_as("AI B")
    comparison = AiComparison.of([a, b])
    assert [c.attempts for c in comparison.cards] == [3, 3]
    assert comparison.cards[0].situations > 0
    text = describe(
        comparison,
        differences=behaviour_differences(a, b),
        space=t.space,
        knowledge=lab.discovery.knowledge,
    )
    assert "AI A" in text and "AI B's" in text and "against" in text
