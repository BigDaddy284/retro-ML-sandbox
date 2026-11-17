"""T-1055: a brain changed to or from the screen network starts afresh, and says so."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning import (
    AUTO,
    DQN,
    PICTURE_SIZE,
    Q_TABLE,
    SCREEN,
    SKILLS,
    ActionSpace,
    DQNLearner,
    Learner,
    QLearningAgent,
    ScreenLearner,
    Settings,
    Sight,
    SkillsLearner,
    State,
    carries_over,
    convert,
    make_learner,
)

SPACE = ActionSpace([frozenset(), frozenset({"left"}), frozenset({"right"})])
SETTINGS = Settings(epsilon=0.6, gamma=0.9, explore_steps=50)
OTHERS = (Q_TABLE, DQN, SKILLS)
SEEN = State((("x", 1),), Sight(bytes(PICTURE_SIZE), bytes(PICTURE_SIZE)))
NEXT = State((("x", 2),), Sight(bytes([9]) * PICTURE_SIZE, bytes(PICTURE_SIZE), 2))


def taught(algorithm: str) -> Learner:
    """A learner of that kind that has learned from thirty steps over two goes."""
    agent = make_learner(algorithm, SPACE, SETTINGS, seed=1)
    for i in range(30):
        agent.learn(SEEN, i % 3, 1.0, NEXT)
    agent.end_episode()
    agent.end_episode()
    assert agent.updates == 30 and agent.episodes == 2 and agent.epsilon < 0.6
    return agent


def numbers(agent: ScreenLearner) -> list[float]:
    return [float(v) for p in agent.network.parameters for v in p.ravel()[:5]]


def test_only_a_change_to_or_from_the_screen_network_loses_what_was_learned() -> None:
    for kind in (*OTHERS, SCREEN):
        agent = make_learner(kind, SPACE, seed=1)
        assert carries_over(agent, AUTO) and carries_over(agent, kind)
        for other in OTHERS:
            assert carries_over(agent, other) is (kind != SCREEN)
        assert carries_over(agent, SCREEN) is (kind == SCREEN)
        for unknown in ("magic", "", "Screen"):
            with pytest.raises(ConfigurationError, match="no learning algorithm called"):
                carries_over(agent, unknown)
            with pytest.raises(ConfigurationError, match="no learning algorithm called"):
                convert(agent, unknown)


@pytest.mark.parametrize("kind", OTHERS)
def test_changed_to_the_screen_network_it_knows_nothing_and_keeps_its_goes(kind: str) -> None:
    before = taught(kind)
    screen = convert(before, SCREEN, seed=7)
    assert isinstance(screen, ScreenLearner) and screen is not before
    assert screen.space == SPACE and screen.settings == SETTINGS
    assert screen.episodes == 2  # the goes played are still the goes played
    assert (screen.updates, screen.known, screen.trained, screen.unseen) == (0, 0, 0, 0)
    assert screen.epsilon == 0.6  # it explores as a learner that knows nothing does
    # What it was is untouched, and the new one is the one a seed gives.
    assert before.updates == 30 and before.algorithm == kind
    again = convert(taught(kind), SCREEN, seed=7)
    assert isinstance(again, ScreenLearner) and numbers(again) == numbers(screen)
    other = convert(taught(kind), SCREEN, seed=8)
    assert isinstance(other, ScreenLearner) and numbers(other) != numbers(screen)


@pytest.mark.parametrize(
    ("kind", "made"),
    [(Q_TABLE, QLearningAgent), (DQN, DQNLearner), (SKILLS, SkillsLearner)],
)
def test_changed_from_the_screen_network_it_is_a_new_learner_of_that_kind(
    kind: str, made: type
) -> None:
    screen = taught(SCREEN)
    assert isinstance(screen, ScreenLearner) and screen.known == 30
    fresh = convert(screen, kind, seed=7)
    assert isinstance(fresh, made) and fresh.algorithm == kind
    assert fresh.space == SPACE and fresh.settings == SETTINGS  # type: ignore[attr-defined]
    assert (fresh.updates, fresh.episodes, fresh.known) == (0, 0, 0)
    assert fresh.epsilon == 0.6
    assert fresh.values(SEEN) == (0.0, 0.0, 0.0) or kind == DQN  # a table knows nothing here
    assert screen.updates == 30 and screen.known == 30  # and the screen network is as it was


def test_the_screen_network_stays_itself_where_nothing_changes() -> None:
    screen = taught(SCREEN)
    assert convert(screen, SCREEN) is screen and convert(screen, AUTO) is screen
    table = taught(Q_TABLE)
    assert convert(table, AUTO) is table  # choosing for it never picks the screen network
