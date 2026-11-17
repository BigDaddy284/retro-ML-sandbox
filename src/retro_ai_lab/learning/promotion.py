"""Moving a brain between the simple table and the neural network (DESIGN.md §6.1).

The app starts every game with the table and, when the game turns out to be
big, promotes the brain to a network that first learns everything the table
knew, so the child never sees the AI forget. The child can also move it
either way by hand.

The screen network (T-1055) is the exception. What a table or the first network
knows is told in numbers from the game's memory; the screen network knows only
pictures, and neither can be put into the other's words. A brain changed to or
from it starts learning afresh, and :func:`carries_over` says so beforehand.
"""

from __future__ import annotations

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.dqn import DEFAULT_DISTIL_STEPS, DQN, DQNConfig, DQNLearner
from retro_ai_lab.learning.learner import (
    AUTO,
    Q_TABLE,
    Learner,
    algorithm_choices,
    make_learner,
)
from retro_ai_lab.learning.qlearning import QLearningAgent
from retro_ai_lab.learning.screen import SCREEN
from retro_ai_lab.learning.skills import DEFAULT_SKILL, SKILLS, SkillsLearner


def promote(
    agent: QLearningAgent,
    *,
    seed: int | None = None,
    config: DQNConfig | None = None,
    steps: int = DEFAULT_DISTIL_STEPS,
) -> tuple[DQNLearner, float]:
    """A network that knows what the table knew; also how often their best moves agree."""
    dqn = DQNLearner(agent.space, agent.settings, seed=seed, config=config)
    agreement = dqn.distil(agent.table, steps=steps)
    dqn.resume_from(agent)
    return dqn, agreement


def demote(agent: DQNLearner, *, seed: int | None = None) -> QLearningAgent:
    """A table holding the network's scores for every situation it knows."""
    table = QLearningAgent(agent.space, agent.settings, seed=seed)
    for state in agent.known_states():
        table.table.row(state)[:] = agent.values(state)
    table.resume_from(agent)
    return table


def _check(algorithm: str) -> None:
    if algorithm not in algorithm_choices():
        raise ConfigurationError(
            f"There is no learning algorithm called {algorithm!r} "
            f"(choices: {', '.join(algorithm_choices())})."
        )


def carries_over(agent: Learner, algorithm: str) -> bool:
    """Whether ``agent`` changed to ``algorithm`` keeps what it learned.

    False only between the screen network and any other way of learning: pictures
    and numbers from the game's memory cannot be told in each other's words.
    """
    _check(algorithm)
    if algorithm == AUTO or agent.algorithm == algorithm:
        return True
    return SCREEN not in (algorithm, agent.algorithm)


def convert(
    agent: Learner,
    algorithm: str,
    *,
    seed: int | None = None,
    config: DQNConfig | None = None,
    steps: int = DEFAULT_DISTIL_STEPS,
) -> Learner:
    """The agent as ``algorithm``; the same object when it already is (or for ``auto``).

    To or from the screen network it is a new learner that knows nothing
    (:func:`carries_over`); the screen network carries on the count of goes.
    """
    if not carries_over(agent, algorithm):  # which refuses a name that is no way of learning
        fresh = make_learner(algorithm, agent.space, getattr(agent, "settings", None), seed=seed)
        if algorithm == SCREEN:
            fresh.resume_from(agent)  # type: ignore[attr-defined]
        return fresh
    if algorithm == AUTO or agent.algorithm == algorithm:
        return agent
    if algorithm == SKILLS:
        skills = SkillsLearner(agent.space, getattr(agent, "settings", None), seed=seed)
        skills.adopt(DEFAULT_SKILL, agent)
        skills.resume_from(agent)
        return skills
    if isinstance(agent, SkillsLearner):
        general = agent.skill_learner(DEFAULT_SKILL)  # the other skills are dropped
        return convert(general, algorithm, seed=seed, config=config, steps=steps)
    if algorithm == DQN and isinstance(agent, QLearningAgent):
        return promote(agent, seed=seed, config=config, steps=steps)[0]
    if algorithm == Q_TABLE and isinstance(agent, DQNLearner):
        return demote(agent, seed=seed)
    raise ConfigurationError(f"Can't turn a {agent.algorithm!r} brain into a {algorithm!r} one.")


__all__ = ["carries_over", "convert", "demote", "promote"]
