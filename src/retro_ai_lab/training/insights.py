"""What the AI now does differently, and what the child might try next.

DESIGN.md §10 asks the Learning panel for "AI behaviour changes", "identified
improvements" and "suggested experiments". The first two come from comparing
the AI's best known move in every situation now with what it was a few goes
ago; the third is a handful of plain rules over the learning curve, the
exploit detector and the goal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.learner import Learner
from retro_ai_lab.learning.state import State
from retro_ai_lab.rewards.exploitation import ExploitWarning
from retro_ai_lab.rewards.goals import Goal
from retro_ai_lab.rewards.model import RewardProfile
from retro_ai_lab.services.discovery import Question
from retro_ai_lab.training.curve import IMPROVING, SLIPPING, STEADY, TOO_EARLY, LearningCurve
from retro_ai_lab.understanding.knowledge import Knowledge

MAX_EXPERIMENTS = 3
MOSTLY_EXPLORING = 0.5
OFTEN = 0.5


@dataclass(frozen=True, slots=True)
class BehaviourChange:
    state_key: str
    before: int  # action index
    after: int


@dataclass(frozen=True, slots=True)
class Experiment:
    title: str
    why: str


def policy_of(learner: Learner) -> dict[str, int]:
    """The best known move for every situation the learner knows."""
    return learner.policy()


def compare_policies(
    before: Mapping[str, int], after: Mapping[str, int]
) -> tuple[BehaviourChange, ...]:
    """Situations known both times whose best move changed."""
    return tuple(
        BehaviourChange(key, before[key], after[key])
        for key in after
        if key in before and before[key] != after[key]
    )


SITUATION_VALUES = 4  # values named when saying where the AI changed its mind (T-889)


def describe_changes(
    changes: tuple[BehaviourChange, ...],
    known: int,
    goes_ago: int,
    knowledge: Knowledge | None,
    space: ActionSpace,
) -> str:
    if goes_ago <= 0:
        return "Finish a few goes and I'll tell you how the AI's habits change."
    goes = "go" if goes_ago == 1 else "goes"
    if not changes:
        return (
            f"Over the last {goes_ago} {goes} the AI kept the same favourite move "
            "in every situation it knew."
        )
    n = len(changes)
    text = (
        f"Over the last {goes_ago} {goes} the AI changed its mind in {n} of {known} "
        f"situation{'s' if known != 1 else ''}."
    )
    example = changes[0]
    where = State.from_key(example.state_key).describe(knowledge, most=SITUATION_VALUES)
    text += (
        f" Where {where} it used to press {space[example.before].label}; "
        f"now it presses {space[example.after].label}."
    )
    return text


def suggest_experiments(
    curve: LearningCurve,
    agent: Learner | None,
    warning: ExploitWarning | None,
    goal: Goal | None,
    profile: RewardProfile | None,
    questions: Sequence[Question] = (),
) -> tuple[Experiment, ...]:
    """Up to three things worth trying, most useful first."""
    out: list[Experiment] = []
    if warning is not None:
        out.append(Experiment("Lower a reward it is farming", warning.suggestion))
    for question in questions:  # a confirmed guess that behaves unlike its kind (T-895)
        out.append(Experiment(f'Check "{question.name}"', question.text))
    if agent is not None and agent.epsilon > MOSTLY_EXPLORING:
        out.append(
            Experiment(
                "Keep going",
                f"The AI still tries random moves {agent.epsilon:.0%} of the time; "
                "it settles down with more goes.",
            )
        )
    trend = curve.trend
    if curve.attempts and trend == TOO_EARLY:
        out.append(
            Experiment(
                "Practise fast",
                "A few more goes will show whether it is getting better; "
                "fast mode gets there sooner.",
            )
        )
    nxt = goal.next() if goal is not None else None
    if curve.attempts >= curve.window and curve.successes == 0 and nxt is not None:
        out.append(
            Experiment(
                f'Raise the reward for "{nxt.description}"',
                "The AI hasn't reached that step yet; a bigger prize makes it worth chasing.",
            )
        )
    if curve.attempts >= curve.window and curve.success_rate >= OFTEN and goal is not None:
        out.append(
            Experiment(
                "Add a harder step to the goal",
                f"It reaches a goal in {curve.success_rate:.0%} of goes; time for a new challenge.",
            )
        )
    if trend == SLIPPING:
        other = "explorer" if profile is None or profile.personality != "explorer" else "collector"
        out.append(
            Experiment(
                f"Try the {other} personality",
                "Recent goes scored less than before; different rewards push it a different way.",
            )
        )
    if trend in (IMPROVING, STEADY) and not out:
        out.append(
            Experiment(
                "Change one reward and watch",
                "The AI is settled; nudge a reward and see whether its habits follow.",
            )
        )
    return tuple(out[:MAX_EXPERIMENTS])


def describe_experiments(experiments: tuple[Experiment, ...]) -> str:
    if not experiments:
        return "Nothing to suggest yet."
    return "Try next: " + " ".join(f"{e.title} ({e.why})" for e in experiments)


__all__ = [
    "MAX_EXPERIMENTS",
    "BehaviourChange",
    "Experiment",
    "compare_policies",
    "describe_changes",
    "describe_experiments",
    "policy_of",
    "suggest_experiments",
]
