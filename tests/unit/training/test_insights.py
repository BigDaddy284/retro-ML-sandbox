"""T-605: behaviour changes and suggested experiments."""

from __future__ import annotations

from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings, State
from retro_ai_lab.rewards import ExploitWarning, Goal, Objective, RewardProfile
from retro_ai_lab.services.discovery import Question
from retro_ai_lab.training import EpisodeRecord, EpisodeResult, LearningCurve, Mode
from retro_ai_lab.training.insights import (
    BehaviourChange,
    compare_policies,
    describe_changes,
    describe_experiments,
    policy_of,
    suggest_experiments,
)

SPACE = ActionSpace.from_buttons(BUTTONS)
A, B = State((("v1", 1),)), State((("v1", 2),))


def go(n: int, reward: float, objectives: int = 0) -> EpisodeRecord:
    return EpisodeRecord(
        n, Mode.GUIDED, EpisodeResult(10, 40, reward, "max_steps", 2, objectives), n * 40
    )


def curve(*rewards: float, objectives: int = 0, window: int = 3) -> LearningCurve:
    return LearningCurve([go(i + 1, r, objectives) for i, r in enumerate(rewards)], window)


def test_policies_compare_by_hand_and_read_well() -> None:
    agent = QLearningAgent(SPACE, Settings(epsilon=0.0, epsilon_min=0.0), seed=1)
    up, left = SPACE.index_of(["up"]), SPACE.index_of(["left"])
    agent.learn(A, up, 5.0, B, True)
    agent.learn(B, left, 5.0, A, True)
    before = policy_of(agent)
    assert before == {A.key: up, B.key: left}
    agent.learn(A, left, 50.0, B, True)  # A now prefers left; B unchanged
    after = policy_of(agent)
    changes = compare_policies(before, after)
    assert len(changes) == 1 and changes[0].state_key == A.key
    assert (changes[0].before, changes[0].after) == (up, left)
    assert compare_policies(before, before) == ()
    assert compare_policies({}, after) == ()  # unknown before: nothing to compare
    assert describe_changes(changes, 2, 4, None, SPACE) == (
        "Over the last 4 goes the AI changed its mind in 1 of 2 situations. "
        "Where v1=1 it used to press up; now it presses left."
    )
    wide = State(tuple((f"v{i}", i) for i in range(6)))  # a real game: dozens of values (T-889)
    text = describe_changes((BehaviourChange(wide.key, up, left),), 9, 3, None, SPACE)
    assert "Where v0=0, v1=1, v2=2, v3=3 and 2 more it used to press up" in text
    assert describe_changes((), 2, 1, None, SPACE) == (
        "Over the last 1 go the AI kept the same favourite move in every situation it knew."
    )
    assert describe_changes((), 0, 0, None, SPACE).startswith("Finish a few goes")


def test_suggestion_rules() -> None:
    explorer = QLearningAgent(SPACE, Settings(epsilon=0.9), seed=1)
    settled = QLearningAgent(SPACE, Settings(epsilon=0.1), seed=1)
    goal = Goal("g", "Win")
    goal.add(Objective("o1", "Reach the flag"))
    warning = ExploitWarning("farming", ("x",), 0.9, 100, "Farming.", "Lower the reward for x.")
    # Nothing at all: nothing to suggest.
    assert suggest_experiments(LearningCurve([]), None, None, None, None) == ()
    assert describe_experiments(()) == "Nothing to suggest yet."
    # A warning comes first; a still-exploring agent says keep going; too early says practise.
    titles = [e.title for e in suggest_experiments(curve(1, 2), explorer, warning, None, None)]
    assert titles == ["Lower a reward it is farming", "Keep going", "Practise fast"]
    # T-895: a confirmed guess that behaves unlike its kind is checked right after a warning.
    question = Question("v1", "Lifes", "busy", 'You confirmed "Lifes", but it changed a lot.')
    checks = suggest_experiments(curve(1, 2), explorer, warning, None, None, [question])
    assert [e.title for e in checks] == [
        "Lower a reward it is farming",
        'Check "Lifes"',
        "Keep going",
    ]
    assert checks[1].why == question.text
    # Never more than three, even when more rules fire.
    many = suggest_experiments(curve(1, 2, 3), explorer, warning, goal, None)
    assert len(many) == 3
    # No success after a window of goes: raise the reward for the step it's working on.
    none = [e.title for e in suggest_experiments(curve(1, 2, 3), settled, None, goal, None)]
    assert none == ["Practise fast", 'Raise the reward for "Reach the flag"']
    # Reaching goals often: a harder step.
    often = suggest_experiments(curve(1, 2, 3, objectives=1), settled, None, goal, None)
    assert [e.title for e in often] == ["Practise fast", "Add a harder step to the goal"]
    # Slipping: another personality (not the one already chosen).
    slipping = suggest_experiments(curve(5, 5, 5, 1, 1, 1), settled, None, None, None)
    assert slipping[0].title == "Try the explorer personality"
    profile = RewardProfile("g", "explorer")
    assert (
        suggest_experiments(curve(5, 5, 5, 1, 1, 1), settled, None, None, profile)[0].title
        == "Try the collector personality"
    )
    # Settled and steady with nothing else to say: nudge a reward.
    steady = suggest_experiments(curve(2, 2, 2, 2, 2, 2), settled, None, None, None)
    assert [e.title for e in steady] == ["Change one reward and watch"]
    text = describe_experiments(steady)
    assert text.startswith("Try next: Change one reward and watch (")
