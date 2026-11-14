"""T-1027: the reward of a step comes with what it was made of, and the rows that
have gone from the rewards are collected for whoever keeps accounts by them."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.services.goals import ObjectiveCompleted, ObjectiveSetback
from retro_ai_lab.services.rewards import RewardEarned, RewardProfileChanged
from retro_ai_lab.training.episode import EpisodeSignals
from retro_ai_lab.understanding.events import EventKind, GameEvent

EVENT = GameEvent(EventKind.PROGRESS, 9, "v1", "")
GOAL: Any = SimpleNamespace(id="g1", description="Reach the flag")


def test_a_steps_reward_comes_with_what_it_was_made_of() -> None:
    bus = EventBus()
    signals = EpisodeSignals(bus)
    assert signals.parts == ()
    bus.publish(RewardEarned(9, 5.0, ("a",), EVENT))  # no go is running: nobody's
    signals.start()
    assert signals.take() == (0.0, 0) and signals.parts == ()
    # One step: two payments, a goal step, and a setback, which is that step's too.
    bus.publish(RewardEarned(9, 12.0, ("a", "b"), EVENT, ("Alpha", "Beta"), (10.0, 2.0)))
    bus.publish(RewardEarned(9, 4.0, ("c", "a"), EVENT))  # no shares: split evenly
    bus.publish(ObjectiveCompleted(9, GOAL, 25.0))
    bus.publish(ObjectiveSetback(9, GOAL, -3.0))
    assert signals.parts == ()  # the parts are those of the reward last taken
    assert signals.take() == (38.0, 0)
    assert signals.parts == (("a", 12.0), ("b", 2.0), ("c", 2.0), ("step:g1", 22.0))
    assert sum(amount for _, amount in signals.parts) == 38.0  # every point has a name
    # The next step carries nothing of that one, and an empty step has no parts.
    bus.publish(RewardEarned(10, 1.5, ("b",), EVENT, ("Beta",), (1.5,)))
    assert signals.take() == (1.5, 0) and signals.parts == (("b", 1.5),)
    assert signals.take() == (0.0, 0) and signals.parts == ()
    # A new go starts clean: of what was taken, and of what was not.
    bus.publish(RewardEarned(11, 7.0, ("a",), EVENT))
    assert signals.take() == (7.0, 0) and signals.parts == (("a", 7.0),)
    bus.publish(RewardEarned(12, 2.0, ("b",), EVENT))
    signals.start()
    assert signals.parts == ()
    assert signals.take() == (0.0, 0) and signals.parts == ()
    # Not listening: nothing is added to the parts either.
    signals.stop()
    bus.publish(RewardEarned(13, 2.0, ("b",), EVENT))
    assert signals.take() == (0.0, 0) and signals.parts == ()
    # A row is known by the name it was last paid under, and by its id until then.
    assert signals.name_of("a") == "Alpha" and signals.name_of("b") == "Beta"
    assert signals.name_of("c") == "c" and signals.name_of("never paid") == "never paid"
    assert signals.name_of("step:g1") == "Reach the flag"


def test_rows_that_go_are_collected_whether_a_go_runs_or_not() -> None:
    bus = EventBus()
    signals = EpisodeSignals(bus)
    assert signals.take_gone() == frozenset()
    # Before any go: rows added and changed are nobody's business here.
    bus.publish(RewardProfileChanged(("new",), ("changed",), ("x", "y"), "suggestions"))
    signals.start()
    bus.publish(RewardProfileChanged((), (), ("z",), "control"))
    signals.stop()
    bus.publish(RewardProfileChanged((), (), ("x",), "control"))
    signals.start()  # a new go does not lose them
    assert signals.take_gone() == frozenset({"x", "y", "z"})
    assert signals.take_gone() == frozenset()
    bus.publish(RewardProfileChanged(("x",), (), (), "suggestions"))
    assert signals.take_gone() == frozenset()
