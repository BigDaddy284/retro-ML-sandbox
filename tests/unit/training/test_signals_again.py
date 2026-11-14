"""T-1028: what a go played straight would be paid as well is collected with each
step's reward, apart from it: it is nobody's pay."""

from __future__ import annotations

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.services.rewards import PaidAgain, RewardEarned
from retro_ai_lab.training.episode import EpisodeSignals, _with
from retro_ai_lab.understanding.events import EventKind, GameEvent

EVENT = GameEvent(EventKind.NEW_GROUND, 9, None, "")


def test_what_was_announced_is_kept_apart_from_the_reward() -> None:
    bus = EventBus()
    signals = EpisodeSignals(bus)
    assert signals.again == ()
    bus.publish(PaidAgain(9, 5.0, ("a",), (5.0,)))  # no go is running: nobody's
    signals.start()
    assert signals.take() == (0.0, 0) and signals.again == ()
    # One step: a payment, and three things only a go played straight would be paid.
    bus.publish(RewardEarned(9, 2.0, ("b",), EVENT, ("Beta",), (2.0,)))
    bus.publish(PaidAgain(9, 1.0, ("ground",), (1.0,)))
    bus.publish(PaidAgain(9, 1.0, ("ground",), (1.0,)))
    bus.publish(PaidAgain(9, 12.0, ("b", "a"), (2.0, 10.0)))
    bus.publish(PaidAgain(9, 4.0, ("c", "a")))  # no shares: split evenly
    assert signals.again == ()  # it is that of the reward last taken
    assert signals.take() == (2.0, 0)  # none of it is pay
    assert signals.parts == (("b", 2.0),)
    assert signals.again == (("a", 12.0), ("b", 2.0), ("c", 2.0), ("ground", 2.0))
    assert [e.id for e in signals.earned] == ["b"]  # nor is it in the go's summary
    assert signals.name_of("ground") == "ground"  # nor a name learnt from it
    # The next step carries nothing of that one.
    assert signals.take() == (0.0, 0) and signals.again == ()
    # A new go starts clean: of what was taken, and of what was not.
    bus.publish(PaidAgain(10, 1.0, ("ground",), (1.0,)))
    assert signals.take() == (0.0, 0) and signals.again == (("ground", 1.0),)
    bus.publish(PaidAgain(11, 1.0, ("ground",), (1.0,)))
    signals.start()
    assert signals.again == ()
    assert signals.take() == (0.0, 0) and signals.again == ()
    # Not listening: nothing is collected.
    signals.stop()
    bus.publish(PaidAgain(12, 1.0, ("ground",), (1.0,)))
    assert signals.take() == (0.0, 0) and signals.again == ()


def test_the_two_are_told_to_the_route_as_one() -> None:
    parts = (("a", 1.0), ("ground", 2.0))
    assert _with(parts, ()) is parts
    assert _with((), (("ground", 1.0),)) == (("ground", 1.0),)
    # A row in both is one figure; the ids stay in order.
    assert _with(parts, (("b", 4.0), ("ground", 1.0))) == (("a", 1.0), ("b", 4.0), ("ground", 3.0))
    assert parts == (("a", 1.0), ("ground", 2.0))
