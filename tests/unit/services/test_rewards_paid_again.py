"""T-1028: what a go played straight would be paid is announced after a going back.

The rewards that pay only the first time in a go (ground, a progress number or the
score past the go's best, a switch flipped) keep their counts when the game is put
back to a saved moment, so the go is not paid twice. The same counts are kept a
second time for the story of the go, which goes back with the game: what they
would pay and the go's own do not is announced as ``PaidAgain``, and nobody is paid.
"""

from __future__ import annotations

import pytest

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.rewards import GROUND_ID
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    GoStarted,
    GroundCovered,
    MomentSaved,
    PaidAgain,
    RewardEarned,
    RewardService,
    WentBack,
)
from retro_ai_lab.understanding import EventKind, GameEvent, VariableKind
from tests.unit.services import test_rewards as base
from tests.unit.services.test_rewards import RAM, SURE, teach

bus = base.bus  # the real services on the practice racer (pytest fixtures)
connection = base.connection
discovery = base.discovery
store = base.store
rewards = base.rewards
earned = base.earned


@pytest.fixture
def again(bus: EventBus) -> list[PaidAgain]:
    seen: list[PaidAgain] = []
    bus.subscribe(PaidAgain, seen.append)
    return seen


def test_ground_covered_again_is_announced_and_not_paid(
    rewards: RewardService,
    connection: ConnectionService,
    bus: EventBus,
    earned: list[RewardEarned],
    again: list[PaidAgain],
) -> None:
    bus.publish(GroundCovered(1, 0, 2))  # nothing connected: ignored
    connection.connect("fake", seed=3)
    bus.publish(GroundCovered(10, 2))
    assert len(earned) == 2 and again == []
    tallied = rewards.tally.count
    # Steps the go was paid for before it was put back: told, a step at a time, with
    # the row that would pay and what it would give. Nobody is paid and nothing tallied.
    bus.publish(GroundCovered(20, 0, 3))
    assert again == [PaidAgain(20, 1.0, (GROUND_ID,), (1.0,))] * 3
    assert len(earned) == 2 and rewards.tally.count == tallied
    # One look that is part old ground, part new.
    bus.publish(GroundCovered(30, 1, 2))
    assert [(e.frame, e.amount) for e in earned[2:]] == [(30, 1.0)]
    assert [(a.frame, a.amount) for a in again[3:]] == [(30, 1.0)] * 2
    assert rewards.tally.count == tallied + 1
    # It is the person's row: at their value, and nothing when it is worth nothing,
    # switched off or the rewards are.
    rewards.set_value(GROUND_ID, 2.5)
    bus.publish(GroundCovered(40, 0, 1))
    assert again[5:] == [PaidAgain(40, 2.5, (GROUND_ID,), (2.5,))]
    rewards.set_value(GROUND_ID, 0.0)
    bus.publish(GroundCovered(50, 0, 1))
    rewards.set_value(GROUND_ID, 1.0)
    rewards.set_enabled(GROUND_ID, False)
    bus.publish(GroundCovered(60, 0, 1))
    assert len(again) == 6 and len(earned) == 3
    rewards.set_enabled(GROUND_ID, True)
    connection.disconnect()
    bus.publish(GroundCovered(70, 0, 1))
    assert len(again) == 6


def test_what_the_story_would_be_paid_after_going_back(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    earned: list[RewardEarned],
    again: list[PaidAgain],
) -> None:
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    checkpoint = k.at(RAM, LAYOUT.checkpoint)
    score = k.at(RAM, LAYOUT.score)
    assert checkpoint and score
    door = k.propose(RAM, LAYOUT.lap, kind=VariableKind.FLAG, name="Door", evidence=SURE)
    lamp = k.propose(RAM, LAYOUT.lane, kind=VariableKind.FLAG, name="Lamp", evidence=SURE)
    bus.publish(DiscoveryUpdated((door, lamp), (), (), ()))

    def progress(frame: int, value: int) -> None:
        step = GameEvent(EventKind.PROGRESS, frame, checkpoint.id, "", 0, value)
        bus.publish(DiscoveryUpdated((), (), (step,), ()))

    def points(frame: int, value: int) -> None:
        gain = GameEvent(EventKind.SCORE_GAIN, frame, score.id, "", 0, value)
        bus.publish(DiscoveryUpdated((), (), (gain,), ()))

    def switch(frame: int, var_id: str) -> None:
        flip = GameEvent(EventKind.FLAG_ON, frame, var_id, "", 0, 1)
        bus.publish(DiscoveryUpdated((), (), (flip,), ()))

    def paid() -> list[int]:
        out = [e.frame for e in earned]
        earned.clear()
        return out

    def told() -> list[tuple[int, float, tuple[str, ...], tuple[float, ...]]]:
        out = [(a.frame, a.amount, a.behaviour_ids, a.shares) for a in again]
        again.clear()
        return out

    def go() -> None:
        """A go to the second checkpoint, a moment saved there, and on to the fourth."""
        bus.publish(GoStarted(0))
        progress(10, 1)
        switch(12, lamp.id)
        progress(20, 2)
        points(22, 100)
        bus.publish(MomentSaved(20))
        progress(30, 3)
        points(32, 300)
        switch(34, door.id)
        progress(40, 4)
        assert paid() == [10, 12, 20, 22, 30, 32, 34, 40] and told() == []

    profile = rewards.profile
    assert profile is not None
    cp_row = profile.matching(GameEvent(EventKind.PROGRESS, 0, checkpoint.id, "", 0, 1))
    sc_row = profile.matching(GameEvent(EventKind.SCORE_GAIN, 0, score.id, "", 0, 1))
    assert len(cp_row) == 1 and len(sc_row) == 1
    cp, cp_worth = cp_row[0].id, float(cp_row[0].value)
    sc, sc_worth = sc_row[0].id, float(sc_row[0].value)
    on = f"flag_on:{door.id}"

    go()
    # Put back to the moment: the go has been to the fourth checkpoint, its story to
    # the second. Up to the fourth again nothing is paid, and each is told once.
    bus.publish(WentBack(20))
    assert paid() == [] and told() == []
    progress(50, 2)  # where the story already was
    assert told() == []
    progress(52, 3)
    assert told() == [(52, cp_worth, (cp,), (cp_worth,))]
    progress(54, 2)
    progress(56, 3)  # a step back and forward again is not progress for the story either
    assert told() == []
    points(58, 200)  # the score likewise, by its own best
    assert told() == [(58, sc_worth, (sc,), (sc_worth,))]
    points(60, 150)
    assert told() == []
    switch(62, door.id)  # first flipped after the moment: the story's first time again
    switch(64, door.id)
    assert told() == [(62, 2.0, (on,), (2.0,))]
    switch(66, lamp.id)  # flipped before the moment: the story has had it
    assert told() == []
    progress(68, 4)
    assert told() == [(68, cp_worth, (cp,), (cp_worth,))]
    assert paid() == []
    # Past the go's best it is paid as ever, and nothing is told.
    progress(70, 5)
    points(72, 400)
    assert paid() == [70, 72] and told() == []

    # Going back to a moment nobody named changes nothing.
    bus.publish(WentBack(999))
    progress(80, 3)
    points(82, 100)
    switch(84, door.id)
    assert paid() == [] and told() == []

    # Moments saved after the one gone back to are no longer the story's: going
    # "back" to one of them later changes nothing.
    go()
    bus.publish(MomentSaved(40))  # at the fourth checkpoint
    bus.publish(WentBack(20))
    bus.publish(WentBack(40))
    progress(90, 3)
    assert told() == [(90, cp_worth, (cp,), (cp_worth,))]
    # A moment saved now is named by its step like any other, and holds what the
    # story has by now: the third checkpoint, not the second.
    bus.publish(MomentSaved(30))
    progress(92, 4)
    bus.publish(WentBack(30))
    progress(94, 3)
    assert told() == [(92, cp_worth, (cp,), (cp_worth,))]
    progress(96, 4)
    assert told() == [(96, cp_worth, (cp,), (cp_worth,))]
    # The first moment still stands, and going back to it twice tells it twice.
    bus.publish(WentBack(20))
    progress(98, 3)
    bus.publish(WentBack(20))
    progress(99, 3)
    assert [frame for frame, *_ in told()] == [98, 99]
    assert paid() == []

    # A new go starts with nothing kept: no moment of the last one, and no count.
    bus.publish(GoStarted(100))
    bus.publish(WentBack(20))
    progress(110, 1)
    assert paid() == [110] and told() == []
    go()
    bus.publish(WentBack(20))
    progress(120, 3)
    assert told() == [(120, cp_worth, (cp,), (cp_worth,))]
    # So does a game connected afresh.
    connection.disconnect()
    connection.connect("fake", seed=3)
    teach(discovery, bus)
    paid()
    progress(130, 5)
    bus.publish(WentBack(20))
    progress(132, 4)
    assert paid() == [130] and told() == []
    bus.publish(MomentSaved(7))
    progress(134, 6)
    bus.publish(WentBack(7))
    progress(136, 6)
    assert paid() == [134] and [frame for frame, *_ in told()] == [136]


def test_without_going_back_nothing_is_announced(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    earned: list[RewardEarned],
    again: list[PaidAgain],
) -> None:
    """Moments are saved all through a go; only going back to one makes a story that
    is not the go."""
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    checkpoint = k.at(RAM, LAYOUT.checkpoint)
    score = k.at(RAM, LAYOUT.score)
    assert checkpoint and score
    door = k.propose(RAM, LAYOUT.lap, kind=VariableKind.FLAG, name="Door", evidence=SURE)
    bus.publish(DiscoveryUpdated((door,), (), (), ()))
    bus.publish(GoStarted(0))
    frames: list[int] = []
    for step, value in enumerate((1, 2, 1, 2, 3, 3, 2, 4)):
        frame = 10 * step + 10
        events = (
            GameEvent(EventKind.PROGRESS, frame, checkpoint.id, "", 0, value),
            GameEvent(EventKind.SCORE_GAIN, frame + 1, score.id, "", 0, 100 * value),
            GameEvent(EventKind.FLAG_ON, frame + 2, door.id, "", 0, 1),
            GameEvent(EventKind.PROGRESS, frame + 3, checkpoint.id, ""),  # no value: paid
        )
        bus.publish(DiscoveryUpdated((), (), events, ()))
        bus.publish(GroundCovered(frame + 4, 1))
        bus.publish(MomentSaved(step))
        frames.append(frame)
    assert again == []
    # What is paid is what was paid before there was a second count: past the best
    # (the 1st, 2nd, 5th and 8th), the switch once, and what has no value every time.
    by_frame = sorted(e.frame for e in earned)
    best = [10, 20, 50, 80]
    expected = (
        best
        + [frame + 1 for frame in best]
        + [12]
        + [frame + 3 for frame in frames]
        + [frame + 4 for frame in frames]
    )
    assert by_frame == sorted(expected)
