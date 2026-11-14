"""T-1028: the runner knows which reward rows keep their count when the game is put
back to a saved moment, so that what they paid for the steps gone back over can be
owed to the go's story."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.services.goals import ObjectiveCompleted
from retro_ai_lab.services.rewards import (
    KEPT_KINDS,
    ONCE_A_GO_KINDS,
    PAST_BEST_KINDS,
    RewardEarned,
)
from retro_ai_lab.training.episode import EpisodeSignals
from retro_ai_lab.understanding.events import EventKind, GameEvent

GOAL: Any = SimpleNamespace(id="g1", description="Reach the flag")


def paid(bus: EventBus, kind: EventKind, *rows: str) -> None:
    bus.publish(RewardEarned(9, 2.0, rows, GameEvent(kind, 9, "v1", "")))


def test_the_rows_that_keep_their_count_are_known() -> None:
    # What pays only past the go's best, and new ground: the go's best and the ground
    # it has reached are not put back with the game.
    assert set(KEPT_KINDS) == {EventKind.PROGRESS, EventKind.SCORE_GAIN, EventKind.NEW_GROUND}
    assert KEPT_KINDS >= PAST_BEST_KINDS
    assert not KEPT_KINDS & ONCE_A_GO_KINDS  # a switch never pays again in the go

    bus = EventBus()
    signals = EpisodeSignals(bus)
    assert signals.kept == frozenset()
    paid(bus, EventKind.PROGRESS, "early")  # no go is running: not heard
    assert signals.kept == frozenset()
    signals.start()
    paid(bus, EventKind.PROGRESS, "progress:v1")
    paid(bus, EventKind.SCORE_GAIN, "score:v2", "both:v2")  # every row of a payment
    paid(bus, EventKind.NEW_GROUND, "screen:new_ground")
    assert signals.kept == {"progress:v1", "score:v2", "both:v2", "screen:new_ground"}
    # What happens again is paid again, or never again in the go: not these.
    for kind in EventKind:
        if kind not in KEPT_KINDS:
            paid(bus, kind, f"row of {kind.value}")
    bus.publish(ObjectiveCompleted(9, GOAL, 25.0))  # a goal's step is reached for good
    assert signals.kept == {"progress:v1", "score:v2", "both:v2", "screen:new_ground"}
    # What a row is does not change from one go to the next.
    signals.take()
    signals.start()
    assert signals.kept == {"progress:v1", "score:v2", "both:v2", "screen:new_ground"}
    # It is a copy: nobody changes it from outside.
    assert isinstance(signals.kept, frozenset)
