"""T-1053: the last life going is one life lost, however the detector tells it.

The detector now tells a count's fall below nought as a fall (DOWN, SETBACK). The
runner read it itself (T-1050) because it used to come as a rise, and must not count
the life twice.
"""

from __future__ import annotations

import pytest

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.services import RewardEarned
from retro_ai_lab.training.episode import EpisodeSignals
from retro_ai_lab.understanding.events import EventKind, GameEvent
from tests.unit.training import test_a_go_ends_when_the_game_is_over as older
from tests.unit.training.test_a_go_ends_when_the_game_is_over import (
    BELOW,
    HEALTH,
    LIFES,
    NAME,
    counts,
    taken,
    tell,
)

bus, signals = older.bus, older.signals  # (pytest fixtures)

UP, DOWN, SETBACK, RECOVERY = (
    EventKind.UP,
    EventKind.DOWN,
    EventKind.SETBACK,
    EventKind.RECOVERY,
)


def told(kind: EventKind, value: str = LIFES, before: int = 0, after: int = 255) -> GameEvent:
    return GameEvent(kind, 7, value, "", before, after)


@pytest.mark.parametrize(
    "kinds", [(DOWN,), (SETBACK,), (DOWN, SETBACK), (SETBACK, DOWN), (UP,), (UP, RECOVERY)]
)
def test_one_life_however_it_is_told(
    bus: EventBus, signals: EpisodeSignals, kinds: tuple[EventKind, ...]
) -> None:
    counts(bus, LIFES, 2, 1, 0)
    assert taken(signals) == (0, 2, False)
    tell(bus, *(told(kind) for kind in kinds))
    assert taken(signals) == (1, 1, True)
    assert signals.last_reset == f"{NAME} went from 0 to 255: {BELOW} (frame 7)"


@pytest.mark.parametrize("kinds", [(SETBACK,), (DOWN, SETBACK), (UP,), (UP, RECOVERY)])
def test_one_life_where_every_setback_counts(
    bus: EventBus, signals: EpisodeSignals, kinds: tuple[EventKind, ...]
) -> None:
    signals.watch_setbacks(None)  # no values named: each SETBACK is a life or health lost
    for before in (2, 1):
        tell(bus, told(SETBACK, LIFES, before, before - 1))
    assert taken(signals) == (0, 2, False)
    tell(bus, *(told(kind) for kind in kinds))
    assert taken(signals) == (1, 1, True)


def test_a_fall_nobody_counted_is_counted_by_the_reading(
    bus: EventBus, signals: EpisodeSignals
) -> None:
    # Lives lost are counted on the health alone; the game being over is still read
    # from the count, and its last life is a life (as when it was told as a rise).
    signals.watch_setbacks(frozenset({HEALTH}))
    for before in (2, 1):
        tell(bus, told(DOWN, LIFES, before, before - 1))
    assert taken(signals) == (0, 0, False)
    tell(bus, told(DOWN), told(SETBACK))
    assert taken(signals) == (1, 1, True)


def test_another_values_fall_is_not_the_counts(bus: EventBus, signals: EpisodeSignals) -> None:
    # The health falls in the update the last life goes in, told as a rise: two lost.
    counts(bus, LIFES, 2, 1, 0)
    signals.take_setbacks()
    tell(bus, told(DOWN, HEALTH, 5, 4), told(UP))
    assert taken(signals) == (1, 2, True)
    # And told as a fall: the same two.
    signals.start()
    counts(bus, LIFES, 2, 1, 0)
    signals.take_setbacks()
    tell(bus, told(SETBACK, HEALTH, 5, 4), told(SETBACK))
    assert taken(signals) == (1, 2, True)
    # And where every setback counts, with the count's own told as a rise: two again.
    signals.watch_setbacks(None)
    signals.start()
    for before in (2, 1):
        tell(bus, told(SETBACK, LIFES, before, before - 1))
    signals.take_setbacks()
    tell(bus, told(SETBACK, HEALTH, 5, 4), told(UP))
    assert taken(signals) == (1, 2, True)


def test_a_plain_fall_is_counted_as_ever(bus: EventBus, signals: EpisodeSignals) -> None:
    tell(bus, told(DOWN, LIFES, 3, 2), told(SETBACK, LIFES, 3, 2))
    assert taken(signals) == (0, 1, False)


# ------------------------------------------------------------ what was paid for


def paid(kind: EventKind, before: int, after: int, amount: float = -10.0) -> RewardEarned:
    return RewardEarned(7, amount, ("row",), GameEvent(kind, 7, LIFES, "", before, after))


@pytest.mark.parametrize("kind", [SETBACK, DOWN])
def test_a_penalty_for_the_fall_is_no_rise_paid_for(
    bus: EventBus, signals: EpisodeSignals, kind: EventKind
) -> None:
    # It reads higher than it was (0 to 255) and is a fall: nothing rose.
    bus.publish(paid(kind, 0, 255))
    assert signals.take_rises() == ()
    reward, _ = signals.take()
    assert reward == -10.0  # (and it is paid all the same)


@pytest.mark.parametrize("kind", [UP, RECOVERY, EventKind.PROGRESS, EventKind.SCORE_GAIN])
def test_a_rise_paid_for_is_noted_as_before(
    bus: EventBus, signals: EpisodeSignals, kind: EventKind
) -> None:
    bus.publish(paid(kind, 0, 255, 5.0))
    assert signals.take_rises() == (LIFES,)
    bus.publish(paid(kind, 3, 2, 5.0))  # (going down: no rise)
    assert signals.take_rises() == ()
