"""T-1053: a count that falls below nought has fallen, it has not risen.

Boss's game counts the lives in reserve: when the last one goes the count leaves 0
for 255. That read as lives gained, and paid as such.
"""

from __future__ import annotations

import pytest

from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.understanding import (
    Encoding,
    EventDetector,
    EventKind,
    Evidence,
    GameEvent,
    Knowledge,
    VariableKind,
)

RAM = "ram"
SURE = [Evidence("test", "", 3.0)]  # Medium
SETBACK, RECOVERY, UP, DOWN = (
    EventKind.SETBACK,
    EventKind.RECOVERY,
    EventKind.UP,
    EventKind.DOWN,
)


class Watched:
    """One value under the detector, fed the bytes it reads one snapshot at a time."""

    def __init__(
        self,
        kind: VariableKind = VariableKind.LIVES,
        layout: Encoding = Encoding.U8,
        length: int = 0,
        *,
        asked: bool = False,
    ) -> None:
        self.knowledge = Knowledge("g")
        self.value = self.knowledge.propose(
            RAM, 0, encoding=layout, kind=kind, name=kind.label, evidence=SURE, length=length
        )
        self.detector = EventDetector(self.knowledge)
        if asked:
            self.detector.ask([self.value.id])
        self.frame = 0

    def reads(self, *raw: int) -> tuple[GameEvent, ...]:
        self.frame += 1
        return self.detector.observe(MemorySnapshot(self.frame, {RAM: bytes(raw)}))

    def each(self, *values: int) -> list[list[EventKind]]:
        """The kinds of event each one-byte reading in turn gives."""
        return [[e.kind for e in self.reads(v)] for v in values]


# ------------------------------------------------------------ what is told


def test_the_last_life_going_is_a_life_lost_and_none_gained() -> None:
    lives = Watched()
    assert lives.each(2, 1, 0) == [[], [SETBACK], [SETBACK]]
    (told,) = lives.reads(255)
    assert told == GameEvent(
        SETBACK, 4, lives.value.id, "Lives fell below nought (it reads 255).", 0, 255
    )


@pytest.mark.parametrize(
    ("layout", "length", "two", "one", "none", "most", "reads"),
    [
        (Encoding.U16LE, 0, (2, 0), (1, 0), (0, 0), (255, 255), 65_535),
        (Encoding.U16BE, 0, (0, 2), (0, 1), (0, 0), (255, 255), 65_535),
        (Encoding.BCD, 0, (2,), (1,), (0,), (0x99,), 99),
        (Encoding.BCD_BE, 2, (0, 2), (0, 1), (0, 0), (0x99, 0x99), 9_999),
        (Encoding.DIGITS_BE, 2, (0, 2), (0, 1), (0, 0), (9, 9), 99),
        (Encoding.UINT_LE, 1, (2,), (1,), (0,), (255,), 255),
    ],
)
def test_on_every_layout_that_holds_a_most(
    layout: Encoding,
    length: int,
    two: tuple[int, ...],
    one: tuple[int, ...],
    none: tuple[int, ...],
    most: tuple[int, ...],
    reads: int,
) -> None:
    lives = Watched(layout=layout, length=length)
    lives.reads(*two)
    lives.reads(*one)
    lives.reads(*none)
    (told,) = lives.reads(*most)
    assert (told.kind, told.before, told.after) == (SETBACK, 0, reads)
    assert told.detail == f"Lives fell below nought (it reads {reads})."


def test_a_value_a_reward_is_on_went_down() -> None:
    # A person's own "goes down" row is paid on DOWN: the last life pays it too.
    count = Watched(VariableKind.UNKNOWN, asked=True)
    assert count.each(2, 1, 0) == [[], [DOWN], [DOWN]]
    (told,) = count.reads(255)
    assert told == GameEvent(
        DOWN, 4, count.value.id, "Unknown went down below nought (it reads 255).", 0, 255
    )


def test_a_lives_value_a_reward_is_on_tells_both_once() -> None:
    lives = Watched(asked=True)
    lives.each(2, 1, 0)
    down, setback = lives.reads(255)
    assert (down.kind, setback.kind) == (DOWN, SETBACK)
    assert down.detail == "Lives went down below nought (it reads 255)."
    assert setback.detail == "Lives fell below nought (it reads 255)."
    assert (down.before, down.after) == (setback.before, setback.after) == (0, 255)


@pytest.mark.parametrize("kind", [VariableKind.POSITION, VariableKind.COUNTER, VariableKind.SPEED])
def test_of_any_kind_but_health_where_a_reward_is_on_it(kind: VariableKind) -> None:
    assert Watched(kind, asked=True).each(2, 1, 0, 255)[-1] == [DOWN]


# ------------------------------------------------------------ what is left as it was


@pytest.mark.parametrize("to", [1, 3, 99, 254])
def test_a_rise_from_nought_to_less_than_the_most_is_a_rise(to: int) -> None:
    lives = Watched(asked=True)
    lives.each(2, 1, 0)
    up, recovery = lives.reads(to)
    assert (up.kind, recovery.kind) == (UP, RECOVERY)
    assert up.detail == f"Lives went up to {to}." and recovery.detail == f"Lives rose to {to}."


def test_only_from_nought() -> None:
    lives = Watched(asked=True)
    assert lives.each(3, 2, 1, 255)[-1] == [UP, RECOVERY]


def test_a_value_never_seen_to_fall_by_one_is_no_count() -> None:
    # A byte that switches between 0 and 255, taken for lives: it rose, as before.
    switch = Watched(asked=True)
    assert switch.each(255, 0, 255, 0, 255) == [
        [],
        [DOWN, SETBACK],
        [UP, RECOVERY],
        [DOWN, SETBACK],
        [UP, RECOVERY],
    ]
    # A fall of two is no fall by one, and neither is a rise by one.
    for readings in ((4, 2, 0, 255), (0, 1, 3, 0, 255), (254, 255, 0, 255)):
        assert Watched(asked=True).each(*readings)[-1] == [UP, RECOVERY]


def test_the_fall_by_one_may_be_long_ago_and_from_anywhere() -> None:
    lives = Watched()
    assert lives.each(9, 8, 20, 0, 255)[-1] == [SETBACK]


def test_health_is_filled_up() -> None:
    # From nothing to 99 on a packed decimal byte is a refill: health is left alone.
    health = Watched(VariableKind.HEALTH, Encoding.BCD, asked=True)
    assert health.each(2, 1, 0, 0x99)[-1] == [UP, RECOVERY]
    health = Watched(VariableKind.HEALTH)
    assert health.each(2, 1, 0, 255)[-1] == [RECOVERY]


def test_a_signed_byte_reads_below_nought_as_it_is() -> None:
    lives = Watched(layout=Encoding.S8, asked=True)
    lives.each(2, 1, 0)
    down, setback = lives.reads(255)  # -1
    assert (down.kind, setback.kind) == (DOWN, SETBACK)
    assert down.detail == "Lives went down to -1." and setback.detail == "Lives dropped to -1."
    assert lives.each(0, 127)[-1] == [UP, RECOVERY]  # the most it reads: no fall


def test_a_flag_switches() -> None:
    flag = Watched(VariableKind.FLAG, Encoding.FLAG, asked=True)
    assert flag.each(1, 0, 255)[-1] == [UP, EventKind.FLAG_ON]


def test_a_value_of_another_kind_that_no_reward_is_on_tells_nothing() -> None:
    assert Watched(VariableKind.UNKNOWN).each(2, 1, 0, 255) == [[], [], [], []]
    # Progress goes up as it did (only lives are read anew where no reward is on).
    progress = Watched(VariableKind.PROGRESS)
    assert progress.each(2, 1, 0, 255)[-1] == [EventKind.PROGRESS]


# ------------------------------------------------------------ what is kept


def test_the_fall_by_one_is_kept_by_where_the_value_is_held() -> None:
    # The guess is judged again between the fall and the last life going: the watch
    # on it is made anew, and what was seen of the place stands.
    lives = Watched()
    lives.each(2, 1, 0)
    lives.knowledge.support(lives.value.id, Evidence("more", "", 1.0))
    assert lives.each(0, 255) == [[], [SETBACK]]
    # A new guess at the same place, read the same way, takes it over.
    lives.knowledge.remove(lives.value.id)
    again = lives.knowledge.propose(RAM, 0, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
    assert lives.each(0, 255) == [[], [SETBACK]]
    assert lives.detector.watching == (again.id,)


def test_another_place_or_another_reading_of_it_has_not_fallen() -> None:
    lives = Watched()
    lives.each(2, 1, 0)
    k = lives.knowledge
    k.remove(lives.value.id)
    k.propose(RAM, 1, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
    packed = k.propose(
        RAM, 0, encoding=Encoding.BCD, kind=VariableKind.LIVES, name="Lives", evidence=SURE
    )
    lives.reads(0, 0)
    told = lives.reads(0x99, 255)
    assert [(e.kind, e.variable_id == packed.id) for e in told] == [
        (RECOVERY, False),
        (RECOVERY, True),
    ]


def test_it_is_kept_as_the_lowest_and_highest_seen_are() -> None:
    # reset() forgets the watches, not what was seen of the game (T-1022).
    lives = Watched()
    lives.each(2, 1, 0)
    lives.detector.reset()
    assert lives.each(0, 255) == [[], [SETBACK]]
