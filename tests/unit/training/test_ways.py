"""T-999: the small changes to what it did that might get the AI past where it died."""

from __future__ import annotations

from collections import Counter

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.training.ways import HOLDS, STRIDE, Change, carry, changes

PAD = ActionSpace.from_buttons(("up", "down", "left", "right", "a", "b"))
RIGHT = PAD.index_of(["right"])
JUMP = PAD.index_of(["a", "right"])
DASH = PAD.index_of(["b", "right"])
LEFT = PAD.index_of(["left"])
STILL = PAD.index_of([])


def test_a_change_replaces_a_stretch_and_carries_on() -> None:
    trace = [RIGHT, RIGHT, LEFT, RIGHT, RIGHT, LEFT]
    change = Change(at=1, move=JUMP, hold=3)
    assert change.apply(trace) == (RIGHT, JUMP, JUMP, JUMP, RIGHT, LEFT)
    # It runs on past the end of what it did, doing what it mostly did.
    assert change.apply(trace, 4) == (
        RIGHT,
        JUMP,
        JUMP,
        JUMP,
        RIGHT,
        LEFT,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
    )
    # A hold that runs past the end is cut there.
    assert Change(at=4, move=JUMP, hold=16).apply(trace, 1) == (
        RIGHT,
        RIGHT,
        LEFT,
        RIGHT,
        JUMP,
        JUMP,
        RIGHT,
    )
    assert trace == [RIGHT, RIGHT, LEFT, RIGHT, RIGHT, LEFT]  # the trace is left alone
    # The commonest move; of two as common, the one made later.
    assert carry([LEFT, RIGHT, RIGHT, LEFT]) == LEFT
    assert carry([RIGHT, LEFT, LEFT, RIGHT]) == RIGHT
    assert carry([STILL]) == STILL
    for bad in (
        Change(at=6, move=JUMP, hold=1),  # past the end
        Change(at=-1, move=JUMP, hold=1),
        Change(at=0, move=JUMP, hold=0),
        Change(at=0, move=len(PAD), hold=1),
    ):
        with pytest.raises(ConfigurationError):
            bad.apply(trace, space=PAD)
    with pytest.raises(ConfigurationError):
        change.apply(trace, -1)
    with pytest.raises(ConfigurationError):
        change.apply([])
    with pytest.raises(ConfigurationError):
        carry([])


def test_changes_come_shortest_first_from_the_end_back() -> None:
    trace = [RIGHT] * 9
    found = list(changes(trace, PAD))
    assert (HOLDS, STRIDE) == ((4, 8, 16), 2)
    # Shortest hold first; within a hold, from the last step back, every other step.
    first = [c for c in found if c.hold == 4]
    assert found[: len(first)] == first
    assert [c.at for c in first] == [8, 8, 6, 6, 4, 4, 2, 2, 0, 0]
    assert [c.move for c in first[:2]] == list(PAD.near(RIGHT)) == [JUMP, DASH]
    assert [c.hold for c in found] == sorted(c.hold for c in found)
    # Another stride, other holds.
    wide = list(changes(trace, PAD, holds=(2,), stride=4))
    assert [(c.at, c.hold) for c in wide] == [(8, 2), (8, 2), (4, 2), (4, 2), (0, 2), (0, 2)]


def test_only_moves_near_the_one_made() -> None:
    trace = [RIGHT, STILL, JUMP, LEFT, STILL, RIGHT]
    for change in changes(trace, PAD, stride=1):
        assert 0 <= change.at < len(trace)
        assert change.move in PAD.near(trace[change.at])
    # Each place where it held something is tried with every near move.
    at_jump = {c.move for c in changes(trace, PAD, stride=1) if c.at == 2 and c.hold == 4}
    assert at_jump == set(PAD.near(JUMP)) == {RIGHT, PAD.index_of(["a"])}
    # Where it held nothing there is nothing to change by one button.
    assert not [c for c in changes(trace, PAD, stride=1) if c.at in (1, 4)]


def test_nothing_near_gives_nothing() -> None:
    assert list(changes([], PAD)) == []
    assert list(changes([STILL] * 30, PAD)) == []
    singles = ActionSpace.from_buttons(("left", "right", "a"), combos=False)
    assert list(changes([1, 2, 3, 1, 1], singles)) == []


def test_the_same_moves_are_not_tried_twice() -> None:
    trace = [RIGHT] * 9
    found = list(changes(trace, PAD))
    played = [c.apply(trace) for c in found]
    assert len(set(played)) == len(played)
    # At the last step a hold of 8 or 16 is the hold of 4 cut short: tried once.
    assert Counter(c.at for c in found)[8] == 2
    # Four steps from the end a hold of 8 is new (the hold of 4 stopped short of it)
    # and a hold of 16 is that again.
    assert sorted(c.hold for c in found if c.at == 4 and c.move == JUMP) == [4, 8]
    assert sorted(c.hold for c in found if c.at == 0 and c.move == JUMP) == [4, 8, 16]


def test_most_stops_the_list_and_nonsense_is_refused() -> None:
    trace = [RIGHT] * 40
    everything = list(changes(trace, PAD))
    assert list(changes(trace, PAD, most=7)) == everything[:7]
    assert list(changes(trace, PAD, most=10_000)) == everything
    for bad in (
        dict(most=0),
        dict(most=True),
        dict(stride=0),
        dict(stride=1.5),
        dict(holds=()),
        dict(holds=(4, 0)),
        dict(holds=(True,)),
    ):
        with pytest.raises(ConfigurationError):
            list(changes(trace, PAD, **bad))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        list(changes([RIGHT, len(PAD)], PAD))
    with pytest.raises(ConfigurationError):
        list(changes([RIGHT, -1], PAD))


def test_the_jump_before_the_gap_is_among_the_first_hundred() -> None:
    """What got the practice game's AI over a gap, time after time: walking right, and
    17 steps before the life went, right and A for 8 steps."""
    trace = [RIGHT] * 40
    found = list(changes(trace, PAD))
    wanted = Change(at=len(trace) - 17, move=JUMP, hold=8)
    assert found.index(wanted) < 100
    played = wanted.apply(trace, 20)
    assert len(played) == 60 and played[23:31] == (JUMP,) * 8
    assert set(played[:23]) == set(played[31:]) == {RIGHT}
