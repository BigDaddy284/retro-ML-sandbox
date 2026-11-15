"""T-1031: changes are placed before the last chance only.

A game goes on playing out a loss for a while (the fall, the dying pause) before
its life counter drops, and nothing pressed in that while changes anything. The
places a change is tried at stop where that while begins; what the change is,
and for how long it is held, goes by the whole of what the AI did.
"""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.training.ways import HOLDS, STRIDE, Change, changes

PAD = ActionSpace.from_buttons(("up", "down", "left", "right", "a", "b"))
RIGHT = PAD.index_of(["right"])
JUMP = PAD.index_of(["a", "right"])
TRACE = [RIGHT] * 20


def test_changes_are_placed_before_the_step_given() -> None:
    assert (HOLDS, STRIDE) == ((4, 8, 16), 2)
    near = PAD.near(RIGHT)
    assert len(near) >= 2 and JUMP in near
    listed = list(changes(TRACE, PAD, upto=12))
    # Every place is before step 12: the nearest first, every second step back.
    assert {change.at for change in listed} == {11, 9, 7, 5, 3, 1}
    assert [change.at for change in listed[: 3 * len(near)]] == (
        [11] * len(near) + [9] * len(near) + [7] * len(near)
    )
    assert [change.move for change in listed[: len(near)]] == list(near)
    # The shortest hold at every place before the next hold anywhere.
    assert [change.hold for change in listed] == sorted(change.hold for change in listed)
    assert len(listed) == 6 * len(near) * len(HOLDS)
    # With the earliest first, the same places the other way round.
    early = list(changes(TRACE, PAD, upto=12, early=True))
    assert [change.at for change in early[: 3 * len(near)]] == (
        [1] * len(near) + [3] * len(near) + [5] * len(near)
    )
    assert {change.at for change in early} == {1, 3, 5, 7, 9, 11}
    # The same places as from the end backwards, whichever of the two the trace's own
    # last step would give: before step 11 they are the even ones, either way round.
    even = list(changes(TRACE, PAD, upto=11, early=True))
    assert [change.at for change in even[:: len(near)][:6]] == [0, 2, 4, 6, 8, 10]
    assert {change.at for change in even} == {c.at for c in changes(TRACE, PAD, upto=11)}
    # One step given: the first step is the only place.
    assert {change.at for change in changes(TRACE, PAD, upto=1)} == {0}
    assert {change.at for change in changes(TRACE, PAD, upto=2)} == {1}
    # most still stops the list.
    assert len(list(changes(TRACE, PAD, upto=12, most=5))) == 5


def test_a_hold_runs_on_past_it_and_is_told_apart() -> None:
    listed = list(changes(TRACE, PAD, upto=12))
    # At the last place a jump held 4, 8 and 16 steps are three changes: they differ
    # only past step 12, and there the game listens again once the change got past.
    last = [change for change in listed if (change.at, change.move) == (11, JUMP)]
    assert [change.hold for change in last] == [4, 8, 16]
    assert Change(11, JUMP, 4).apply(TRACE) == tuple([RIGHT] * 11 + [JUMP] * 4 + [RIGHT] * 5)
    assert Change(11, JUMP, 8).apply(TRACE) == tuple([RIGHT] * 11 + [JUMP] * 8 + [RIGHT])
    # Only the end of what it did cuts a hold, as ever: 9 steps are left from step 11.
    assert Change(11, JUMP, 16).apply(TRACE) == tuple([RIGHT] * 11 + [JUMP] * 9)
    # Played out whole, each is itself right to the end of the trace.
    whole = [c for c in changes(TRACE, PAD, upto=12, whole=True) if (c.at, c.move) == (11, JUMP)]
    assert [change.hold for change in whole] == [4, 8, 16]
    # Where the end does cut two holds to the same moves, one is listed (as before).
    short = list(changes([RIGHT] * 6, PAD, upto=6))
    assert [c.hold for c in short if (c.at, c.move) == (5, JUMP)] == [4]
    assert [c.hold for c in short if (c.at, c.move) == (1, JUMP)] == [4, 8]


def test_the_whole_list_without_it_and_nonsense_refused() -> None:
    everything = list(changes(TRACE, PAD))
    assert list(changes(TRACE, PAD, upto=None)) == everything
    assert list(changes(TRACE, PAD, upto=20)) == everything  # the end of what it did
    assert list(changes(TRACE, PAD, upto=99)) == everything  # or past it
    assert {change.at for change in everything} == set(range(1, 20, 2))
    assert list(changes(TRACE, PAD, upto=19)) != everything
    for bad in (0, -3, True, 2.5, "12"):
        with pytest.raises(ConfigurationError, match="upto must be a whole number, 1 or more"):
            list(changes(TRACE, PAD, upto=bad))  # type: ignore[arg-type]
