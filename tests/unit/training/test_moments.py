"""T-1003: the moments saved during a go, and what the AI did since each."""

from __future__ import annotations

import random

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training.moments import EVERY, LEAD, Moment, Moments

SLOTS = ("back0", "back1", "back2")


def play(moments: Moments, moves: list[int]) -> None:
    for move in moves:
        moments.did(move)


def test_a_save_is_due_at_once_and_then_every_so_many_steps() -> None:
    moments = Moments(SLOTS, every=4, lead=6)
    assert moments.due() and moments.steps == 0 and len(moments) == 0
    # Not saving does not make it less due.
    play(moments, [1, 2])
    assert moments.due() and moments.steps == 2
    first = moments.saved("a mark")
    assert first == Moment(slot="back0", step=2, mark="a mark")
    assert moments.kept == (first,)
    for _ in range(4):
        assert not moments.due()
        moments.did(5)
    assert moments.due() and moments.steps == 6  # four steps after the save, not before
    moments.did(5)
    assert moments.due()  # one step late is still due
    second = moments.saved()
    assert second == Moment(slot="back1", step=7, mark=None)
    assert not moments.due()
    play(moments, [5, 5, 5, 5])
    assert moments.due()
    # The defaults are the prototype's: a save every 20 steps, back at least 30.
    assert (EVERY, LEAD) == (20, 30)
    plain = Moments(SLOTS)
    plain.saved()
    play(plain, [0] * 19)
    assert not plain.due()
    plain.did(0)
    assert plain.due()


def test_the_slots_can_be_read() -> None:
    moments = Moments(["back0", "back1"])
    assert moments.slots == ("back0", "back1")
    moments.saved()
    assert moments.slots == ("back0", "back1")  # the names, whether in use or not


def test_slots_go_round_and_the_oldest_moment_is_forgotten() -> None:
    moments = Moments(SLOTS, every=2, lead=3)
    made = []
    for number in range(5):
        assert moments.next_slot() == SLOTS[number % 3]
        made.append(moments.saved(number))
        play(moments, [10 + number, 20 + number])
    assert [moment.slot for moment in made] == ["back0", "back1", "back2", "back0", "back1"]
    assert [moment.step for moment in made] == [0, 2, 4, 6, 8]
    # Three slots: the three newest moments are kept, oldest first.
    assert moments.kept == tuple(made[2:])
    # What it did is kept from the oldest moment still kept, and no further back.
    found = moments.before_loss()
    assert found is not None
    assert found == (made[3], (13, 23, 14, 24))
    assert Moments(SLOTS, every=2, lead=6).before_loss() is None
    far = Moments(("only",), every=2, lead=1)
    far.saved()
    play(far, [1, 2, 3])
    assert far.next_slot() == "only"
    newer = far.saved()
    assert far.kept == (newer,)
    far.did(4)
    assert far.before_loss() == (newer, (4,))


def test_the_moment_before_a_loss_and_what_it_did_since() -> None:
    moments = Moments(SLOTS, every=4, lead=6)
    assert moments.before_loss() is None  # nothing is kept
    start = moments.saved("start")
    play(moments, [1, 1, 1, 1])
    later = moments.saved("later")
    play(moments, [2])
    # Five steps in: no moment lies six steps back yet.
    assert moments.before_loss() is None
    play(moments, [2])
    assert moments.before_loss() == (start, (1, 1, 1, 1, 2, 2))
    play(moments, [2, 2, 3])
    assert moments.before_loss() == (start, (1, 1, 1, 1, 2, 2, 2, 2, 3))
    # Ten steps in, the later moment is six steps back: the newest that is far enough.
    moments.did(3)
    assert moments.before_loss() == (later, (2, 2, 2, 2, 3, 3))
    # Asking changes nothing.
    assert moments.before_loss() == (later, (2, 2, 2, 2, 3, 3))
    assert moments.steps == 10 and moments.kept == (start, later)


def test_the_moment_before_it_came_to_a_spot() -> None:
    # T-1018: where to go back to for a standstill that began at some step.
    moments = Moments(SLOTS, every=4, lead=6)
    assert moments.since(0) is None  # nothing is kept
    start = moments.saved("start")
    play(moments, [1, 1, 1, 1])
    later = moments.saved("later")
    play(moments, [2, 2, 2])
    # The newest moment at or before the step, and all it did since that moment.
    assert moments.since(0) == (start, (1, 1, 1, 1, 2, 2, 2))
    assert moments.since(3) == (start, (1, 1, 1, 1, 2, 2, 2))
    assert moments.since(4) == (later, (2, 2, 2))
    assert moments.since(7) == moments.since(99) == (later, (2, 2, 2))
    # Asking changes nothing.
    assert moments.steps == 7 and moments.kept == (start, later)
    # The slots go round and the first moments are written over: every moment kept
    # is then later than the step, and the oldest of them is the one.
    play(moments, [3])
    third = moments.saved()
    play(moments, [3, 3, 3, 3])
    fourth = moments.saved()
    play(moments, [4])
    assert moments.kept == (later, third, fourth)
    assert moments.since(2) == (later, (2, 2, 2, 3, 3, 3, 3, 3, 4))
    assert moments.since(8) == (third, (3, 3, 3, 3, 4))
    # Gone back, the moments after are no longer there to be given.
    moments.went_back(third)
    assert moments.since(12) == (third, ())
    # Forgotten, there is none.
    moments.forget()
    assert moments.since(0) is None
    for bad in (-1, 1.5, True, None, "3"):
        with pytest.raises(ConfigurationError):
            moments.since(bad)  # type: ignore[arg-type]


def test_going_back_drops_what_came_after() -> None:
    moments = Moments(SLOTS, every=4, lead=2)
    first = moments.saved()
    play(moments, [1, 1, 1, 1])
    second = moments.saved()
    play(moments, [2, 2, 2, 2])
    third = moments.saved()
    play(moments, [3, 3])
    moments.went_back(second)
    assert moments.steps == 4
    assert moments.kept == (first, second)
    # The slot of the moment that was dropped is free again, and nothing is due yet.
    assert moments.next_slot() == third.slot
    assert not moments.due()
    assert moments.before_loss() == (first, (1, 1, 1, 1))
    # The story goes on from there, and the next save is due four steps after the moment.
    play(moments, [7, 7, 7])
    assert not moments.due()
    moments.did(7)
    assert moments.due()
    assert moments.before_loss() == (second, (7, 7, 7, 7))
    # Going back to where it already is changes nothing; a moment no longer kept is refused.
    moments.went_back(first)
    moments.went_back(first)
    assert moments.steps == 0 and moments.kept == (first,)
    for gone in (second, third, Moment(slot="back0", step=0, mark=None)):
        with pytest.raises(ConfigurationError):
            moments.went_back(gone)
    # The step it is at is already saved.
    with pytest.raises(ConfigurationError):
        moments.saved()


def test_a_hole_in_what_it_did_and_forgetting() -> None:
    moments = Moments(SLOTS, every=4, lead=2)
    moments.saved()
    play(moments, [1, 1, 1])
    assert moments.before_loss() is not None
    # A step that cannot be played again: nothing saved before it can be replayed to here.
    moments.did(None)
    assert moments.steps == 4 and len(moments) == 0
    assert moments.before_loss() is None
    assert moments.due() and moments.next_slot() == "back0"
    again = moments.saved()
    assert again.step == 4
    play(moments, [2, 2])
    assert moments.before_loss() == (again, (2, 2))
    moments.forget()
    assert moments.steps == 6 and moments.kept == ()
    assert moments.before_loss() is None and moments.due()
    # Steps played while nothing is kept are counted and not kept.
    play(moments, [3, 3])
    fresh = moments.saved()
    moments.did(4)
    moments.did(4)
    assert moments.before_loss() == (fresh, (4, 4))


def test_nonsense_is_refused() -> None:
    for slots in ((), ("a", "a"), ("a", ""), ("a", 3), "ab", ("with space",), None):
        with pytest.raises(ConfigurationError):
            Moments(slots)  # type: ignore[arg-type]
    for bad in (0, -1, 1.5, True, None):
        with pytest.raises(ConfigurationError):
            Moments(SLOTS, every=bad)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            Moments(SLOTS, lead=bad)  # type: ignore[arg-type]
    moments = Moments(SLOTS)
    for move in (-1, 1.0, True, "a"):
        with pytest.raises(ConfigurationError):
            moments.did(move)  # type: ignore[arg-type]
    assert moments.steps == 0
    moments.saved()
    with pytest.raises(ConfigurationError):
        moments.saved()
    with pytest.raises(ConfigurationError):
        moments.went_back("back0")  # type: ignore[arg-type]


def test_a_long_story_against_a_plain_record() -> None:
    """Random saves, steps, losses and goings back, checked against a list of everything done."""
    rng = random.Random(1003)
    slots = ("s0", "s1", "s2", "s3")
    every, lead = 5, 8
    moments = Moments(slots, every=every, lead=lead)
    story: list[int] = []  # every move of the story as it now stands
    saves: list[Moment] = []  # the moments that should still be kept, oldest first
    went_back = found = 0
    for _ in range(4000):
        assert moments.steps == len(story)
        assert moments.kept == tuple(saves)
        newest = saves[-1].step if saves else None
        assert moments.due() == (newest is None or len(story) - newest >= every)
        if moments.due() and rng.random() < 0.9:
            used = {moment.slot for moment in saves}
            free = [slot for slot in slots if slot not in used]
            want = free[0] if free else saves[0].slot
            assert moments.next_slot() == want
            if not free:
                saves.pop(0)
            saves.append(moments.saved(len(story)))
            assert saves[-1].slot == want and saves[-1].step == len(story)
        roll = rng.random()
        if roll < 0.9:
            move = rng.randrange(15)
            moments.did(move)
            story.append(move)
        elif roll < 0.97:
            old = [moment for moment in saves if len(story) - moment.step >= lead]
            got = moments.before_loss()
            if not old:
                assert got is None
                moments.forget()
                saves.clear()
            else:
                assert got is not None
                moment, since = got
                assert moment is old[-1]
                assert since == tuple(story[moment.step :])
                found += 1
                if rng.random() < 0.7:
                    moments.went_back(moment)
                    del story[moment.step :]
                    saves = [kept for kept in saves if kept.step <= moment.step]
                    went_back += 1
                else:
                    moments.forget()
                    saves.clear()
        elif roll < 0.99:
            moments.did(None)
            story.append(-1)
            saves.clear()
        else:
            moments.forget()
            saves.clear()
    assert went_back > 20 and found > went_back
