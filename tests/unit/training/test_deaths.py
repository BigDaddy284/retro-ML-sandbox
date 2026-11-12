"""What the AI saw before it died (T-968)."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.state import State
from retro_ai_lab.training.deaths import DEATHS_AT_LEAST, DeathWatch


def state(**parts: int) -> State:
    return State(tuple(parts.items()))


def test_the_sights_before_most_deaths_and_seldom_otherwise_are_named() -> None:
    """T-968: the user's deaths stayed at 1.2 a go while the points climbed; what did the
    AI see just before them, and is that a sight it has at any other step?"""
    assert DEATHS_AT_LEAST == 3
    w = DeathWatch(window=2)
    # Go 1: 100 steps, mostly with nothing near; the enemy (near=1) in 15 of them.
    for i in range(100):
        w.step(state(near=1 if i < 15 else 5, speed=0 if i < 40 else 2, score=3))
    w.death(state(near=1, speed=0, score=3))
    w.death(state(near=1, speed=2, score=3))
    assert w.describe() == ""  # two deaths say nothing yet
    w.end_go()
    for i in range(100):
        w.step(state(near=1 if i < 15 else 5, speed=0 if i < 40 else 2, score=3))
    w.death(state(near=1, speed=0, score=3))
    w.death(state(near=5, speed=0, score=3))
    assert w.deaths() == 4
    # near=1: 3 of 4 deaths (75%), 15% of steps: lift 0.60. speed=0: 3 of 4, 40%: 0.35.
    # score=3: every death but every step too: no lift.
    assert w.describe() == (
        "Before the 4 deaths in the last 2 goes the AI saw near=1 in 3 of them (15% of all "
        "steps) and speed=0 in 3 of them (40% of all steps)."
    )
    # The window moves: after two more goes the first go's deaths are gone.
    w.end_go()
    w.end_go()
    assert w.deaths() == 0 and w.describe() == ""


def test_nothing_standing_out_says_the_ai_may_not_see_what_kills_it() -> None:
    w = DeathWatch(window=10)
    for i in range(50):
        w.step(state(near=i % 2, speed=1))
    for near in (1, 0, 1):  # near=1 before 2 of 3 deaths, and at half of all steps: lift 0.17
        w.death(state(near=near, speed=1))
    assert w.describe() == (
        "Before the 3 deaths in the last 1 goes nothing the AI sees stood out: the same "
        "sights as any other step. It may not see what kills it."
    )
    with pytest.raises(ConfigurationError):
        DeathWatch(window=0)


def test_the_death_watch_names_its_standouts() -> None:
    """T-979: the sights the line names are handed to the encoder to look closer at."""
    w = DeathWatch(window=2)
    assert w.standouts() == ()
    for i in range(100):
        w.step(state(near=1 if i < 15 else 5, speed=0 if i < 40 else 2, score=3))
    w.death(state(near=1, speed=0, score=3))
    w.death(state(near=1, speed=2, score=3))
    assert w.standouts() == ()  # too few deaths, as the line says nothing
    w.death(state(near=1, speed=0, score=3))
    w.death(state(near=5, speed=0, score=3))
    assert w.standouts() == (("near", 1), ("speed", 0))  # the line's order: by lift
    assert "near=1 in 3 of them" in w.describe()
    w.end_go()
    for i in range(50):
        w.step(state(near=i % 2, speed=1))
    w.end_go()  # the first go's deaths are out of the window
    for near in (1, 0, 1):
        w.death(state(near=near, speed=1))
    assert "nothing the AI sees stood out" in w.describe() and w.standouts() == ()
