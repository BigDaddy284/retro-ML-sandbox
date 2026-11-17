"""T-1054: the learning bench on the practice games, whole runs."""

from __future__ import annotations

import pytest

from retro_ai_lab.tools.learning_bench import run


@pytest.mark.slow
def test_the_network_alone_on_the_practice_racer_is_whole_and_seeded() -> None:
    """Fifty goes of 300 steps for the network with every helper off: the run the
    bench makes for a first look at what the learner learns by itself."""
    report = run("racer", 50, seed=1)
    assert (report.algorithm, report.max_steps, report.block, report.helpers) == (
        "dqn",
        300,
        25,
        False,
    )
    assert len(report.played) == 50
    for go in report.played:
        assert 0 <= go.lost <= 3 and go.far >= 0 and 1 <= go.steps <= 300 and go.ended_by
    assert (report.ways_sought, report.remembered, report.looked_ahead) == (0, 0, 0)
    first, last = report.blocks()
    assert (first.first, first.last, last.first, last.last) == (1, 25, 26, 50)
    lines = report.lines()
    assert len(lines) == 4
    assert lines[0] == (
        "Learning bench on Toy Racer: neural network, 50 goes of up to 300 steps, seed 1, "
        "the learner alone, every helper off. How far is in the game's own units "
        "(a lap is 256)."
    )
    # It drives: it gets somewhere, in both halves.
    assert first.best > 0 and last.best > 0
    again = run("racer", 50, seed=1)
    assert again.played == report.played and again.lines() == lines


@pytest.mark.slow
def test_with_the_helpers_on_the_hopper_is_trained_as_the_app_trains_it() -> None:
    """The same goes with the helpers on and off: on, it goes back for a life lost
    or seen going, as the app does; off, it never does."""
    alone = run("hopper", 4, seed=3, algorithm="q_table", max_steps=200, block=2)
    helped = run("hopper", 4, seed=3, algorithm="q_table", max_steps=200, block=2, helpers=True)
    assert (alone.ways_sought, alone.remembered, alone.looked_ahead) == (0, 0, 0)
    assert helped.helpers and helped.looked_ahead > 0
    assert helped.ways_sought > 0
    assert len(alone.played) == len(helped.played) == 4
    assert "trained as the app does, with its helpers on." in helped.lines()[0]
    assert (
        helped.played
        == run(
            "hopper", 4, seed=3, algorithm="q_table", max_steps=200, block=2, helpers=True
        ).played
    )
