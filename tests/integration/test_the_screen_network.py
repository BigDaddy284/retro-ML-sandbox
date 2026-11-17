"""T-1055: the screen network trained by the app's trainer on the practice games,
whole runs through the learning bench."""

from __future__ import annotations

import pytest

from retro_ai_lab.tools.learning_bench import run

TITLES = {"racer": "Toy Racer", "hopper": "Toy Hopper"}
UNITS = {"racer": "(a lap is 256)", "hopper": "(a level is 2,560)"}


@pytest.mark.slow
@pytest.mark.parametrize("game", ["racer", "hopper"])
def test_the_screen_network_alone_on_a_practice_game_is_whole_and_seeded(game: str) -> None:
    """Eight goes of 100 steps with every helper off: the same build, the same
    learner and the same settings on a road seen from above and on a level seen
    from the side."""
    report = run(game, 8, seed=1, algorithm="screen", max_steps=100, block=4)
    assert (report.game, report.algorithm, report.max_steps, report.block, report.helpers) == (
        game,
        "screen",
        100,
        4,
        False,
    )
    assert len(report.played) == 8
    for go in report.played:
        assert 0 <= go.lost <= 3 and go.far >= 0 and 1 <= go.steps <= 100 and go.ended_by
    assert (report.ways_sought, report.remembered, report.looked_ahead) == (0, 0, 0)
    first, last = report.blocks()
    assert (first.first, first.last, last.first, last.last) == (1, 4, 5, 8)
    lines = report.lines()
    assert lines[0] == (
        f"Learning bench on {TITLES[game]}: screen network, 8 goes of up to 100 steps, "
        "seed 1, the learner alone, every helper off. How far is in the game's own units "
        f"{UNITS[game]}."
    )
    # It plays: the buttons it presses get it somewhere in both halves.
    assert first.best > 0 and last.best > 0
    # The same seed, the same goes, step for step; another seed, other goes.
    again = run(game, 8, seed=1, algorithm="screen", max_steps=100, block=4)
    assert again.played == report.played and again.lines() == lines
    other = run(game, 8, seed=2, algorithm="screen", max_steps=100, block=4)
    assert other.played != report.played


@pytest.mark.slow
def test_on_the_practice_racer_it_gets_further_than_it_began() -> None:
    """Fifty goes of 300 steps with every helper off, as the bench first measures a
    learner: in the second block of 25 it drives half as far again as in the first.
    (One seed, and a guard against it going blind, not the measure of how well it
    learns: the tracker holds that, over 300 goes and two seeds.)"""
    report = run("racer", 50, seed=2, algorithm="screen")
    first, last = report.blocks()
    assert (first.first, last.last) == (1, 50)
    assert last.far > 1.4 * first.far and last.far > 400
    assert last.best >= first.best > 0
