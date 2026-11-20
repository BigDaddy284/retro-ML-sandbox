"""T-1052: the practice bench with looking ahead asked for, as the app trains."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.tools.practice_bench import main, run


@pytest.mark.slow
def test_looking_ahead_on_the_practice_game() -> None:
    """Level 3 with the simple table, twelve goes of 300 steps: the run of
    ``test_the_ground_holds_through_going_back``, which is made without looking ahead
    and loses 9 lives to walkers, finishes 12 levels and goes back 36 times. Measured
    with it: 2 lives lost to walkers, 15 levels finished, 28 searches of the goes' own
    and 18 more before a life went, 17 of which found a way; it looked ahead 78 times.

    Since T-1062 a search after the game has put the player back finds a way on where
    it made every try for nothing, and from the first of those (go 6) the run goes
    another way: 4 lives lost to walkers, 15 levels finished, 30 searches of the goes'
    own and 10 more before a life went, 7 of which found a way; it looked ahead 62
    times. The least number of lives seen going is 7 for that, where it was 12 (Boss
    agreed to this one change on 20 Nov)."""
    far = run(goes=12, seed=3, algorithm="q_table", max_steps=300, look_ahead=True)
    assert far.look_ahead
    assert far.lines()[0] == (
        "Practice bench: Toy Hopper level 3, 12 goes of up to 300 steps, Simple table, "
        "looking ahead."
    )
    # It keeps away from what has cost it a life: few lives lost to walkers, and more of
    # the level finished for it.
    assert far.walkers <= 4
    assert sum(far.finished) >= 14
    # It looked ahead often, saw lives going, and found the way past for most of them
    # before they went.
    assert far.looked_ahead >= 50 and far.ways_ahead >= 7
    assert far.ways_ahead_found >= far.ways_ahead * 3 // 4
    assert far.lines()[-1].startswith(f"Looking ahead: {far.looked_ahead:,} times a thing")
    # Those are counted apart: the goes' own searches stay at most three a go.
    assert far.ways_found <= far.ways_sought <= 3 * 12
    # And the ground still holds through all that going back (T-1023), before a life is
    # lost as after.
    assert far.ways_sought + far.ways_ahead >= 40
    assert far.ground_goes == 12 and far.ground_right >= 11


def test_the_bench_looks_ahead_only_where_asked(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ConfigurationError, match="The bench needs look_ahead to be True or"):
        run(1, look_ahead="yes")  # type: ignore[arg-type]
    code = main(["--goes", "1", "--steps", "20", "--play-frames", "600", "--look-ahead"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and out[0] == (
        "Practice bench: Toy Hopper level 0, 1 goes of up to 20 steps, Neural network, "
        "looking ahead."
    )
    # Not asked, the run is the one the bench made before: nothing is said of it.
    plain = run(1, max_steps=20, play_frames=600)
    assert not plain.look_ahead and plain.looked_ahead == 0
    assert plain.lines()[0] == (
        "Practice bench: Toy Hopper level 0, 1 goes of up to 20 steps, Neural network."
    )
    assert not any(line.startswith("Looking ahead") for line in plain.lines())
    code = main(["--goes", "1", "--steps", "20", "--play-frames", "600"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and out == plain.lines()
