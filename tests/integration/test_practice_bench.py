"""T-986: the practice bench runs the whole app on Toy Hopper and counts against the truth."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.tools import practice_bench
from retro_ai_lab.tools.practice_bench import Report, main, run

SHORT = dict(seed=0, algorithm="q_table", max_steps=80, play_frames=2000)


@pytest.fixture(scope="module")
def report() -> Report:
    return run(5, **SHORT)  # type: ignore[arg-type]


@pytest.mark.slow
def test_the_bench_counts_agree_with_the_game(report: Report) -> None:
    assert (report.seed, report.goes, report.max_steps, report.algorithm) == (0, 5, 80, "q_table")
    # One entry a go, and nothing reached that the level does not have.
    assert len(report.furthest) == len(report.lost) == len(report.finished) == 5
    assert all(32 <= x < 160 * 16 for x in report.furthest)
    assert all(lost >= 0 for lost in report.lost) and all(n >= 0 for n in report.finished)
    # The AI looked at the screen, and the counts nest: right <= followed <= looks.
    assert report.looks > 0
    assert 0 <= report.right <= report.followed <= report.looks
    # Every death is a fall or a walker; the watch can blame no more deaths than it saw.
    assert report.deaths == report.falls + report.walkers
    assert 0 <= report.blamed <= report.deaths + 1  # a dying pause can straddle the run's end
    assert 0 <= report.fell <= report.deaths + 1
    assert sum(report.lost) <= report.deaths
    assert report.kinds >= 1 and report.hurts >= 0 and report.items >= 0
    # Memory: it names a value in the game's RAM as its own position.
    assert report.own_position.startswith("ram[") and isinstance(report.own_position_right, bool)
    assert report.lives_found  # the scripted first go loses lives; discovery sees the counter


@pytest.mark.slow
def test_the_bench_is_repeatable(report: Report) -> None:
    assert run(5, **SHORT) == report  # type: ignore[arg-type]


@pytest.mark.slow
def test_the_report_reads_in_words(report: Report) -> None:
    lines = report.lines()
    assert len(lines) == 6
    assert lines[0] == "Practice bench: Toy Hopper level 0, 5 goes of up to 80 steps, Simple table."
    assert lines[1].startswith(
        f"Eyes: {report.looks:,} looks; followed what it takes for itself in "
    )
    assert "(target 95%)" in lines[1] and "(target under 30)" in lines[1]
    assert (
        f"read as the level has it in {round(100 * report.floor_right / report.around)}% of "
        f"the {report.around:,} looks in which it read what is around it (target 95%)."
    ) in lines[1]
    assert lines[2].startswith(
        f"Camera: it moved between two looks {report.camera_moved:,} times, measured right in "
    )
    assert (
        "(target 90%)" in lines[2] and f"it stood still {report.camera_still:,} times" in lines[2]
    )
    assert lines[2].endswith(f"{report.camera_cuts} cuts to another place not counted.")
    assert lines[3].startswith(f"Deaths: {report.deaths} ({report.falls} falls, ")
    assert "target 80%" in lines[3]
    assert f"the eyes put {report.fell} down to a fall and {report.blamed} to touching" in lines[3]
    assert ("% explained" in lines[3]) == bool(report.deaths)
    assert lines[4].startswith(f"Memory: measures distances from {report.own_position} (")
    assert "lives found" in lines[4]
    assert lines[5].startswith("Brain: furthest reached ") and "of 2,496 to the flag" in lines[5]
    # A run in which the AI never looked says so instead of dividing by nothing.
    blind = Report(seed=1, goes=2, max_steps=10, algorithm="dqn", furthest=[40, 50], lost=[0, 1])
    blind.finished = [0, 0]
    words = blind.lines()
    assert words[1].startswith("Eyes: the AI never looked at the screen")
    assert words[2] == "Camera: not followed (the AI never looked at the screen twice in a go)."
    assert "none to explain" in words[3] and "Neural network" in words[0]
    assert (
        "furthest reached 45 (best 50)" in words[5] and "lives lost a go 0.50 then 0.50" in words[5]
    )


def test_the_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--goes", "1", "--steps", "20", "--play-frames", "600", "--algorithm", "q_table"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and len(out) == 6
    assert out[0] == "Practice bench: Toy Hopper level 0, 1 goes of up to 20 steps, Simple table."
    for bad in (
        ["--goes", "0"],
        ["--steps", "many"],
        ["--seed", "70000"],
        ["--algorithm", "magic"],
    ):
        with pytest.raises(SystemExit):
            main(bad)
    for kwargs in (dict(goes=0), dict(goes=1, max_steps=0), dict(goes=1, play_frames=True)):
        with pytest.raises(ConfigurationError):
            run(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="No way of learning called"):
        run(1, algorithm="magic")
    assert practice_bench.FOLLOWED_TARGET == 95 and practice_bench.EXPLAINED_TARGET == 80
    assert practice_bench.FLOOR_TARGET == 95 and practice_bench.CAMERA_TARGET == 90


@pytest.mark.slow
def test_the_eyes_are_on_the_player(report: Report) -> None:
    """T-987: the AI knows itself by its colours. In all but the looks in which the
    player is dying or has fallen off the screen, what it takes for itself is the
    player (the baseline before: followed in 39% of looks, rightly in 19%)."""
    assert report.looks >= 100
    assert report.followed == report.right  # never anything else taken for itself
    assert report.right >= 0.85 * report.looks
    # With the player out of the tracker's picture, it no longer makes kinds of its own
    # (itself half behind something, itself and what it touches as one patch).
    assert report.kinds <= 6


@pytest.mark.slow
def test_deaths_are_explained() -> None:
    """T-988: every life lost reaches the eyes (on this game none used to: the dying
    pause is a whole number of looks long), and a fall is known for a fall."""
    report = run(12, seed=3, algorithm="q_table", max_steps=200, play_frames=2000)
    assert report.deaths >= 3 and report.walkers == 0  # level 3's first deaths are falls
    # Each fall the game counted, the eyes counted; none was put down to the pillar
    # the player stood on before it fell. (The last death's pause can outlast the run.)
    assert report.falls - 1 <= report.fell <= report.falls
    assert report.blamed == 0 and report.hurts == 0
    assert "% explained" in report.lines()[3]


@pytest.mark.slow
def test_the_floor_is_read_right(report: Report) -> None:
    """T-989: what the AI reads of the floor to its right and left, at every look in
    which it sees itself, is what the game's own level has there."""
    assert report.around >= 0.85 * report.looks
    assert report.around <= report.followed  # read only where it sees itself
    assert report.floor_right >= 0.95 * report.around
    # A longer run on a level whose first deaths are falls: it is at the gaps a lot.
    gaps = run(12, seed=3, algorithm="q_table", max_steps=200, play_frames=2000)
    assert gaps.falls >= 3 and gaps.around >= 500
    assert gaps.floor_right >= 0.95 * gaps.around


@pytest.mark.slow
def test_the_camera_is_followed(report: Report) -> None:
    """T-992: the camera's move between two looks, as the tracker measures it, is the
    game's own (before: right in about 3 of 10 looks in which it moved)."""
    counted = report.camera_moved + report.camera_still + report.camera_cuts
    assert counted == report.looks - report.goes  # every look but each go's first
    assert report.camera_moved >= 15  # a short run: the table mostly stands and jumps
    assert report.camera_moved_right >= 0.9 * report.camera_moved
    assert report.camera_still_right >= 0.99 * report.camera_still
    # A level on which it runs and falls more.
    gaps = run(12, seed=3, algorithm="q_table", max_steps=200, play_frames=2000)
    assert gaps.camera_moved >= 100
    assert gaps.camera_moved_right >= 0.9 * gaps.camera_moved
    assert gaps.camera_still_right >= 0.99 * gaps.camera_still
