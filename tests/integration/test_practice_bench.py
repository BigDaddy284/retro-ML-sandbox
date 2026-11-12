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
    assert len(lines) == 5
    assert lines[0] == "Practice bench: Toy Hopper level 0, 5 goes of up to 80 steps, Simple table."
    assert lines[1].startswith(
        f"Eyes: {report.looks:,} looks; followed what it takes for itself in "
    )
    assert "(target 95%)" in lines[1] and "(target under 30)" in lines[1]
    assert lines[2].startswith(f"Deaths: {report.deaths} ({report.falls} falls, ")
    assert "target 80%" in lines[2]
    assert lines[3].startswith(f"Memory: measures distances from {report.own_position} (")
    assert "lives found" in lines[3]
    assert lines[4].startswith("Brain: furthest reached ") and "of 2,496 to the flag" in lines[4]
    # A run in which the AI never looked says so instead of dividing by nothing.
    blind = Report(seed=1, goes=2, max_steps=10, algorithm="dqn", furthest=[40, 50], lost=[0, 1])
    blind.finished = [0, 0]
    words = blind.lines()
    assert words[1].startswith("Eyes: the AI never looked at the screen")
    assert "none to explain" in words[2] and "Neural network" in words[0]
    assert (
        "furthest reached 45 (best 50)" in words[4] and "lives lost a go 0.50 then 0.50" in words[4]
    )


def test_the_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--goes", "1", "--steps", "20", "--play-frames", "600", "--algorithm", "q_table"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and len(out) == 5
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
