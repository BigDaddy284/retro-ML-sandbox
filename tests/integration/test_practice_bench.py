"""T-986: the practice bench runs the whole app on Toy Hopper and counts against the truth."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.tools import practice_bench
from retro_ai_lab.tools.practice_bench import Report, main, run

SHORT = dict(seed=0, algorithm="q_table", max_steps=80, play_frames=2000)
# The run the eyes were measured on in T-988, T-992 and T-993: level 3, on which the
# simple table runs and falls, without the way it remembers (T-1015). With it the same
# brain gets to the end of the level, past walkers, where the eyes have gaps that are
# known and written down (TASK_TRACKER.md, T-1015): these three tests are about the
# first screens of the level, and stay on the goes they were written for.
NEAR = dict(goes=12, seed=3, algorithm="q_table", max_steps=200, play_frames=2000, remember=False)


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
    assert len(lines) == 8
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
    assert lines[3].startswith(
        f"Ground: {report.ground_steps:,} steps of new ground counted for pay; the furthest it "
        "counted to the right was as far as the game's own camera went, to within a step, in "
    )
    assert lines[3].endswith(f"% of the {report.ground_goes} goes (target 90%).")
    assert lines[4].startswith(f"Deaths: {report.deaths} ({report.falls} falls, ")
    assert "target 80%" in lines[4]
    assert f"the eyes put {report.fell} down to a fall and {report.blamed} to touching" in lines[4]
    assert ("% explained" in lines[4]) == bool(report.deaths)
    assert lines[5].startswith(f"Memory: measures distances from {report.own_position} (")
    assert "lives found" in lines[5]
    assert lines[6].startswith("Brain: furthest reached ") and "of 2,496 to the flag" in lines[6]
    assert lines[7].startswith("Ways past: ")
    # A run in which the AI never looked says so instead of dividing by nothing.
    blind = Report(seed=1, goes=2, max_steps=10, algorithm="dqn", furthest=[40, 50], lost=[0, 1])
    blind.finished = [0, 0]
    words = blind.lines()
    assert words[1].startswith("Eyes: the AI never looked at the screen")
    assert words[2] == "Camera: not followed (the AI never looked at the screen twice in a go)."
    assert words[3] == "Ground: none counted (the AI never looked at the screen)."
    assert "none to explain" in words[4] and "Neural network" in words[0]
    assert (
        "furthest reached 45 (best 50)" in words[6] and "lives lost a go 0.50 then 0.50" in words[6]
    )


def test_the_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--goes", "1", "--steps", "20", "--play-frames", "600", "--algorithm", "q_table"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and len(out) == 8
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
    # T-1015: without the way it remembers, said in the first line.
    with pytest.raises(ConfigurationError, match="remember"):
        run(1, remember="no")  # type: ignore[arg-type]
    code = main(["--goes", "1", "--steps", "20", "--play-frames", "600", "--no-memory"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and out[0] == (
        "Practice bench: Toy Hopper level 0, 1 goes of up to 20 steps, Neural network, "
        "without the way it remembers."
    )
    assert practice_bench.FOLLOWED_TARGET == 95 and practice_bench.EXPLAINED_TARGET == 80
    assert practice_bench.FLOOR_TARGET == 95 and practice_bench.CAMERA_TARGET == 90
    assert practice_bench.GROUND_TARGET == 90


@pytest.mark.slow
def test_the_bench_reports_the_ways_past() -> None:
    """T-1010: training goes back at a life lost, and the bench says what came of it."""
    # Measured: 12 deaths, 9 searches, 8 ways found. (The short runs on level 3 with the
    # simple table, used below, search 4 times and find nothing: its goes wander.)
    first = run(8, seed=0, algorithm="dqn", max_steps=200, play_frames=2000)
    assert first.deaths >= 3
    assert 1 <= first.ways_found <= first.ways_sought <= 3 * 8  # at most three a go
    assert first.ways_sought <= first.deaths  # one search a life lost at most
    share = round(100 * first.ways_found / first.ways_sought)
    assert first.lines()[7] == (
        f"Ways past: after losing a life it went back {first.ways_sought:,} "
        f"time{'' if first.ways_sought == 1 else 's'} and found a way past in "
        f"{first.ways_found:,} ({share}%)."
    )
    # A run in which it never went back says so, and one search is not "1 times".
    none = Report(seed=1, goes=1, max_steps=10, algorithm="dqn", furthest=[40], lost=[0])
    none.finished = [0]
    assert none.lines()[7] == "Ways past: it never went back to look for one."
    none.ways_sought, none.ways_found = 1, 1
    assert none.lines()[7] == (
        "Ways past: after losing a life it went back 1 time and found a way past in 1 (100%)."
    )


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
    report = run(**NEAR)  # type: ignore[arg-type]
    assert report.deaths >= 3 and report.walkers == 0  # level 3's first deaths are falls
    # Each fall the game counted, the eyes counted; none was put down to the pillar
    # the player stood on before it fell. (The last death's pause can outlast the run.)
    assert report.falls - 1 <= report.fell <= report.falls
    assert report.blamed == 0 and report.hurts == 0
    assert "% explained" in report.lines()[4]


@pytest.mark.slow
def test_lives_lost_to_a_walker_are_put_down_to_it() -> None:
    """T-1016: three goes on level 0, the app's way (the remembered way on), get to the
    first walkers and lose lives to them. Before, none of those lives was put down to
    touching anything and no kind was found to hurt (0 of 28 and 0 of 17 in the long
    runs): the walker was seen only by the edge it leads with, or not at all."""
    report = run(goes=3, seed=0, algorithm="q_table", max_steps=250, play_frames=2000)
    assert report.walkers >= 2 and report.falls >= 1
    # Each to its own cause. (The last death's pause can outlast the run.)
    assert report.walkers - 1 <= report.blamed <= report.walkers
    assert report.falls - 1 <= report.fell <= report.falls
    assert report.hurts == 1 and report.kinds <= 6
    assert report.right == report.followed  # and what it follows is still the player


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
    # Every look but each go's first, and the first after the game was put back to an
    # earlier moment (T-1015: this run now goes back once; the camera's jump there is
    # the app's own doing).
    assert counted == report.looks - report.goes - report.camera_skipped
    assert 0 <= report.camera_skipped <= report.ways_found
    assert report.camera_moved >= 15  # a short run: the table mostly stands and jumps
    assert report.camera_moved_right >= 0.9 * report.camera_moved
    assert report.camera_still_right >= 0.99 * report.camera_still
    # A level on which it runs and falls more.
    gaps = run(**NEAR)  # type: ignore[arg-type]
    assert gaps.camera_moved >= 100
    assert gaps.camera_moved_right >= 0.9 * gaps.camera_moved
    assert gaps.camera_still_right >= 0.99 * gaps.camera_still


@pytest.mark.slow
def test_new_ground_is_paid_as_the_game_has_it(report: Report) -> None:
    """T-993: the ground the AI is paid for is counted off the screen: on a game that
    scrolls, the camera's travel. Go by go, the furthest it counted to the right is
    how far the game's own camera went, to within a step of the player's size.

    T-998: in a go that begins before the camera has moved at all since the AI started,
    the watch has only the player's place on the screen to go by, and the game's own
    figure up to the camera's first move is how far the player itself went. The first
    go is always one, and here the second too: the AI stays near the start in both."""
    assert report.ground_goes == report.goes
    assert 1 <= report.ground_before < report.ground_goes
    assert report.ground_right >= 0.9 * report.ground_goes
    assert report.ground_steps >= 1
    # A level on which it runs further and dies more: the count holds through the
    # falls, and a life lost does not get the same ground paid twice. A step is 16
    # pixels, and the camera starts to follow 80 pixels into the level, so a go's
    # steps are those of its furthest less five, give or take one; nothing is paid
    # for jumping or for falling into a gap.
    gaps = run(**NEAR)  # type: ignore[arg-type]
    assert gaps.ground_goes == 12 and gaps.ground_right >= 11
    assert gaps.ground_before == 1  # it runs: the camera moves in the first go
    owed = sum(max(0, far - 80) // 16 for far in gaps.furthest)
    assert owed - 12 <= gaps.ground_steps <= owed + 12 + 3  # the first go's 3 on the screen


@pytest.mark.slow
def test_every_go_that_finishes_a_level_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-1015: the flag is brought to just beyond the first gap, so short goes finish levels."""
    from retro_ai_lab.emulator.fake import toy_hopper
    from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT, ToyHopper

    monkeypatch.setattr(toy_hopper, "FLAG_COL", 20)
    passed = 0  # every time the game itself moved on a level, tries of a search included
    simulate = ToyHopper._simulate

    def counted(game: ToyHopper, *args: object, **kwargs: object) -> None:
        nonlocal passed
        level = game.ram[LAYOUT.level]
        simulate(game, *args, **kwargs)  # type: ignore[arg-type]
        passed += game.ram[LAYOUT.level] > level

    monkeypatch.setattr(ToyHopper, "_simulate", counted)
    before_training = 0

    def first_go(*args: object, **kwargs: object) -> object:
        nonlocal before_training
        out = scripted(*args, **kwargs)
        before_training = passed
        return out

    scripted = practice_bench.first_go
    monkeypatch.setattr(practice_bench, "first_go", first_go)
    report = run(8, seed=0, algorithm="q_table", max_steps=150, play_frames=2000)
    goes = sum(1 for n in report.finished if n)
    # The old count read the level before the go was put back to its start, so of goes
    # that each finish the level only the first was counted.
    assert goes >= 3 and len(report.finished) == 8
    assert goes <= sum(report.finished) <= passed - before_training
    assert 0 <= report.clean <= goes
    assert report.clean == sum(
        1 for n, lost in zip(report.finished, report.lost, strict=True) if n and not lost
    ) or report.clean >= sum(
        1 for n, lost in zip(report.finished, report.lost, strict=True) if n and not lost
    )  # a life lost after the flag does not spoil a level finished before it
    line = report.lines()[6]
    assert (
        f"levels finished {sum(report.finished)}, in {goes} goes, {report.clean} of them " in line
    )
    assert line.endswith("before any life was lost.")
    # None finished: nothing more is said. One go is not "1 goes".
    none = Report(seed=1, goes=2, max_steps=10, algorithm="dqn", furthest=[40, 50], lost=[0, 1])
    none.finished = [0, 0]
    assert none.lines()[6].endswith("levels finished 0.")
    none.finished, none.clean = [0, 2], 1
    assert none.lines()[6].endswith(
        "levels finished 2, in 1 go, 1 of them before any life was lost."
    )
