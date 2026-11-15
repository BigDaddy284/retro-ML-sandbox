"""T-1037: the bench can give the learner other dice on the same level."""

from __future__ import annotations

from typing import Any

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.services.connection import ConnectionService
from retro_ai_lab.services.observer import ObservationService
from retro_ai_lab.tools import practice_bench
from retro_ai_lab.tools.practice_bench import Report, main, run
from retro_ai_lab.training.service import TrainingService

FIRST = "Practice bench: Toy Hopper level 3, 1 goes of up to 20 steps, Simple table"
QUICK: dict[str, Any] = {"max_steps": 20, "play_frames": 600, "algorithm": "q_table"}


@pytest.fixture
def seeds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int | None]]:
    """The seed the learner was built with and the seed the game was connected with."""
    seeds: list[tuple[str, int | None]] = []
    connect = ConnectionService.connect

    class Training(TrainingService):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seeds.append(("learner", kwargs["seed"]))
            super().__init__(*args, **kwargs)

    def connecting(self: ConnectionService, *args: Any, **kwargs: Any) -> Any:
        seeds.append(("level", kwargs["seed"]))
        return connect(self, *args, **kwargs)

    monkeypatch.setattr(practice_bench, "TrainingService", Training)
    monkeypatch.setattr(ConnectionService, "connect", connecting)
    return seeds


def test_the_learner_gets_the_dice_and_the_level_stays(
    seeds: list[tuple[str, int | None]],
) -> None:
    report = run(1, seed=3, dice=7, **QUICK)
    # The level's number and the dice together seed the learner; the game has the level's alone.
    assert seeds == [("learner", 10), ("level", 3)]
    assert report.seed == 3 and report.dice == 7
    assert report.lines()[0] == FIRST + ", with other dice (7)."
    # With none named everything is as it was.
    seeds.clear()
    report = run(1, seed=3, **QUICK)
    assert seeds == [("learner", 3), ("level", 3)]
    assert report.dice == 0 and report.lines()[0] == FIRST + "."
    # The ends of what it takes.
    seeds.clear()
    assert run(1, seed=3, dice=0, **QUICK).lines()[0] == FIRST + "."
    assert run(1, seed=3, dice=65535, **QUICK).lines()[0] == FIRST + ", with other dice (65535)."
    assert seeds == [("learner", 3), ("level", 3), ("learner", 65538), ("level", 3)]


def test_the_first_line_says_which_dice() -> None:
    def first(**said: Any) -> str:
        return Report(seed=3, goes=1, max_steps=20, algorithm="q_table", **said).lines()[0]

    assert first() == FIRST + "."
    assert first(dice=1000) == FIRST + ", with other dice (1000)."
    # Said last, after everything else a run can be with or without.
    assert first(remember=False, stand=False, dying_frames=200, dice=1) == (
        FIRST + ", without the way it remembers, without going back where it stands still, "
        "with a dying pause of 200 frames, with other dice (1)."
    )


def test_the_same_dice_the_same_run_and_other_dice_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    played: list[int] = []  # the frames of each scripted first go
    first_go = practice_bench.first_go

    def first(connection: ConnectionService, observation: ObservationService, frames: int) -> None:
        first_go(connection, observation, frames)
        adapter = connection.adapter
        assert adapter is not None
        played.append(hash(bytes(adapter.game.snapshot())))  # type: ignore[attr-defined]

    monkeypatch.setattr(practice_bench, "first_go", first)
    quick: dict[str, Any] = {"max_steps": 60, "play_frames": 600, "algorithm": "q_table"}
    usual = run(3, seed=3, **quick)
    other = run(3, seed=3, dice=1, **quick)
    again = run(3, seed=3, dice=1, **quick)
    # The same dice: the same run, line for line.
    assert other.lines() == again.lines()
    assert (other.furthest, other.lost, other.deaths) == (again.furthest, again.lost, again.deaths)
    # Other dice: the same level and the same scripted first go, and another training.
    assert played[0] == played[1] == played[2]
    assert usual.lines()[0][:-1] + ", with other dice (1)." == other.lines()[0]
    assert (usual.furthest, usual.lost) != (other.furthest, other.lost)


def test_the_command_line_takes_the_dice(
    seeds: list[tuple[str, int | None]], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--seed", "3", "--goes", "1", "--steps", "20", "--play-frames", "600"]
    argv += ["--algorithm", "q_table"]
    assert main([*argv, "--dice", "7"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == FIRST + ", with other dice (7)."
    assert seeds == [("learner", 10), ("level", 3)]
    seeds.clear()
    assert main(argv) == 0
    assert capsys.readouterr().out.splitlines()[0] == FIRST + "."
    assert seeds == [("learner", 3), ("level", 3)]
    seeds.clear()
    assert main([*argv, "--dice", "0"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == FIRST + "."
    seeds.clear()
    for bad, why in (
        ("-1", "must be from 0 to 65535."),
        ("65536", "must be from 0 to 65535."),
        ("loaded", "'loaded' is not a whole number."),
        ("1.5", "'1.5' is not a whole number."),
        ("", "'' is not a whole number."),
    ):
        with pytest.raises(SystemExit):
            main([*argv, "--dice", bad])
        assert f"argument --dice: {why}" in capsys.readouterr().err
    assert seeds == []  # refused before anything is built or connected


def test_dice_that_are_no_dice_are_refused(seeds: list[tuple[str, int | None]]) -> None:
    for wrong in (True, False, -1, 65536, 2.5, "7", None):
        with pytest.raises(ConfigurationError, match="dice as a whole number from 0 to 65535"):
            run(1, seed=3, dice=wrong, **QUICK)
    assert seeds == []  # refused before anything is built or connected
