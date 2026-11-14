"""T-1030: the bench can be run with the practice game's dying pause set."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import ToyHopper, toy_hopper
from retro_ai_lab.services.connection import ConnectionService
from retro_ai_lab.services.observer import ObservationService
from retro_ai_lab.tools import practice_bench
from retro_ai_lab.tools.practice_bench import Report, main, run
from retro_ai_lab.training.service import TrainingService

FIRST = "Practice bench: Toy Hopper level 0, 1 goes of up to 20 steps, Simple table"


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """The pause the game had when the scripted first go began and when training did."""
    seen: list[tuple[str, int]] = []
    first_go = practice_bench.first_go
    training_itself = practice_bench._train

    def first(connection: ConnectionService, observation: ObservationService, frames: int) -> None:
        adapter = connection.adapter
        assert adapter is not None
        seen.append(("first go", adapter.game.dying_frames))  # type: ignore[attr-defined]
        first_go(connection, observation, frames)

    def train(training: TrainingService, game: ToyHopper, report: Report) -> None:
        seen.append(("training", game.dying_frames))
        training_itself(training, game, report)

    monkeypatch.setattr(practice_bench, "first_go", first)
    monkeypatch.setattr(practice_bench, "_train", train)
    return seen


def test_the_bench_trains_with_the_pause_it_is_given(seen: list[tuple[str, int]]) -> None:
    report = run(1, max_steps=20, play_frames=600, algorithm="q_table", dying_frames=200)
    # Set before anything is played: what the app learns, it learns with that pause.
    assert seen == [("first go", 200), ("training", 200)]
    assert report.dying_frames == 200
    assert report.lines()[0] == FIRST + ", with a dying pause of 200 frames."
    # With nothing set the game has its own, and the report is the one it was.
    seen.clear()
    report = run(1, max_steps=20, play_frames=600, algorithm="q_table")
    assert seen == [("first go", toy_hopper.DYING_FRAMES), ("training", toy_hopper.DYING_FRAMES)]
    assert report.dying_frames is None and report.lines()[0] == FIRST + "."
    # Said after the other things a run can be without, and one frame is one frame.
    seen.clear()
    report = run(
        1, max_steps=20, play_frames=600, algorithm="q_table", remember=False, dying_frames=1
    )
    assert seen == [("first go", 1), ("training", 1)]
    assert report.lines()[0] == (
        FIRST + ", without the way it remembers, with a dying pause of 1 frame."
    )
    assert run(1, max_steps=20, play_frames=600, algorithm="q_table", dying_frames=255).lines()[
        0
    ] == (FIRST + ", with a dying pause of 255 frames.")


def test_the_command_line_takes_the_pause(
    seen: list[tuple[str, int]], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--goes", "1", "--steps", "20", "--play-frames", "600", "--algorithm", "q_table"]
    assert main([*argv, "--dying-frames", "200"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == FIRST + ", with a dying pause of 200 frames."
    assert seen == [("first go", 200), ("training", 200)]
    seen.clear()
    assert main(argv) == 0
    assert capsys.readouterr().out.splitlines()[0] == FIRST + "."
    assert seen == [("first go", toy_hopper.DYING_FRAMES), ("training", toy_hopper.DYING_FRAMES)]
    seen.clear()
    for bad in ("0", "256", "-5", "long", "1.5"):
        with pytest.raises(SystemExit):
            main([*argv, "--dying-frames", bad])
    assert "must be from 1 to 255" in capsys.readouterr().err
    for wrong in (True, 0, 256, 2.5, "200"):
        with pytest.raises(ConfigurationError, match="dying pause as a whole number of frames"):
            run(1, max_steps=20, play_frames=600, dying_frames=wrong)  # type: ignore[arg-type]
    assert seen == []  # refused before anything is connected
