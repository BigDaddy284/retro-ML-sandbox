"""T-1056: whole runs of the learning bench count each go from where it began."""

from __future__ import annotations

from typing import Any

import pytest

from retro_ai_lab.tools import learning_bench
from retro_ai_lab.tools.learning_bench import run


def _places(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every place the bench reads off the game, as the game has it."""
    seen: list[int] = []
    real = learning_bench.how_far

    def spy(game: Any) -> int:
        place = real(game)
        seen.append(place)
        return place

    monkeypatch.setattr(learning_bench, "how_far", spy)
    return seen


def test_a_run_in_which_the_app_moves_the_start_counts_those_goes_from_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The table on the practice racer, seed 2, twelve goes of 100 steps: after a few
    goes that lose a life at the same step the app moves where its goes start."""
    places = _places(monkeypatch)
    report = run("racer", 12, seed=2, algorithm="q_table", max_steps=100, block=6)
    began = [go.began for go in report.played]
    moved = [go for go in report.played if go.began]
    # It began at the game's start, then somewhere further on, and stayed there.
    assert began[0] == 0 and moved and began == sorted(began)
    assert len({go.began for go in moved}) == 1
    # Those are the goes that began a life down, as the report already said.
    assert report.began_short == len(moved)
    assert f"{len(moved)} goes began with a life already gone" in report.lines()[-2]
    for go in report.played:
        # The furthest it got, with the ground it was put past, is a place the game
        # really had it at; without, a moved go that got anywhere would not be.
        assert go.far >= 0
        assert go.far == 0 or go.far + go.began in places
    # A go put 100 or more past the start cannot have got less far than that by the
    # game's own place: what is reported for some of them is less, so it was taken off.
    assert any(0 < go.far < go.began for go in moved)
    # The furthest the game was ever at is further than any go is said to have got.
    assert max(places) > max(go.far for go in report.played)
    again = run("racer", 12, seed=2, algorithm="q_table", max_steps=100, block=6)
    assert again.played == report.played and again.lines() == report.lines()


def test_a_run_in_which_no_start_is_moved_is_counted_by_the_games_own_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The network on the same road for the same goes: its start is left alone."""
    places = _places(monkeypatch)
    report = run("racer", 12, seed=2, algorithm="dqn", max_steps=100, block=6)
    assert [go.began for go in report.played] == [0] * 12 and report.began_short == 0
    assert all(go.far in places for go in report.played)
    assert max(go.far for go in report.played) == max(places) > 0
    assert len(report.lines()) == 4  # no line about goes begun further on
