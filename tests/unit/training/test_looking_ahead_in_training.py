"""T-1052: the training service has its goes look ahead where it is asked to, the app
and the practice bench ask, and the bench counts it."""

from __future__ import annotations

from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.tools.practice_bench import Report
from retro_ai_lab.training.moments import Moments
from retro_ai_lab.ui.context import AppContext
from tests.unit.training import test_going_back_in_training as in_training
from tests.unit.training.test_going_back_in_training import NO_SAVES, Lab, walking_right

lab = in_training.lab  # the services a person trains with (pytest fixture)


def test_the_service_has_its_goes_look_ahead_where_asked(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    for wrong in (1, 0, "yes", None):
        with pytest.raises(ConfigurationError, match="look_ahead must be True or False"):
            lab.training(look_ahead=wrong)
    t = lab.training(look_ahead=True)
    assert t.looks_ahead
    lab.connect()
    walking_right(t, monkeypatch)
    assert t.step()
    go = t.episode
    assert go is not None and go.looks_ahead
    t.stop()
    lab.connection.disconnect()
    # Not asked (a service made without saying so trains as before), or where a go
    # cannot go back at all: the runner is not told to.
    for kw, connect in (
        ({}, {}),
        ({"look_ahead": False}, {}),
        ({"look_ahead": True, "ways_per_go": 0}, {}),
        ({"look_ahead": True}, {"capabilities": NO_SAVES}),
    ):
        t = lab.training(**kw)
        assert t.looks_ahead is bool(kw.get("look_ahead"))
        lab.connect(**connect)
        walking_right(t, monkeypatch)
        assert t.step()
        go = t.episode
        assert go is not None and not go.looks_ahead
        t.stop()
        lab.connection.disconnect()


def test_the_app_asks_for_it(tmp_path: Path) -> None:
    assert AppContext(data_dir=tmp_path).training.looks_ahead


def test_the_moment_before_a_life_that_would_go_some_steps_on() -> None:
    moments = Moments(("a", "b", "c"), every=20, lead=30)
    for _ in range(45):
        if moments.due():
            moments.saved()
        moments.did(1)
    assert [m.step for m in moments.kept] == [0, 20, 40]
    start, second, third = moments.kept

    def before(ahead: int) -> tuple[object, int] | None:
        found = moments.before_loss(ahead)
        return None if found is None else (found[0], len(found[1]))

    assert moments.before_loss() == moments.before_loss(0)
    assert before(0) == (start, 45)  # 45 steps back: the only one 30 before now
    assert before(4) == (start, 45)
    assert before(5) == (second, 25)  # 25 steps back and 5 on: 30 before that step
    assert before(24) == (second, 25)
    assert before(25) == (third, 5)
    assert before(1000) == (third, 5)
    young = Moments(("a",), every=20, lead=30)
    young.saved()
    for _ in range(3):
        young.did(1)
    assert young.before_loss() is None and young.before_loss(26) is None
    found = young.before_loss(27)
    assert found is not None and found[0].step == 0 and found[1] == (1, 1, 1)
    for wrong in (-1, 1.5, True, "3", None):
        with pytest.raises(ConfigurationError, match="steps ahead"):
            moments.before_loss(wrong)  # type: ignore[arg-type]


def report(**counts: int) -> Report:
    made = Report(seed=1, goes=2, max_steps=300, algorithm="q_table")
    for name, value in counts.items():
        setattr(made, name, value)
    return made


def test_the_bench_report_says_how_often_it_looked_ahead() -> None:
    # A run that never looked ahead reads as before, line for line.
    old = report(ways_sought=5, ways_found=3, ways_stood=2).lines()
    assert old[-1] == (
        "Ways past: it went back 5 times, 3 after losing a life and 2 where it stood "
        "still with none lost, and found a way past in 3 (60%)."
    )
    assert not any(line.startswith("Looking ahead") for line in old)
    # The searches made before a life went are counted apart, in a line of their own:
    # what is said of the go's own searches does not change.
    lines = report(
        ways_sought=5, ways_found=3, ways_stood=2, looked_ahead=31, ways_ahead=4, ways_ahead_found=3
    ).lines()
    assert lines[:-1] == old
    assert lines[-1] == (
        "Looking ahead: 31 times a thing of a kind that had cost a life came near and it "
        "played ahead what it was about to do; that would have cost a life in 4, and it "
        "went back and found a way past before the life went in 3 of those (75%)."
    )
    lines = report(ways_sought=4, ways_found=1, looked_ahead=1, ways_ahead=1).lines()
    assert lines[-2:] == [
        "Ways past: after losing a life it went back 4 times and found a way past in 1 (25%).",
        "Looking ahead: 1 time a thing of a kind that had cost a life came near and it "
        "played ahead what it was about to do; that would have cost a life in 1, and it "
        "went back and found a way past before the life went in 0 of those (0%).",
    ]
    # Looked ahead and never saw a life going, in a run that never went back.
    lines = report(looked_ahead=7).lines()
    assert lines[-2:] == [
        "Ways past: it never went back to look for one.",
        "Looking ahead: 7 times a thing of a kind that had cost a life came near and it "
        "played ahead what it was about to do; that would have cost a life in 0.",
    ]
