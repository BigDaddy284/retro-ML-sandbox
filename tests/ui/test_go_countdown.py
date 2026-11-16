"""T-1046: the Learning panel's countdown for the go in progress."""

from __future__ import annotations

import pytest

from retro_ai_lab.training import Mode, Status
from retro_ai_lab.training.service import GoLeft, TrainingService
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.learning_panel import (
    COUNTDOWN,
    COUNTDOWN_UP,
    IDLE_TEXT,
    LearningPanel,
    countdown_line,
    seconds_to_steps,
)
from tests.ui import test_learning_panel as the_panel
from tests.ui.test_learning_panel import connect

context = the_panel.context  # the app's services on a folder of their own (pytest fixture)
panel = the_panel.panel  # the Learning panel, shown (pytest fixture)

FOR_A_TIME = "This go has {} s left."
UNTIL_OVER = "This go ends if nothing is paid in the next {} s."
TIME_UP = "This go's time is up: it ends when the way it has just found is played out."


def lines(panel: LearningPanel) -> list[str]:
    return panel.status_text.split("\n")


def countdowns(panel: LearningPanel) -> list[str]:
    """The status lines that are a countdown."""
    return [line for line in lines(panel) if line.startswith("This go")]


def test_the_words() -> None:
    assert COUNTDOWN == {False: FOR_A_TIME, True: UNTIL_OVER} and COUNTDOWN_UP == TIME_UP
    # Whole seconds of the game's time, rounded up: at 4 frames a step, 15 steps a second.
    for steps, seconds in ((600, 40), (599, 40), (586, 40), (585, 39), (16, 2), (15, 1), (1, 1)):
        assert countdown_line(GoLeft(steps, False), 4) == FOR_A_TIME.format(seconds)
        assert countdown_line(GoLeft(steps, True), 4) == UNTIL_OVER.format(seconds)
    # Other frames to a step: 60 frames are a second whatever the step.
    assert countdown_line(GoLeft(60, False), 1) == "This go has 1 s left."
    assert countdown_line(GoLeft(61, False), 1) == "This go has 2 s left."
    assert countdown_line(GoLeft(7, True), 9) == UNTIL_OVER.format(2)  # 63 frames
    assert countdown_line(GoLeft(27_000, False), 4) == "This go has 1800 s left."
    # Nothing left of a go still being played: only a way just found goes on.
    assert countdown_line(GoLeft(0, False), 4) == countdown_line(GoLeft(0, True), 4) == TIME_UP


def test_no_countdown_until_a_go_is_played(panel: LearningPanel, context: AppContext) -> None:
    assert panel.status_text == IDLE_TEXT
    connect(context)
    panel.refresh()
    assert lines(panel)[0] == "Ready to train." and countdowns(panel) == []


def test_a_go_of_a_set_time_counts_down(panel: LearningPanel, context: AppContext) -> None:
    connect(context)
    t = context.training
    assert panel.go_seconds == 40.0 and t.max_steps == seconds_to_steps(40, t.frames_per_step)
    panel.choose_mode(Mode.INDEPENDENT)
    panel.start()
    # Under the go's own line, from before its first step.
    assert lines(panel)[:2] == ["Go 1: playing, step 0, +0 points so far.", FOR_A_TIME.format(40)]
    read = []
    for _ in range(40):
        assert panel.tick() == 1
        shown, left, episode = countdowns(panel), t.go_left, t.episode
        assert left is not None and len(shown) == 1
        assert lines(panel)[1] == shown[0] == countdown_line(left, t.frames_per_step)
        # What the go has played is off its time (a go the game ended has a next one).
        played = 0 if episode is None else episode.own_steps
        assert left == GoLeft(t.max_steps - played, False)
        if t.episodes == 0:  # in its first go
            read.append(shown[0])
    # A step is a fifteenth of a second: the same second for fifteen steps, then one less.
    assert read[:14] == [FOR_A_TIME.format(40)] * len(read[:14])
    assert read[14:29] == [FOR_A_TIME.format(39)] * len(read[14:29])
    assert read[29:] == [FOR_A_TIME.format(38)] * len(read[29:])
    # Paused, it stays as it was.
    before = countdowns(panel)
    panel.pause_or_resume()
    assert t.status is Status.PAUSED and "paused" in lines(panel)[0]
    assert countdowns(panel) == before and lines(panel)[1] == before[0]
    panel.pause_or_resume()
    assert countdowns(panel) == before
    # Stopped, there is no go to count down.
    panel.stop()
    assert countdowns(panel) == [] and lines(panel)[0].startswith("Resting after")
    # The next go has the whole time again.
    panel.start()
    assert lines(panel)[1] == FOR_A_TIME.format(40)
    # Another time in the box is another countdown.
    panel.stop()
    panel.choose_go_seconds(7.5)
    panel.start()
    assert lines(panel)[1] == FOR_A_TIME.format(8)
    panel.tick()
    assert t.go_left == GoLeft(seconds_to_steps(7.5, t.frames_per_step) - 1, False)


def test_a_go_until_the_game_is_over_counts_its_wait(
    panel: LearningPanel, context: AppContext
) -> None:
    connect(context)
    t = context.training
    panel.choose_mode(Mode.INDEPENDENT)
    panel.choose_go_until_over(True)
    panel.choose_go_seconds(2.0)
    assert countdowns(panel) == []
    panel.start()
    assert lines(panel)[1] == UNTIL_OVER.format(2)
    for _ in range(40):
        panel.tick()
        left = t.go_left
        assert left is not None and left.unpaid and left.steps <= seconds_to_steps(2, 4) == 30
        assert countdowns(panel) == [UNTIL_OVER.format(2 if left.steps > 15 else 1)]
    panel.stop()
    assert countdowns(panel) == []


def test_the_panel_says_what_the_service_says(
    panel: LearningPanel, context: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect(context)
    panel.choose_mode(Mode.INDEPENDENT)
    panel.start()
    said: list[GoLeft | None] = [None]
    monkeypatch.setattr(TrainingService, "go_left", property(lambda self: said[0]))
    for left, line in (
        (GoLeft(31, False), "This go has 3 s left."),
        (GoLeft(30, False), "This go has 2 s left."),
        (GoLeft(1, False), "This go has 1 s left."),
        (GoLeft(450, True), "This go ends if nothing is paid in the next 30 s."),
        (GoLeft(1, True), "This go ends if nothing is paid in the next 1 s."),
        (GoLeft(0, False), TIME_UP),
        (GoLeft(0, True), TIME_UP),
    ):
        said[0] = left
        panel.refresh()
        assert lines(panel)[0].startswith("Go 1: playing") and lines(panel)[1] == line
        assert countdowns(panel) == [line]
    said[0] = None
    panel.refresh()
    assert countdowns(panel) == [] and lines(panel)[0].startswith("Go 1: playing")


def test_a_go_the_child_plays_has_the_countdown(panel: LearningPanel, context: AppContext) -> None:
    connect(context)
    panel.choose_mode(Mode.MANUAL)
    panel.start()
    assert "watching you play" in lines(panel)[0]
    assert lines(panel)[1] == FOR_A_TIME.format(40)
