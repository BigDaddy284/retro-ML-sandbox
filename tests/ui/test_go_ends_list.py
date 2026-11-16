"""T-1045: the Learning panel's list for how long a go lasts."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLabel

from retro_ai_lab.training import DEFAULT_MAX_STEPS, EpisodeRecord, EpisodeResult, Mode, Status
from retro_ai_lab.training.service import until_over_most_steps
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.learning_panel import (
    ENDED_BY,
    GO_ENDS,
    GO_NOTES,
    GO_SECONDS_HINTS,
    GO_SECONDS_WORDS,
    LearningPanel,
    go_cells,
    seconds_to_steps,
    steps_to_seconds,
)
from tests.ui import test_learning_panel as the_panel
from tests.ui.test_learning_panel import connect

context = the_panel.context  # the app's services on a folder of their own (pytest fixture)
panel = the_panel.panel  # the Learning panel, shown (pytest fixture)

TIME, OVER = 0, 1  # the list's rows
HINT = 3  # Qt's tooltip role


class Timers:
    """The widgets of the "Each go lasts" row."""

    def __init__(self, panel: LearningPanel) -> None:
        ends = panel.findChild(QComboBox, "go_ends_combo")
        words = panel.findChild(QLabel, "go_seconds_words")
        box = panel.findChild(QDoubleSpinBox, "go_seconds_spin")
        assert ends is not None and words is not None and box is not None
        self.ends, self.words, self.box = ends, words, box

    def pick(self, row: int) -> None:
        """What picking a row of the list with the mouse does."""
        self.ends.setCurrentIndex(row)
        self.ends.activated.emit(row)

    def shown(self) -> tuple[str, str, str, float]:
        return self.ends.currentText(), self.words.text(), self.box.toolTip(), self.box.value()


FOR_A_TIME = ("for a set time", "of", "How long each go lasts before the AI starts the next one")
UNTIL_OVER = (
    "until the game is over",
    "or until nothing is paid for",
    "How long a go may be paid nothing before the AI starts the next one",
)


def test_the_list_and_its_hints(panel: LearningPanel) -> None:
    timers = Timers(panel)
    assert [timers.ends.itemText(row) for row in range(timers.ends.count())] == [
        "for a set time",
        "until the game is over",
    ]
    assert [timers.ends.itemData(row) for row in (TIME, OVER)] == ["time", "over"]
    assert [value for value, _, _ in GO_ENDS] == ["time", "over"]
    assert timers.ends.itemData(TIME, HINT) == (
        "Each go lasts as long as the box says. Then the AI starts the next one."
    )
    assert timers.ends.itemData(OVER, HINT) == (
        "Each go lasts until the game starts over, or until the AI has been paid nothing for "
        "as long as the box says (30 minutes at the most). The restart is the next go."
    )
    assert (GO_SECONDS_WORDS[False], GO_SECONDS_HINTS[False]) == FOR_A_TIME[1:]
    assert (GO_SECONDS_WORDS[True], GO_SECONDS_HINTS[True]) == UNTIL_OVER[1:]
    # With no game there is nothing to choose for.
    assert not timers.ends.isEnabled() and not timers.box.isEnabled()
    assert timers.shown()[:3] == FOR_A_TIME and panel.go_until_over is False


def test_choosing_until_the_game_is_over_and_back(
    panel: LearningPanel, context: AppContext
) -> None:
    timers = Timers(panel)
    connect(context)
    t = context.training
    fps = t.frames_per_step
    seconds = steps_to_seconds(DEFAULT_MAX_STEPS, fps)
    assert timers.ends.isEnabled() and timers.box.isEnabled()
    assert timers.shown() == (*FOR_A_TIME, seconds) and t.go_until_over is False
    timers.pick(OVER)
    assert t.go_until_over is True and panel.go_until_over is True
    assert t.go_limits == (until_over_most_steps(fps), DEFAULT_MAX_STEPS)
    # The seconds are what they were, and are now the wait with no pay.
    assert timers.shown() == (*UNTIL_OVER, seconds) and timers.box.isEnabled()
    assert panel.note_text == (
        "Each go now lasts until the game is over, or until it has been paid nothing "
        f"for {seconds:g} seconds."
    )
    # Other seconds are the same time in the service, and the note says what they are.
    timers.box.setValue(25)
    assert t.max_steps == seconds_to_steps(25, fps)
    assert t.go_limits == (until_over_most_steps(fps), seconds_to_steps(25, fps))
    assert timers.shown() == (*UNTIL_OVER, 25.0)
    assert panel.note_text == (
        "Each go now lasts until the game is over, or until it has been paid nothing "
        "for 25 seconds."
    )
    # Picking what is already chosen changes nothing and says nothing.
    saves = t.saves
    panel.choose_mode(Mode.INDEPENDENT)  # anything that leaves another note
    note = panel.note_text
    timers.pick(OVER)
    assert (t.saves, panel.note_text, t.go_until_over) == (saves, note, True)
    # And back: the seconds are the timer again.
    timers.pick(TIME)
    assert t.go_until_over is False and panel.go_until_over is False
    assert t.go_limits == (seconds_to_steps(25, fps), None)
    assert timers.shown() == (*FOR_A_TIME, 25.0)
    assert panel.note_text == "Each go now lasts 25 seconds." == GO_NOTES[False].format(25.0)
    timers.box.setValue(30)
    assert panel.note_text == "Each go now lasts 30 seconds."


def test_it_cannot_be_changed_while_training_runs(
    panel: LearningPanel, context: AppContext
) -> None:
    timers = Timers(panel)
    connect(context)
    t = context.training
    panel.choose_mode(Mode.INDEPENDENT)
    panel.start()
    assert t.status is Status.RUNNING
    assert not timers.ends.isEnabled() and not timers.box.isEnabled()
    # Asked for all the same, it is refused in the note and the list is as it was.
    panel.choose_go_until_over(True)
    assert "Stop training before changing" in panel.note_text
    assert t.go_until_over is False and timers.shown()[:3] == FOR_A_TIME
    panel.stop()
    assert timers.ends.isEnabled()
    timers.pick(OVER)
    assert t.go_until_over is True
    panel.start()
    assert not timers.ends.isEnabled() and timers.shown()[:3] == UNTIL_OVER
    panel.choose_go_until_over(False)
    assert "Stop training before changing" in panel.note_text
    assert t.go_until_over is True and timers.shown()[:3] == UNTIL_OVER
    panel.stop()


def test_the_list_shows_what_the_game_has(panel: LearningPanel, context: AppContext) -> None:
    timers = Timers(panel)
    connect(context)
    t = context.training
    timers.pick(OVER)
    timers.box.setValue(20)
    context.connection.disconnect()
    assert not timers.ends.isEnabled()
    # The same game again: its choice and its seconds.
    connect(context)
    assert t.go_until_over is True and timers.ends.isEnabled()
    assert timers.shown() == (*UNTIL_OVER, 20.0)
    # Another game has a timer, and the first still has its choice afterwards.
    context.connection.disconnect()
    context.connection.connect("fake", seed=9)
    assert t.go_until_over is False
    assert timers.shown() == (*FOR_A_TIME, steps_to_seconds(DEFAULT_MAX_STEPS, t.frames_per_step))
    context.connection.disconnect()
    connect(context)
    assert timers.shown() == (*UNTIL_OVER, 20.0)
    # A choice made behind the panel's back shows at its next refresh.
    t.set_go_until_over(False)
    panel.refresh()
    assert timers.shown() == (*FOR_A_TIME, 20.0) and panel.go_until_over is False


@pytest.mark.parametrize(
    ("ended_by", "words"),
    [
        ("no_gain", "nothing more was paid"),
        ("max_steps", "time was up"),
        ("reset", "the game started over"),
        ("stuck", "the screen stopped changing"),
        ("something new", "something new"),  # an end with no words is shown as it is
    ],
)
def test_the_goes_table_says_why_a_go_ended(ended_by: str, words: str) -> None:
    record = EpisodeRecord(7, Mode.INDEPENDENT, EpisodeResult(50, 200, 2.5, ended_by, 3), 900)
    assert go_cells(record) == ("7", "On its own", "50", "+2.5", words)
    assert ENDED_BY.get(ended_by, ended_by) == words
