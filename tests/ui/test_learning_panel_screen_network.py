"""T-1055: the screen network in the "how the AI learns" picker."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from retro_ai_lab.learning import (
    AUTO,
    DQN,
    Q_TABLE,
    SCREEN,
    DQNLearner,
    QLearningAgent,
    ScreenLearner,
    describe_algorithm,
)
from retro_ai_lab.training import Mode
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.learning_panel import AFRESH_QUESTION, LearningPanel


@pytest.fixture
def context(tmp_path: Path) -> AppContext:
    return AppContext(data_dir=tmp_path)


@pytest.fixture
def panel(qtbot: QtBot, context: AppContext) -> LearningPanel:
    p = LearningPanel(context)
    qtbot.addWidget(p)
    p.show()
    return p


def asking(panel: LearningPanel, answer: bool) -> list[str]:
    asked: list[str] = []

    def confirm(question: str) -> bool:
        asked.append(question)
        return answer

    panel.confirm = confirm  # type: ignore[method-assign]
    return asked


def test_the_picker_offers_the_screen_network_and_says_what_it_is(panel: LearningPanel) -> None:
    combo = panel.findChild(object, "algorithm_combo")
    assert combo is not None
    index = combo.findData(SCREEN)  # type: ignore[attr-defined]
    assert index >= 0
    assert combo.itemText(index) == "Screen network"  # type: ignore[attr-defined]
    label, sentence = describe_algorithm(SCREEN)
    assert combo.itemData(index, 3) == sentence and label == "Screen network"  # type: ignore[attr-defined]
    assert "Watches the screen itself" in sentence


def test_a_change_that_loses_what_was_learned_is_asked_first(
    panel: LearningPanel, context: AppContext
) -> None:
    context.connection.connect("fake", seed=3)
    t = context.training
    combo = panel.findChild(object, "algorithm_combo")
    assert combo is not None and isinstance(t.agent, QLearningAgent)
    # Nothing learned yet: nothing to lose, nothing to ask.
    asked = asking(panel, False)
    panel.choose_algorithm(SCREEN)
    assert asked == [] and isinstance(t.agent, ScreenLearner)
    panel.choose_algorithm(Q_TABLE)
    assert asked == [] and isinstance(t.agent, QLearningAgent)
    t.train(2)
    table = t.agent
    learned = table.updates
    assert learned > 0
    # Asked, and answered no: the brain is kept and the picker shows what it showed.
    panel.choose_algorithm(SCREEN)
    assert asked == [AFRESH_QUESTION] and "cannot be undone" in AFRESH_QUESTION
    assert "screen network" in AFRESH_QUESTION and "loses everything" in AFRESH_QUESTION
    assert panel.note_text == "Kept the brain."
    assert t.agent is table and table.updates == learned and t.algorithm == Q_TABLE
    assert panel.algorithm == Q_TABLE
    assert combo.currentData() == Q_TABLE  # type: ignore[attr-defined]
    # Between the table and the first network nothing is lost, so nothing is asked.
    panel.choose_algorithm(DQN)
    assert isinstance(t.agent, DQNLearner) and len(asked) == 1
    assert "keeping what it learned" in panel.note_text
    panel.choose_algorithm(AUTO)
    assert len(asked) == 1
    # Asked, and answered yes: it is the screen network, and the note says what happened.
    asked = asking(panel, True)
    panel.choose_algorithm(SCREEN)
    assert asked == [AFRESH_QUESTION] and isinstance(t.agent, ScreenLearner)
    assert panel.algorithm == SCREEN and t.agent.updates == 0
    assert panel.note_text == (
        "The AI moved from the neural network to the screen network, and starts learning "
        "afresh: what it had learned could not be taken along."
    )
    assert "screen network" in panel.status_text
    # Chosen again it is the same brain: nothing to ask.
    panel.choose_algorithm(SCREEN)
    assert len(asked) == 1
    # Once it has learned, leaving it is asked too.
    t.train(1)
    asked = asking(panel, False)
    panel.choose_algorithm(Q_TABLE)
    assert asked == [AFRESH_QUESTION] and isinstance(t.agent, ScreenLearner)
    assert panel.algorithm == SCREEN and panel.note_text == "Kept the brain."
    context.connection.disconnect()


def test_nothing_is_asked_while_it_trains_and_the_change_is_refused(
    panel: LearningPanel, context: AppContext
) -> None:
    context.connection.connect("fake", seed=3)
    t = context.training
    t.train(1)
    asked = asking(panel, True)
    panel.choose_mode(Mode.INDEPENDENT)
    panel.start()
    panel.choose_algorithm(SCREEN)
    assert asked == [] and "Stop training" in panel.note_text
    assert isinstance(t.agent, QLearningAgent)
    panel.stop()
    panel.choose_algorithm("magic")  # not a way of learning: refused, and nothing asked
    assert asked == [] and "magic" in panel.note_text
    context.connection.disconnect()
