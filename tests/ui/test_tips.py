"""T-607: the "What's this?" window and the panel buttons that open it."""

from __future__ import annotations

import pytest
from pytestqt.qtbot import QtBot

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.education import PANEL_TIP, TIPS, TOPIC_KEYS, tip_for
from retro_ai_lab.ui.main_window import MainWindow
from retro_ai_lab.ui.panels import WHATS_THIS
from retro_ai_lab.ui.tips import TipDialog


@pytest.fixture
def dialog(qtbot: QtBot) -> TipDialog:
    d = TipDialog()
    qtbot.addWidget(d)
    return d


def test_dialog_shows_a_tip_and_switches_words(dialog: TipDialog) -> None:
    assert dialog.current_key is None and dialog.body_text == "" and not dialog.advanced
    assert set(dialog.topic_keys()) == set(TIPS) and len(dialog.topic_keys()) == len(TIPS)
    dialog.show_tip("rewards")
    tip = tip_for("rewards")
    assert dialog.isVisible() and dialog.current_key == "rewards"
    assert dialog.title_text == tip.title and dialog.body_text == tip.simple
    assert dialog.see_also_keys() == tip.see_also
    dialog.set_advanced(True)
    assert dialog.advanced and dialog.body_text == tip.advanced
    dialog.show_tip("exploration")  # the choice of words is remembered
    assert dialog.body_text == tip_for("exploration").advanced
    dialog.findChild(object, "tip_level").setChecked(False)  # type: ignore[attr-defined]
    assert not dialog.advanced and dialog.body_text == tip_for("exploration").simple
    dialog.set_advanced(False)  # idempotent
    assert dialog.body_text == tip_for("exploration").simple
    with pytest.raises(ConfigurationError):
        dialog.show_tip("nope")
    assert dialog.current_key == "exploration"


def test_see_also_and_topic_list_navigate(dialog: TipDialog) -> None:
    dialog.show_tip("panel.rewards")
    assert dialog.see_also_keys() == ("rewards", "reward_exploitation")
    dialog.findChild(object, "tip_see_also_1").click()  # type: ignore[attr-defined]
    assert dialog.current_key == "reward_exploitation"
    dialog.follow_see_also(0)
    assert dialog.current_key == "rewards"
    topics = dialog.findChild(object, "tip_topics")
    assert topics.currentItem().text() == tip_for("rewards").title  # type: ignore[attr-defined]
    row = dialog.topic_keys().index("neural_networks")
    topics.setCurrentRow(row)  # type: ignore[attr-defined]
    assert dialog.current_key == "neural_networks"
    assert dialog.see_also_keys() == ("machine_learning",)
    for key in TOPIC_KEYS:  # every topic renders without error
        dialog.show_tip(key)
        assert dialog.title_text == TIPS[key].title


def test_panel_buttons_open_the_shared_dialog(qtbot: QtBot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    dialog = window.tip_dialog
    assert isinstance(dialog, TipDialog) and not dialog.isVisible()
    for panel in window.panels:
        assert panel.help_button.text() == WHATS_THIS and panel.help_button.isVisible()
        panel.help_button.click()
        assert dialog.isVisible() and dialog.current_key == PANEL_TIP[panel.key]
        assert dialog.title_text == tip_for(PANEL_TIP[panel.key]).title
    assert window.findChildren(TipDialog) == [dialog]  # one window, however often opened
    dialog.set_advanced(True)
    window.explain("learning")
    assert dialog.body_text == tip_for("panel.learning").advanced
