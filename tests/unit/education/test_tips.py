"""T-607: the tips are complete, plain and well linked."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.education import PANEL_TIP, TIPS, TOPIC_KEYS, Tip, tip_for
from retro_ai_lab.ui.panels import PANELS

JARGON = ("algorithm", "policy", "q-value", "epsilon", "gradient", "hyperparameter", "discount")
TOPICS_FROM_DESIGN = {
    "artificial_intelligence",
    "machine_learning",
    "reinforcement_learning",
    "rewards",
    "neural_networks",
    "training",
    "decision_making",
    "exploration",
    "reward_exploitation",
    "hierarchical_learning",
}


def test_every_panel_and_every_design_topic_has_a_tip() -> None:
    assert set(PANEL_TIP) == {p.key for p in PANELS}
    assert set(TOPIC_KEYS) == TOPICS_FROM_DESIGN
    assert len(TIPS) == len(PANELS) + len(TOPICS_FROM_DESIGN)
    for key in PANEL_TIP.values():
        assert key in TIPS


def test_tips_are_plain_in_simple_words_and_fuller_in_advanced_ones() -> None:
    for tip in TIPS.values():
        assert isinstance(tip, Tip) and tip.key and tip.title.strip()
        assert tip.simple.endswith((".", "!")) and tip.advanced.endswith(".")
        assert 80 <= len(tip.simple) <= 400, tip.key
        assert len(tip.advanced) > len(tip.simple) * 0.8, tip.key
        for word in JARGON:
            assert word not in tip.simple.lower(), (tip.key, word)
        assert tip.text(False) == tip.simple and tip.text(True) == tip.advanced
        for other in tip.see_also:
            assert other in TIPS and other != tip.key, (tip.key, other)


def test_tip_for_and_unknown_keys() -> None:
    assert tip_for("panel.rewards").title == "Rewards"
    assert tip_for(PANEL_TIP["learning"]).key == "panel.learning"
    with pytest.raises(ConfigurationError, match="no tip called 'nope'"):
        tip_for("nope")
