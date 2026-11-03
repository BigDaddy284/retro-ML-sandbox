"""T-802: saving the connected AI, loading one back, deleting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from retro_ai_lab.comparison import SavedAiChanged, SavedAiService
from retro_ai_lab.core import ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.rewards.goals import Goal
from retro_ai_lab.rewards.model import RewardProfile
from retro_ai_lab.services import GoalChanged, RewardProfileChanged
from retro_ai_lab.storage import SavedAiStore
from retro_ai_lab.training import Brain, TrainingStateChanged
from tests.unit.training.test_service import Lab


@pytest.fixture
def ais(lab: Lab, tmp_path: Path) -> SavedAiService:
    return SavedAiService(
        lab.rewards, lab.goals, lab.training, lab.bus, store=SavedAiStore(tmp_path / "ais")
    )


def _jsonish(value: Any) -> Any:
    """As it comes back from disk: tuples become lists."""
    return json.loads(json.dumps(value))


def _first_behaviour(lab: Lab) -> str:
    profile = lab.rewards.profile
    assert profile is not None and profile.ids()
    return profile.ids()[0]


def test_disconnected_service_refuses_everything(ais: SavedAiService) -> None:
    assert not ais.enabled and ais.game_id is None and ais.names() == ()
    for call in (lambda: ais.save_as("A"), lambda: ais.load("A"), lambda: ais.delete("A")):
        with pytest.raises(UnsupportedCapabilityError, match="Connect to a game"):
            call()
    with pytest.raises(UnsupportedCapabilityError):
        ais.get("A")


def test_save_list_load_and_delete(lab: Lab, ais: SavedAiService) -> None:
    lab.connect()
    assert ais.enabled and ais.game_id == lab.training.game_id
    events: list[object] = []
    for kind in (SavedAiChanged, RewardProfileChanged, GoalChanged, TrainingStateChanged):
        lab.bus.subscribe(kind, events.append)
    t = lab.training
    t.train(2)
    behaviour = _first_behaviour(lab)
    lab.rewards.set_value(behaviour, 9.0)
    lab.goals.add_objective("Reach the first bend")
    objectives = len(lab.goals.goal)  # type: ignore[arg-type]
    saved = ais.save_as(" AI A ", notes="explorer")
    assert saved.name == "AI A" and saved.notes == "explorer" and saved.episodes == 2
    assert saved.rewards == lab.rewards.profile.to_dict()  # type: ignore[union-attr]
    assert saved.goal == lab.goals.goal.to_dict()  # type: ignore[union-attr]
    assert saved.brain == t.brain().to_dict()  # type: ignore[union-attr]
    assert ais.names() == ("AI A",)
    assert ais.get("AI A").to_dict() == _jsonish(saved.to_dict())
    assert events[-1] == SavedAiChanged(saved.game_id, "AI A", "saved")

    # change all three parts, save that as AI B
    t.train(1)
    lab.rewards.set_value(behaviour, -2.0)
    lab.goals.add_objective("Finish a lap")
    ais.save_as("AI B")
    assert ais.names() == ("AI A", "AI B")
    assert t.episodes == 3 and len(lab.goals.goal) == objectives + 1  # type: ignore[arg-type]
    assert lab.rewards.profile.get(behaviour).value == -2.0  # type: ignore[union-attr]

    # loading AI A brings back its rewards, goal and brain, on disk too
    events.clear()
    back = ais.load("AI A")
    assert back.to_dict() == _jsonish(saved.to_dict())
    assert lab.rewards.profile.get(behaviour).value == 9.0  # type: ignore[union-attr]
    assert len(lab.goals.goal) == objectives  # type: ignore[arg-type]
    assert t.episodes == 2 and len(t.history) == 2
    assert _jsonish(t.brain().to_dict()) == _jsonish(saved.brain)  # type: ignore[union-attr]
    assert lab.rewards.store.load(saved.game_id).to_dict() == saved.rewards  # type: ignore[union-attr]
    assert lab.goals.store.load(saved.game_id).to_dict() == saved.goal  # type: ignore[union-attr]
    assert t.store.load(saved.game_id) == _jsonish(saved.brain)  # type: ignore[union-attr]
    kinds = [type(e).__name__ for e in events]
    assert kinds.count("SavedAiChanged") == 1 and events[-1] == SavedAiChanged(
        saved.game_id, "AI A", "loaded"
    )
    assert any(isinstance(e, RewardProfileChanged) and e.reason == "loaded" for e in events)
    assert any(isinstance(e, GoalChanged) and e.reason == "loaded" for e in events)
    assert any(isinstance(e, TrainingStateChanged) and e.reason == "loaded" for e in events)
    # training carries on from the loaded brain
    t.train(1)
    assert t.episodes == 3 and [r.number for r in t.history] == [1, 2, 3]

    # saving again under a name replaces it
    ais.save_as("AI A", notes="second")
    assert ais.get("AI A").notes == "second" and ais.names() == ("AI A", "AI B")

    events.clear()
    assert ais.delete("AI B") and not ais.delete("AI B")
    assert ais.names() == ("AI A",)
    assert events == [SavedAiChanged(saved.game_id, "AI B", "deleted")]
    lab.connection.disconnect()
    assert not ais.enabled and ais.names() == ()


def test_refusals_change_nothing(lab: Lab, ais: SavedAiService) -> None:
    lab.connect()
    t = lab.training
    t.train(1)
    ais.save_as("AI A")
    t.train(1)
    with pytest.raises(ConfigurationError, match="Give the AI a name"):
        ais.save_as("   ")
    with pytest.raises(ConfigurationError, match="No saved AI called 'Nope'"):
        ais.load("Nope")
    t.start()
    with pytest.raises(ConfigurationError, match="Stop training before loading"):
        ais.load("AI A")
    t.stop()
    assert t.episodes == 2 and ais.names() == ("AI A",)
    # a saved AI whose brain does not fit the connected controls is refused whole
    saved = ais.get("AI A")
    odd = dict(saved.brain)
    odd["agent"] = dict(odd["agent"])
    odd["agent"]["space"] = {
        **odd["agent"]["space"],
        "actions": odd["agent"]["space"]["actions"][:2],
    }
    odd["agent"]["table"] = {"actions": 2, "rows": {}}
    ais.store.save(type(saved)(saved.name, saved.game_id, saved.rewards, saved.goal, odd).to_dict())
    behaviour = _first_behaviour(lab)
    lab.rewards.set_value(behaviour, 4.0)
    with pytest.raises(ConfigurationError):
        ais.load("AI A")
    assert t.episodes == 2 and lab.rewards.profile.get(behaviour).value == 4.0  # type: ignore[union-attr]


def test_replace_entry_points_keep_to_the_connected_game(lab: Lab) -> None:
    with pytest.raises(UnsupportedCapabilityError):
        lab.rewards.replace(RewardProfile("x"))
    with pytest.raises(UnsupportedCapabilityError):
        lab.goals.replace(Goal("x"))
    with pytest.raises(ConfigurationError, match="Connect to a game"):
        lab.training.replace_brain({})
    lab.connect()
    t = lab.training
    game_id = t.game_id
    assert game_id is not None
    with pytest.raises(ConfigurationError, match="not the connected game"):
        lab.rewards.replace(RewardProfile("other"))
    with pytest.raises(ConfigurationError, match="not the connected game"):
        lab.goals.replace(Goal("other"))
    agent = t.agent
    assert agent is not None
    with pytest.raises(ConfigurationError, match="not the connected game"):
        t.replace_brain(Brain("other", agent).to_dict())
    with pytest.raises(ConfigurationError, match="Bad brain"):
        t.replace_brain({"game_id": game_id})
    assert t.episodes == 0
    t.train(2)
    kept = t.brain()
    assert kept is not None
    data = kept.to_dict()
    t.train(1)
    assert t.episodes == 3
    brain = t.replace_brain(data)
    assert brain.episodes == 2 and t.episodes == 2 and t.total_steps == kept.total_steps
    assert t.store.load(game_id)["episodes"] == 2  # type: ignore[index]
    assert t.summary() == "Resting after 2 goes."
    t.start()
    with pytest.raises(ConfigurationError, match="Stop training"):
        t.replace_brain(data)
    t.stop()
