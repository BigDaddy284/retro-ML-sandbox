"""T-801: a saved AI and its store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from retro_ai_lab.comparison import MAX_NAME_LENGTH, SavedAi, check_name
from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.learning import ActionSpace, QLearningAgent
from retro_ai_lab.rewards.goals import Goal
from retro_ai_lab.rewards.model import RewardProfile
from retro_ai_lab.storage import SavedAiStore
from retro_ai_lab.training import Brain

GAME = "toyracer-0"


def parts(game_id: str = GAME) -> dict[str, dict[str, Any]]:
    brain = Brain(game_id, QLearningAgent(ActionSpace.from_buttons(BUTTONS), seed=1), episodes=7)
    data = {
        "rewards": RewardProfile(game_id, personality="explorer").to_dict(),
        "goal": Goal(game_id, title="Finish a lap").to_dict(),
        "brain": {**brain.to_dict(), "total_steps": 300},
    }
    return json.loads(json.dumps(data))


def sample(name: str = "AI A", **extra: Any) -> SavedAi:
    return SavedAi(name, GAME, **parts(), **extra)


@pytest.fixture
def store(tmp_path: Path) -> SavedAiStore:
    return SavedAiStore(tmp_path / "ais")


def test_round_trip_and_summaries() -> None:
    saved = sample("  AI A ", notes="Likes exploring")
    assert saved.name == "AI A" and saved.game_id == GAME and saved.notes == "Likes exploring"
    assert saved.created.endswith("+00:00") and "T" in saved.created
    data = saved.to_dict()
    assert data["schema"] == 1 and data["kind"] == "saved_ai"
    back = SavedAi.from_dict(json.loads(json.dumps(data)))
    assert back == saved
    assert back.episodes == 7 and back.total_steps == 300
    assert back.algorithm == "q_table" and back.personality == "explorer"
    bare = SavedAi("B", GAME, {}, {}, {}, created="2026-01-01T00:00:00+00:00")
    assert (bare.episodes, bare.total_steps, bare.algorithm, bare.personality) == (0, 0, "", "")
    odd = SavedAi("C", GAME, {"personality": 3}, {}, {"episodes": True, "agent": "x"})
    assert (odd.episodes, odd.algorithm, odd.personality) == (0, "", "")


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("", "Give the AI a name"),
        ("   ", "Give the AI a name"),
        ("x" * (MAX_NAME_LENGTH + 1), f"{MAX_NAME_LENGTH} characters"),
        ("tab\there", "can't be shown"),
        (7, "must be text"),
    ],
)
def test_bad_names_are_refused(name: Any, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        check_name(name)
    with pytest.raises(ConfigurationError, match=match):
        SavedAi(name, GAME, **parts())


def test_bad_data_is_refused() -> None:
    good = sample().to_dict()
    cases: list[tuple[dict[str, Any], str]] = [
        ({**good, "schema": 2}, "newer version"),
        ({**good, "schema": "two"}, "not a number"),
        ({k: v for k, v in good.items() if k != "brain"}, "missing its brain"),
        (
            {k: v for k, v in good.items() if k not in ("rewards", "goal")},
            "missing its rewards, goal",
        ),
        ({k: v for k, v in good.items() if k != "name"}, "needs a name and a game"),
        ({**good, "rewards": []}, "rewards must be an object"),
        ({**good, "goal": {**good["goal"], "game_id": "other"}}, "goal belong to 'other'"),
        ({**good, "game_id": ""}, "must name its game"),
        ({**good, "notes": 1}, "Notes must be text"),
        ({**good, "created": ""}, "time it was saved"),
    ]
    for data, match in cases:
        with pytest.raises(ConfigurationError, match=match):
            SavedAi.from_dict(data)
    with pytest.raises(ConfigurationError, match="must be an object"):
        SavedAi.from_dict([])  # type: ignore[arg-type]


def test_store_keeps_games_apart_and_lists_names(store: SavedAiStore, tmp_path: Path) -> None:
    assert store.games() == () and store.names(GAME) == ()
    a = sample("AI A")
    b = sample("ai b")
    other = SavedAi("AI A", "other/game:1", **parts("other/game:1"))
    path = store.save(a.to_dict())
    assert path == store.path_for(GAME, "AI A") and path.is_file()
    store.save(b.to_dict())
    store.save(other.to_dict())
    assert store.games() == ("other/game:1", GAME)
    assert store.names(GAME) == ("AI A", "ai b")
    assert store.names("other/game:1") == ("AI A",)
    assert store.names("nope") == ()
    assert store.exists(GAME, "AI A") and not store.exists(GAME, "AI C")
    assert SavedAi.from_dict(store.load(GAME, "AI A")) == a
    assert SavedAi.from_dict(store.load("other/game:1", "AI A")) == other
    # saving again under the same name replaces it
    newer = SavedAi("AI A", GAME, notes="second", **parts())
    store.save(newer.to_dict())
    assert store.load(GAME, "AI A")["notes"] == "second"
    assert store.names(GAME) == ("AI A", "ai b")
    with pytest.raises(ConfigurationError, match="Only a saved AI's own data"):
        store.save({"name": "x", "game_id": GAME})
    assert store.delete(GAME, "AI A") and not store.delete(GAME, "AI A")
    assert store.names(GAME) == ("ai b",)
    with pytest.raises(ConfigurationError, match="No saved AI called 'AI A'"):
        store.load(GAME, "AI A")
    # foreign and damaged files in the folder are ignored
    (store.folder_for(GAME) / "notes.json").write_text('{"name": "x"}', "utf-8")
    (store.folder_for(GAME) / "broken.json").write_text("{", "utf-8")
    (store.folder_for(GAME) / "readme.txt").write_text("hi", "utf-8")
    assert store.names(GAME) == ("ai b",) and store.games() == ("other/game:1", GAME)
    assert not store.exists(GAME, "x")


def test_store_refuses_a_name_that_clashes_on_disk(store: SavedAiStore) -> None:
    store.save(sample("AI A").to_dict())
    assert store.path_for(GAME, "AI A") == store.path_for(GAME, "AI_A")
    with pytest.raises(ConfigurationError, match="too close to the saved AI 'AI A'"):
        store.save(sample("AI_A").to_dict())
    assert not store.exists(GAME, "AI_A") and store.names(GAME) == ("AI A",)
    with pytest.raises(ConfigurationError, match="No saved AI called 'AI_A'"):
        store.load(GAME, "AI_A")
    assert not store.delete(GAME, "AI_A")
    assert store.exists(GAME, "AI A")
    with pytest.raises(ConfigurationError, match="can't be turned into a file name"):
        store.save(sample("???").to_dict())
